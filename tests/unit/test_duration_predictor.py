import pytest
import torch

from tts.models.modules.duration_predictor import StochasticDurationPredictor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_duration_predictor(visualize: bool = True):
    torch.manual_seed(0)
    device = torch.device("cuda")

    B, T_max = 4, 50
    in_channels = 128  # Text Encoder 출력 차원과 동일
    filter_channels = 128
    kernel_size = 3
    p_dropout = 0.1
    n_flows = 4
    gin_channels = 256  # Speaker/Style Cond 차원

    x_lengths = torch.tensor([50, 40, 30, 20], dtype=torch.long, device=device)
    idx = torch.arange(T_max, device=device).unsqueeze(0)
    x_mask = (idx < x_lengths.unsqueeze(1)).unsqueeze(1).float()  # (B, 1, T_max)

    x = torch.randn(B, in_channels, T_max, device=device) * x_mask
    cond = torch.randn(B, gin_channels, 1, device=device)
    dur_target = torch.randint(1, 10, (B, 1, T_max), device=device).float() * x_mask

    model = StochasticDurationPredictor(
        in_channels=in_channels,
        filter_channels=filter_channels,
        kernel_size=kernel_size,
        p_dropout=p_dropout,
        n_flows=n_flows,
        gin_channels=gin_channels,
    ).to(device)
    model.eval()

    with torch.no_grad():
        loss = model(
            x=x,
            x_mask=x_mask,
            dur_target=dur_target,
            cond=cond,
            reverse=False,
        )

    with torch.no_grad():
        logw = model(
            x=x,
            x_mask=x_mask,
            dur_target=None,
            cond=cond,
            reverse=True,
            noise_scale=0.8,
        )

        w = torch.exp(logw) * x_mask
        dur_pred = torch.ceil(w).squeeze(1).long()

    print("\n=== StochasticDurationPredictor Output Shapes ===")
    print(f"Input x shape:         {x.shape}")
    print(f"Mask shape:            {x_mask.shape}")
    print(f"Cond shape:            {cond.shape}")
    print("-" * 47)
    print(f"Training Loss shape:   {loss.shape}     # [B]")
    print(f"Inference logw shape:  {logw.shape} # [B, 1, T_max]")
    print(f"Inference dur_pred:    {dur_pred.shape} # [B, T_max]")

    inverse_mask = 1.0 - x_mask
    masked_logw_val = (logw * inverse_mask).abs().sum().item()

    print("-" * 47)
    print(f"Masking Efficiency Check:")
    print(f"Sum of values in masked area (logw): {masked_logw_val:.6f}")

    if masked_logw_val < 1e-5:
        print("Result: ✅ Masking is applied perfectly! No leakage in Flow.")
    else:
        print("Result: ❌ WARNING! Masking is leaking values in Normalizing Flows.")

    batch_idx = 1
    valid_len = int(x_lengths[batch_idx].item())
    print("-" * 47)
    print(f"Predicted Durations (Batch {batch_idx}, Valid Len {valid_len}):")
    print(dur_pred[batch_idx, :valid_len].tolist())

    assert loss.shape == (B,)
    assert logw.shape == (B, 1, T_max)
    assert dur_pred.shape == (B, T_max)
    assert masked_logw_val < 1e-5, f"Mask leakage detected: {masked_logw_val}"
