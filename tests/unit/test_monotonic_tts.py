import pytest
import torch

from tts.config.model_config import MonotonicTTSConfigs
from tts.models.init_monotonic_tts import init_monotonic_tts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_monotonic_tts_synthesizer_train_forward(visualize: bool = True):
    torch.manual_seed(0)
    device = torch.device("cuda")

    config = MonotonicTTSConfigs()
    model = init_monotonic_tts(
        config=config,
        load_spec_encoder=True,
        load_aligner=True,
    ).to(device)
    model.train()

    B = 4
    T_text_max, T_mel_max = 50, 200
    n_vocab = config.txt_enc.n_vocab
    n_mels = config.dec.num_mels
    D_cond = config.aligner.dim_cond

    x_lengths = torch.tensor([50, 40, 30, 20], dtype=torch.long, device=device)
    y_lengths = torch.tensor([200, 160, 120, 80], dtype=torch.long, device=device)

    x = torch.randint(0, n_vocab, (B, T_text_max), dtype=torch.long, device=device)
    y = torch.randn(B, n_mels, T_mel_max, device=device)
    cond = torch.randn(B, D_cond, device=device)

    x_idx = torch.arange(T_text_max, device=device).unsqueeze(0)
    y_idx = torch.arange(T_mel_max, device=device).unsqueeze(0)
    x_mask = (x_idx < x_lengths.unsqueeze(1)).float()
    y_mask = (y_idx < y_lengths.unsqueeze(1)).float()

    x = x * x_mask.long()
    y = y * y_mask.unsqueeze(1)

    out = model(
        x=x,
        x_lengths=x_lengths,
        y=y,
        y_lengths=y_lengths,
        cond=cond,
    )

    if visualize:
        print("\n" + "=" * 55)
        print("🚀 MonotonicTTSSynthesizer Forward Pipeline Output 🚀")
        print("=" * 55)
        print(f"Inputs:")
        print(f"  - Text (x) shape:       {x.shape}")
        print(f"  - Mel (y) shape:        {y.shape}")
        print(f"  - Cond shape:           {cond.shape}")
        print("-" * 55)
        print(f"Outputs:")
        print(f"  - Duration Target:      {out.dur.shape}      # (B, T_text)")
        print(f"  - Attention Map:        {out.attn.shape} # (B, T_mel, T_text)")
        print(f"  - Predicted Mel:        {out.mel_hat.shape}   # (B, T_mel, n_mels)")
        print(f"  - Mel Loss:             {out.mel_loss.item():.4f}             # (Scalar)")
        print(f"  - Dur Loss (SDP):       {out.dur_loss.item():.4f}             # (Scalar)")
        print(f"  - Aux Mel Loss:         {out.mel_aux_loss.item():.4f}         # (Scalar)")
        print(f"  - Align NLL:            {out.align_nll.item():.4f}            # (Scalar)")

        inverse_y_mask = 1.0 - y_mask  # (B, T_mel)
        masked_mel_val = (out.mel_hat * inverse_y_mask.unsqueeze(-1)).abs().sum().item()

        print("-" * 55)
        print(f"Masking Efficiency Check:")
        print(f"Sum of values in padded mel area: {masked_mel_val:.6f}")

        if masked_mel_val < 1e-5:
            print("Result: ✅ Integration PERFECT! No padding leakage detected.")
        else:
            print("Result: ❌ WARNING! Mask leakage in the final output.")

        from ..utils.visualize_2d_map import visualize_2d_map
        from ..utils.visualize_2d_tensor import visualize_2d_tensor

        batch_idx = 1
        s_len = int(y_lengths[batch_idx].item())
        t_len = int(x_lengths[batch_idx].item())

        print("-" * 55)
        print(f"Visualizing Batch {batch_idx} (Mel Len: {s_len}, Text Len: {t_len})")

        valid_attn = out.attn[batch_idx, :s_len, :t_len].detach().cpu()
        visualize_2d_map(
            valid_attn,
            title="Aligned Attention Map",
            save_path="test_monotonic_tts_attn.png",
        )

        valid_mel_hat = out.mel_hat[batch_idx, :s_len, :].detach().cpu().transpose(0, 1)
        visualize_2d_tensor(
            tensor=valid_mel_hat,
            title="Predicted Mel-Spectrogram",
            xlabel="Time (Frames)",
            ylabel="Mel Channels",
            origin="lower",
            save_path="test_monotonic_tts_mel_hat.png",
        )

        valid_mel_gt = y[batch_idx, :, :s_len].detach().cpu()
        visualize_2d_tensor(
            tensor=valid_mel_gt,
            title="Ground Truth Mel-Spectrogram",
            xlabel="Time (Frames)",
            ylabel="Mel Channels",
            origin="lower",
            save_path="test_monotonic_tts_mel_gt.png",
        )

    assert out.dur.shape == (B, T_text_max)
    # assert out.attn.shape == (B, T_mel_max, T_text_max)
    assert out.mel_hat.shape == (B, T_mel_max, n_mels)
    if out.mel_hat_aux is not None:
        assert out.mel_hat_aux.shape == (B, T_mel_max, n_mels)
    assert out.mel_loss.dim() <= 1
    assert out.dur_loss.dim() <= 1
    assert out.mel_aux_loss.dim() <= 1
    assert out.align_nll.dim() <= 1
    assert masked_mel_val < 1e-5, f"Mask leakage detected: {masked_mel_val}"
