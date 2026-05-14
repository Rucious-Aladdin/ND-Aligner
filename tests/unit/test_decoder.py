import pytest
import torch

from tts.models.modules.mel_decoder import MelDecoder


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_decoder(visualize: bool = True):
    device = torch.device("cuda")
    B, T_max = 4, 150

    num_mels = 80
    input_dim = 128
    encoder_dim = 256
    num_encoder_layers = 2
    cond_in_channels = 64

    model = MelDecoder(
        num_mels=num_mels,
        input_dim=input_dim,
        hidden_dim=encoder_dim,
        num_layers=num_encoder_layers,
        cond_in_channels=cond_in_channels,
    ).to(device)
    model.eval()

    input_lengths = torch.tensor([150, 120, 100, 50], dtype=torch.long, device=device)
    inputs = torch.randn(B, T_max, input_dim, device=device)
    cond = torch.randn(B, cond_in_channels, device=device)

    idx = torch.arange(T_max, device=device).unsqueeze(0)
    mask = (idx < input_lengths.unsqueeze(1)).float()

    with torch.no_grad():
        mel_outputs, out_lengths = model(inputs, input_lengths, cond, mask)

    if visualize:
        print("\n=== MelDecoder Output Shapes ===")
        print(f"Inputs shape:            {inputs.shape}")
        print(f"Condition shape:         {cond.shape}")
        print(f"Mask shape:              {mask.shape}")
        print(f"Input lengths:           {input_lengths.tolist()}")
        print("-" * 35)
        print(f"Output mel shape:        {mel_outputs.shape}")  # [B, T_max, 80]
        print(
            f"Output lengths:          {out_lengths.tolist()}"
        )  # Subsampling이 없으므로 input과 동일해야 함

        inverse_mask = (1.0 - mask).unsqueeze(-1)
        masked_mel_val = (mel_outputs * inverse_mask).abs().sum().item()

        print("-" * 35)
        print("Masking Efficiency Check:")
        print(f"Sum of values in masked area (mel_outputs): {masked_mel_val:.6f}")

        if masked_mel_val < 1e-5:
            print("Result: ✅ Masking is applied perfectly! No leakage.")
        else:
            print(
                "Result: ❌ WARNING! Masking is leaking values. Check the linear bias or attention mask."
            )

        assert masked_mel_val < 1e-5, f"Mask leakage detected: {masked_mel_val}"
        assert torch.equal(input_lengths, out_lengths), "Output lengths mismatch!"
