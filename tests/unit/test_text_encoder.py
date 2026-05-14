import pytest
import torch

from tts.models.modules.film_text_encoder import TextEncoder
from tts.models.utils.sequence_mask import sequence_mask


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_text_encoder(visualize: bool = True):
    device = torch.device("cuda")
    n_vocab = 100
    in_channels = 128
    out_channels = 128
    hidden_channels = 256
    cond_dim = 256
    num_layers = 3
    dropout = 0.1

    B, T_text = 4, 300

    model = TextEncoder(
        n_vocab=n_vocab,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        cond_dim=cond_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model.eval()

    x_lengths = torch.randint(low=10, high=T_text, size=(B,)).to(device)
    x = torch.randint(low=0, high=n_vocab, size=(B, T_text)).to(device)
    cond = torch.randn(B, cond_dim).to(device)
    text_mask = sequence_mask(x_lengths, T_text).unsqueeze(1).to(device).to(torch.float32)

    with torch.no_grad():
        x_out = model(x, cond, text_mask)

    if visualize:
        print("\n=== TextEncoder Output Shapes ===")
        print(f"Input x shape:      {x.shape}")
        print(f"Input lengths:      {x_lengths.tolist()}")
        print("-" * 30)
        print(f"Output x_out shape: {x_out.shape}")  # [B, H, T]
        print(f"Mask shape:         {text_mask.shape}")  # [B, 1, T]

        mask_sum = (text_mask == 0).sum().item()
        if mask_sum > 0:
            # x_out is (B, H, T), mask is (B, 1, T)
            masked_x_val = (x_out * (1 - text_mask)).abs().sum().item()

            print("-" * 30)
            print(f"Masking Efficiency Check:")
            print(f"Sum of values in masked area (x_out):    {masked_x_val:.6f}")

            if masked_x_val < 1e-5:
                print("Result: ✅ Masking is applied correctly.")
            else:
                print("Result: ❌ Masking might be leaking values.")
