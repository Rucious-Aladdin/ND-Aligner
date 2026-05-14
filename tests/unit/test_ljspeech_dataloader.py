import pytest
import torch
from torch.utils.data import DataLoader

from tts.config.data_config import DataConfig
from tts.data.data_types import TTSBatch
from tts.data.ljspeech_dataset import LJSpeechDataset, TTSCollate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_ljspeech_dataloader(visualize: bool):
    # 1. Initialize configuration and dataset
    config = DataConfig()
    dataset = LJSpeechDataset(config)

    pad_value = dataset.spec_pad_value

    # 2. Initialize DataLoader with custom collate function
    batch_size = 4  # Reduced batch size for testing efficiency
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TTSCollate(spec_pad_value=pad_value),
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # 3. Fetch a single batch
    batch: TTSBatch = next(iter(dataloader))

    # 4. Extract shapes for validation
    B, T_text_max = batch.text.shape
    B_mel, n_mels, T_mel_max = batch.spec.shape  # pyright: ignore[reportUnusedVariable]

    # 5. Print out the shapes
    if visualize:
        print("\n" + "=" * 55)
        print("📦 TTS DataLoader Output Shapes 📦")
        print("=" * 55)
        print(f"Text padded shape:      {batch.text.shape}  # (B, T_text_max)")
        print(f"Text lengths shape:     {batch.text_lengths.shape}       # (B,)")
        print(f"Spec padded shape:      {batch.spec.shape} # (B, n_mels, T_mel_max)")
        print(f"Spec lengths shape:     {batch.spec_lengths.shape}       # (B,)")
        print(f"Condition vector shape: {batch.cond.shape}    # (B, cond_dim)")
        print("-" * 55)

    # 6. Validate Padding (Leakage Check)
    device = batch.text.device

    # Check text padding (should be exactly 0)
    text_idx = torch.arange(T_text_max, device=device).unsqueeze(0)
    text_mask = text_idx < batch.text_lengths.unsqueeze(1)
    inverse_text_mask = ~text_mask

    # Extract padded regions only
    padded_text_elements = batch.text[inverse_text_mask]
    text_pad_correct = torch.all(padded_text_elements == 0).item()

    # Check mel-spectrogram padding (should be exactly pad_value)
    mel_idx = torch.arange(T_mel_max, device=device).unsqueeze(0)
    mel_mask = mel_idx < batch.spec_lengths.unsqueeze(1)
    inverse_mel_mask = ~mel_mask

    # Transpose spec to (B, T_mel, n_mels) to apply the mask naturally
    padded_spec_elements = batch.spec.transpose(1, 2)[inverse_mel_mask]
    # Use allclose because of floating point precision
    mel_pad_correct = torch.allclose(
        padded_spec_elements,
        torch.tensor(pad_value, dtype=padded_spec_elements.dtype, device=device),
        atol=1e-4,
    )

    if visualize:
        print("Padding Verification Check:")
        print(f"  - Text Padding (== 0):          {'✅ Pass' if text_pad_correct else '❌ Fail'}")
        print(
            f"  - Mel Padding (== {pad_value:.4f}): {'✅ Pass' if mel_pad_correct else '❌ Fail'}"
        )

        # 7. Visualizing the first mel-spectrogram in the batch using the updated function
        from ..utils.visualize_2d_tensor import visualize_2d_tensor

        valid_len = batch.spec_lengths[0].item()

        print("-" * 55)
        print(f"Visualizing Batch Index 0 (Valid Length: {valid_len} / Padded to: {T_mel_max})")

        # Seamlessly pass the 3D tensor directly to the new visualizer
        visualize_2d_tensor(
            tensor=batch.spec,
            batch_idx=0,
            title=f"Dataloader Output Mel (Pad Value: {pad_value:.2f})",
            xlabel="Time (Frames)",
            ylabel="Mel Channels",
            origin="lower",
            save_path="test_ljspeech_dataloader_mel.png",
        )

    # 8. Formal Assertions
    assert B == batch_size
    assert B_mel == batch_size
    assert text_pad_correct, "Text padding contains non-zero values!"
    assert mel_pad_correct, f"Mel-spectrogram padding contains values other than {pad_value}!"
