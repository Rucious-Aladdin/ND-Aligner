import pytest
import torch
import torchaudio

from tts.models.modules.spk_encoder import SpeakerEncoder


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_spk_encoder(visualize: bool):
    device = "cuda"
    print(f"\n🏗️ Initializing SpeakerEncoder on {device}...")
    encoder = SpeakerEncoder(device=device)
    encoder.eval()

    # 1. Single sample test (16kHz, 2 seconds)
    batch_size = 1
    sample_rate = 16000
    duration = 2
    waveform = torch.randn(batch_size, sample_rate * duration).to(device)

    print(f"🚀 Running inference with input shape: {waveform.shape}")
    with torch.no_grad():
        emb = encoder(waveform)

    print(f"✅ Speaker embedding shape: {emb.shape}")
    assert emb.shape == (batch_size, 192), f"Expected (1, 192), got {emb.shape}"
    assert not torch.isnan(emb).any(), "Embedding contains NaN"

    # 2. Batch sample test
    batch_size = 4
    waveform_batch = torch.randn(batch_size, sample_rate * 1).to(device)
    with torch.no_grad():
        emb_batch = encoder(waveform_batch)

    assert emb_batch.shape == (
        batch_size,
        192,
    ), f"Expected ({batch_size}, 192), got {emb_batch.shape}"
    print(f"✅ Batch speaker embedding shape: {emb_batch.shape}")
