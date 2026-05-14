import dataclasses
import os

import pytest
import torch
import torchaudio
import soundfile as sf
from scipy.io.wavfile import write

from tts.audio.mel_spectrogram import MelSpecExtractor
from tts.config.data_config import AudioConfig
from tts.config.model_config import HifiGANVocoderConfigs
from tts.models.modules.hifigan_vocoder import Generator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_hifigan_vocoder(visualize: bool):
    device = torch.device("cuda")
    config = HifiGANVocoderConfigs()
    audio_config = AudioConfig()

    # Check if files exist
    if not os.path.exists(config.config_path) or not os.path.exists(config.ckpt_path):
        pytest.skip(f"HiFi-GAN config or checkpoint not found at {config.config_path} or {config.ckpt_path}")

    print(f"\n🏗️ Initializing HiFi-GAN Generator from {config.config_path}...")
    vocoder = Generator.from_config_path(
        config_path=config.config_path,
        ckpt_path=config.ckpt_path,
        device=device
    )
    vocoder.eval()
    vocoder.remove_weight_norm()

    # 1. Dummy Mel Spectrogram Test
    B = 1
    n_mels = 80
    T_mel = 100
    mel = torch.randn(B, n_mels, T_mel).to(device)

    print(f"🚀 Running inference with dummy mel shape: {mel.shape}")
    with torch.no_grad():
        audio = vocoder(mel)

    hop_size = 256
    expected_length = T_mel * hop_size

    print(f"✅ Generated audio shape: {audio.shape}")
    assert audio.dim() == 3
    assert audio.size(2) == expected_length
    assert not torch.isnan(audio).any()

    # 2. Multi-batch Test
    batch_size = 4
    mel_batch = torch.randn(batch_size, n_mels, T_mel).to(device)
    with torch.no_grad():
        audio_batch = vocoder(mel_batch)
    
    assert audio_batch.shape == (batch_size, 1, expected_length)
    print(f"✅ Batch generated audio shape: {audio_batch.shape}")

    # 3. Real WAV files Test (Reconstruction)
    print("\n🎧 Running reconstruction test with real WAV files...")
    mel_extractor = MelSpecExtractor(audio_config).to(device)
    
    for i in range(1, 4):
        wav_path = f"tests/asset/example{i}.wav"
        if not os.path.exists(wav_path):
            print(f"⚠️ {wav_path} not found, skipping...")
            continue
            
        # Load and preprocess using soundfile
        audio_data, sr = sf.read(wav_path)
        if audio_data.ndim == 1:
            wav = torch.FloatTensor(audio_data).unsqueeze(0)
        else:
            wav = torch.FloatTensor(audio_data[:, 0]).unsqueeze(0)

        if sr != audio_config.sr:
            wav = torchaudio.transforms.Resample(sr, audio_config.sr)(wav)
        
        wav = wav.to(device)
        
        # Extract mel
        with torch.no_grad():
            mel_real = mel_extractor(wav)
            # Generate
            audio_recon = vocoder(mel_real)
            
        print(f"[{wav_path}] Input shape: {wav.shape}, Mel shape: {mel_real.shape}, Output shape: {audio_recon.shape}")
        
        assert audio_recon.dim() == 3
        # Length might differ slightly due to padding in stft, but should be roughly proportional
        # MelSpecExtractor uses reflect padding which might affect exact length slightly
        # but Generator should output mel_len * 256
        assert audio_recon.size(2) == mel_real.size(2) * 256
        
        if visualize:
            from tests.utils.visualize_1d_tensor import visualize_1d_tensor
            
            save_png = f"test_hifigan_reconstruction_example{i}.png"
            visualize_1d_tensor(
                tensor=audio_recon,
                batch_idx=0,
                title=f"Reconstructed: example{i}.wav",
                save_path=save_png
            )
            
            # Save as WAV
            save_wav = f"test_hifigan_reconstruction_example{i}.wav"
            # Scale back to int16 if needed, or keep as float32
            # HiFi-GAN output is typically float32 in [-1, 1]
            wav_np = audio_recon.squeeze().cpu().numpy()
            write(save_wav, audio_config.sr, wav_np)
            print(f"📊 Saved reconstruction plot to {save_png} and audio to {save_wav}")

    if visualize:
        from tests.utils.visualize_1d_tensor import visualize_1d_tensor
        visualize_1d_tensor(
            tensor=audio,
            batch_idx=0,
            title="HiFi-GAN Generated Audio (Dummy)",
            save_path="test_hifigan_vocoder_dummy_wav.png"
        )
