import os

import pytest
import soundfile as sf
import torch
import torchaudio

from tests.utils.visualize_2d_tensor import visualize_2d_tensor
from tts.audio.mel_spectrogram import MelSpecExtractor
from tts.config.data_config import AudioConfig


@pytest.fixture(scope="module")
def mel_extractor():
    config = AudioConfig()
    return MelSpecExtractor(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mel_spec_extractor(
    mel_extractor: MelSpecExtractor,
    visualize: bool,
):
    """Loads 3 example WAV files, extracts Mel-spectrograms, verifies dimensions, and visualizes the results."""
    mel_extractor = mel_extractor.to("cuda")
    config = AudioConfig()

    for i in range(1, 4):
        file_path = f"tests/asset/example{i}.wav"
        assert os.path.exists(file_path), f"Test file not found: {file_path}"

        data, sr = sf.read(file_path, dtype="float32")

        if data.ndim == 1:  # Mono
            audio = torch.from_numpy(data).unsqueeze(0)
        else:  # Stereo
            audio = torch.from_numpy(data).t()

        if sr != config.sr:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=config.sr)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        audio = audio.to("cuda")
        mel_spec = mel_extractor(audio)

        print(f"\n[example{i}.wav] Input audio shape: {audio.shape}, Sample Rate: {config.sr}")
        print(f"[example{i}.wav] Output Mel shape: {mel_spec.shape}")

        assert isinstance(mel_spec, torch.Tensor), "Result is not a Tensor."
        assert mel_spec.device.type == "cuda", "Result tensor is not on GPU."
        assert mel_spec.dim() == 3, f"Expected shape (B, n_mels, T), got {mel_spec.dim()}D"
        assert mel_spec.size(0) == 1, f"Expected batch size 1, got {mel_spec.size(0)}"
        assert (
            mel_spec.size(1) == config.n_mels
        ), f"Expected {config.n_mels} mel channels, got {mel_spec.size(1)}"

        assert not torch.isnan(mel_spec).any(), f"NaN found in example{i}.wav conversion result."
        assert not torch.isinf(mel_spec).any(), f"Inf found in example{i}.wav conversion result."

        if visualize:
            visualize_2d_tensor(
                tensor=mel_spec,
                batch_idx=0,
                title=f"Mel-Spectrogram: example{i}.wav",
                xlabel="Time Frames",
                ylabel="Mel Channels",
                cmap="magma",
                origin="lower",
                save_path=f"test_mel_extractor_example{i}.png",
            )
