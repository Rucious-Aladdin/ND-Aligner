import hashlib
import os
from typing import override

import librosa
import torch
from torch.utils.data import Dataset

from tts.audio.mel_spectrogram import MelSpecExtractor
from tts.config.ndaligner.data_config import DataConfig
from tts.config.ndaligner.model_config import SPK_COND_DIM
from tts.tokenizer.text_tokenizer import TextTokenizer

from .data_types import TTSDatasetInstance, TTSItem


class TTSDataset(Dataset[TTSDatasetInstance]):
    def __init__(
        self,
        items: list[TTSItem],
        config: DataConfig,
    ):
        self.items = items
        self.cache_dir = config.dataset.data_cache_dir
        self.sr = config.audio.sr

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer = TextTokenizer()
        self.mel_extractor = MelSpecExtractor(
            sr=config.audio.sr,
            n_mels=config.audio.n_mels,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            win_length=config.audio.win_length,
            fmin=config.audio.f_min,
            fmax=config.audio.f_max,
        )

    def __len__(self) -> int:
        return len(self.items)

    @override
    def __getitem__(self, idx: int) -> TTSDatasetInstance:
        item = self.items[idx]

        # Use MD5 hash of audio path as unique identifier for caching.
        # Cache stores only tensor payloads so metadata changes do not invalidate cache.
        item_id = hashlib.md5(item.audio_path.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{item_id}.pt") if self.cache_dir else None

        if cache_path and os.path.exists(cache_path):
            try:
                x, y, cond = torch.load(cache_path, weights_only=True)

                return TTSDatasetInstance(
                    text=x,
                    spec=y,
                    cond=cond,
                    wav_path=item.audio_path,
                    script=item.text,
                    utt_id=item.utt_id,
                    spk_id=item.spk_id,
                    dataset=item.dataset,
                )
            except Exception as e:
                print(f"Error loading cache for {item.audio_path}: {e}. Re-processing...")

        # 1. Tokenize text
        x = self.tokenizer(item.text).squeeze(0)

        # 2. Extract Mel-spectrogram
        wav_numpy, sr = librosa.load(item.audio_path, sr=None)
        if sr != self.sr:
            wav_numpy = librosa.resample(wav_numpy, orig_sr=sr, target_sr=self.sr)

        wav_tensor = torch.from_numpy(wav_numpy).float()
        with torch.no_grad():
            y = self.mel_extractor(wav_tensor).squeeze(0)

        # 3. Load Speaker Embedding
        if item.spk_path and os.path.exists(item.spk_path):
            try:
                cond = torch.load(item.spk_path, weights_only=True).float()
            except Exception as e:
                print(f"Error loading speaker embedding {item.spk_path}: {e}. Using zero vector.")
                cond = torch.zeros(SPK_COND_DIM, dtype=torch.float32)
        else:
            cond = torch.zeros(SPK_COND_DIM, dtype=torch.float32)

        data_to_cache = (x, y, cond)

        if cache_path:
            torch.save(data_to_cache, cache_path)

        return TTSDatasetInstance(
            text=x,
            spec=y,
            cond=cond,
            wav_path=item.audio_path,
            script=item.text,
            utt_id=item.utt_id,
            spk_id=item.spk_id,
            dataset=item.dataset,
        )

    @property
    def spec_pad_value(self) -> float:
        return self.mel_extractor.pad_value
