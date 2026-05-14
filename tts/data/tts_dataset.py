import hashlib
import os
from typing import override

import librosa
import torch
from torch.utils.data import Dataset

from tts.audio.mel_spectrogram import MelSpecExtractor
from tts.config.stage1.data_config import DataConfig
from tts.config.stage1.model_config import SPK_COND_DIM
from tts.tokenizer.text_tokenizer import TextTokenizer

from .data_types import TTSBatch, TTSItem


class TTSDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
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
        self.mel_extractor = MelSpecExtractor(config=config.audio)

    def __len__(self) -> int:
        return len(self.items)

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        item = self.items[idx]

        # Use MD5 hash of audio path as unique identifier for caching
        item_id = hashlib.md5(item.audio_path.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{item_id}.pt") if self.cache_dir else None

        if cache_path and os.path.exists(cache_path):
            try:
                x, y, cond = torch.load(cache_path, weights_only=True)
                return x, y, cond, item.text
            except Exception as e:
                print(f"Error loading cache for {item.audio_path}: {e}. Re-processing...")

        # 1. Tokenize text
        x = self.tokenizer(item.text).squeeze(0)

        # 2. Extract Mel-spectrogram
        wav_numpy, sr = librosa.load(item.audio_path, sr=None)
        if sr != self.sr:
            # Attempt to resample if it doesn't match (should match if preprocessed correctly)
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

        return x, y, cond, item.text

    @property
    def spec_pad_value(self) -> float:
        return self.mel_extractor.pad_value
