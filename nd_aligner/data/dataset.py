import hashlib
import os
from typing import override

import librosa
import torch
from torch.utils.data import Dataset

from nd_aligner.audio.linear_spectrogram import LinearSpecExtractor
from nd_aligner.audio.mel_spectrogram import MelSpecExtractor
from nd_aligner.config.ndaligner.data_config import DataConfig
from nd_aligner.tokenizer.load_tokenizer import load_tokenizer

from .data_types import TrainDatasetInstance, TrainItem


class TTSDataset(Dataset[TrainDatasetInstance]):
    def __init__(
        self,
        items: list[TrainItem],
        config: DataConfig,
    ):
        self.items = items
        self.cache_dir = config.dataset.data_cache_dir
        self.sr = config.audio.sr
        self.spk_embedding_dim = config.preprocess.spk_embedding_dim

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer = load_tokenizer(
            tokenizer_type=config.dataset.tokenizer_type,
            fastspeech2_lexicon_path=config.dataset.fastspeech2_lexicon_path,
        )

        self.feature_type = config.audio.feature_type

        self.mel_extractor = MelSpecExtractor(
            sr=config.audio.sr,
            n_mels=config.audio.n_mels,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            win_length=config.audio.win_length,
            fmin=config.audio.f_min,
            fmax=config.audio.f_max,
        )

        if config.audio.feature_type == "mel":
            self.spec_extractor = self.mel_extractor

        elif config.audio.feature_type == "linspec":
            self.spec_extractor = LinearSpecExtractor(
                sr=config.audio.sr,
                n_fft=config.audio.n_fft,
                hop_length=config.audio.hop_length,
                win_length=config.audio.win_length,
                fmax=config.audio.f_max,
            )

        else:
            raise ValueError(f"Unsupported feature_type: {config.audio.feature_type!r}")

    def __len__(self) -> int:
        return len(self.items)

    @override
    def __getitem__(self, idx: int) -> TrainDatasetInstance:
        item = self.items[idx]

        # Use MD5 hash of audio path as unique identifier for caching.
        # Cache stores only tensor payloads so metadata changes do not invalidate cache.
        item_id = hashlib.md5(item.audio_path.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{item_id}.pt") if self.cache_dir else None

        if cache_path and os.path.exists(cache_path):
            try:
                x, y, y_recon, cond = torch.load(cache_path, weights_only=True)

                return TrainDatasetInstance(
                    text=x,
                    spec=y,
                    recon_spec=y_recon,
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
            y = self.spec_extractor(wav_tensor).squeeze(0)

            if self.feature_type == "mel":
                y_recon = y
            else:
                y_recon = self.mel_extractor(wav_tensor).squeeze(0)

        if y.size(1) != y_recon.size(1):
            raise RuntimeError(
                "spec and recon_spec must have the same number of frames. "
                + f"got spec T={y.size(1)}, recon_spec T={y_recon.size(1)}"
            )

        # 3. Load Speaker Embedding
        if item.spk_path and os.path.exists(item.spk_path):
            try:
                cond = torch.load(item.spk_path, weights_only=True).float()
            except Exception as e:
                print(f"Error loading speaker embedding {item.spk_path}: {e}. Using zero vector.")
                cond = torch.zeros(
                    self.spk_embedding_dim,
                    dtype=torch.float32,
                )
        else:
            cond = torch.zeros(
                self.spk_embedding_dim,
                dtype=torch.float32,
            )

        data_to_cache = (x, y, y_recon, cond)

        if cache_path:
            torch.save(data_to_cache, cache_path)

        return TrainDatasetInstance(
            text=x,
            spec=y,
            recon_spec=y_recon,
            cond=cond,
            wav_path=item.audio_path,
            script=item.text,
            utt_id=item.utt_id,
            spk_id=item.spk_id,
            dataset=item.dataset,
        )

    @property
    def spec_pad_value(self) -> float:
        return self.spec_extractor.pad_value

    @property
    def recon_spec_pad_value(self) -> float:
        return self.mel_extractor.pad_value
