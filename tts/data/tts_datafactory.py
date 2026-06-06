import random
from typing import Any

import librosa
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from tts.config.stage1.data_config import DataConfig

from .data_parser import LibriTTSParser, LJSpeechParser, VCTKParser
from .data_types import TTSBatch, TTSDatasetInstance, TTSItem
from .tts_dataset import TTSDataset
from .quantile_bucket_sampler import QuantileDurationBatchSampler


class TTSCollate:
    def __init__(self, spec_pad_value: float):
        self.spec_pad_value = spec_pad_value

    def __call__(
        self,
        batch: list[TTSDatasetInstance],
    ) -> TTSBatch:
        xs = [item.text for item in batch]
        ys = [item.spec for item in batch]
        conds = [item.cond for item in batch]

        scripts = [item.script for item in batch]
        wav_paths = [item.wav_path for item in batch]
        utt_ids = [item.utt_id for item in batch]
        spk_ids = [item.spk_id for item in batch]
        datasets = [item.dataset for item in batch]

        x_lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
        y_lengths = torch.tensor(
            [y.size(1) for y in ys],
            dtype=torch.long,
        )  # y is (n_mels, T_mel)

        x_padded = pad_sequence(
            xs,
            batch_first=True,
            padding_value=0,
        )

        ys_transposed = [y.transpose(0, 1) for y in ys]  # (T_mel, n_mels)
        y_padded = pad_sequence(
            ys_transposed,
            batch_first=True,
            padding_value=self.spec_pad_value,
        )  # (B, T_mel_max, n_mels)
        y_padded = y_padded.transpose(1, 2).contiguous()  # (B, n_mels, T_mel_max)

        cond_batched = torch.stack(conds, dim=0)

        return TTSBatch(
            text=x_padded,
            text_lengths=x_lengths,
            spec=y_padded,
            spec_lengths=y_lengths,
            cond=cond_batched,
            scripts=scripts,
            wav_paths=wav_paths,
            utt_ids=utt_ids,
            spk_ids=spk_ids,
            datasets=datasets,
        )


class TTSDataFactory:
    def __init__(self, config: DataConfig):
        self.config = config
        self.dataset_cfg = config.dataset
        self.train_cfg = config.train

        # 1. Parse all selected datasets
        all_items: list[TTSItem] = []
        for ds_name in self.dataset_cfg.dataset_list:
            ds_name = ds_name.lower()
            if ds_name == "ljspeech":
                parser = LJSpeechParser(self.dataset_cfg.ljspeech_root)
            elif ds_name == "vctk":
                parser = VCTKParser(self.dataset_cfg.vctk_root)
            elif ds_name == "libritts":
                parser = LibriTTSParser(self.dataset_cfg.libritts_root)
            else:
                print(f"[Warning] Unknown dataset name: {ds_name}. Skipping.")
                continue

            items = parser.parse()
            print(f"[Info] Parsed {len(items)} items from {ds_name}")
            all_items.extend(items)

        if not all_items:
            raise ValueError("No data items found. Check your dataset_list and root paths.")

        # 1.5 Filter by duration
        print(
            f"[Info] Filtering items by duration "
            + f"({self.dataset_cfg.min_duration_sec}s ~ {self.dataset_cfg.max_duration_sec}s)..."
        )

        filtered_pairs: list[tuple[TTSItem, float]] = []

        for item in tqdm(all_items, desc="Filtering data"):
            try:
                duration = librosa.get_duration(path=item.audio_path)
                if (
                    self.dataset_cfg.min_duration_sec
                    <= duration
                    <= self.dataset_cfg.max_duration_sec
                ):
                    filtered_pairs.append((item, float(duration)))
            except Exception as e:
                print(f"[Warning] Error checking duration for {item.audio_path}: {e}. Skipping.")

        print(
            f"[Info] Filtered {len(all_items) - len(filtered_pairs)} items. "
            + f"Remaining: {len(filtered_pairs)}"
        )

        # 2. Shuffle and Split
        random.seed(self.dataset_cfg.seed)
        random.shuffle(filtered_pairs)

        num_total = len(filtered_pairs)
        num_valid = int(num_total * self.dataset_cfg.val_ratio)

        valid_pairs = filtered_pairs[:num_valid]
        train_pairs = filtered_pairs[num_valid:]

        self.valid_items = [item for item, _duration in valid_pairs]
        self.train_items = [item for item, _duration in train_pairs]

        self.valid_durations = [duration for _item, duration in valid_pairs]
        self.train_durations = [duration for _item, duration in train_pairs]

        print(
            f"[Info] Data split complete: "
            + f"{len(self.train_items)} train, {len(self.valid_items)} valid"
        )

        # 3. Create Datasets
        self.train_dataset = TTSDataset(self.train_items, config)
        self.valid_dataset = TTSDataset(self.valid_items, config)

        self.collate_fn = TTSCollate(self.train_dataset.spec_pad_value)

    @property
    def train_loader(self) -> DataLoader[Any]:
        batch_sampler = QuantileDurationBatchSampler(
            bucketing_keys=self.train_durations,
            batch_size=self.train_cfg.batch_size,
            num_buckets=self.dataset_cfg.num_buckets,
            seed=self.dataset_cfg.seed,
            shuffle=True,
            drop_last=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.train_cfg.num_workers,
            pin_memory=True,
        )

    @property
    def valid_loader(self) -> DataLoader[Any]:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.train_cfg.val_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.train_cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
