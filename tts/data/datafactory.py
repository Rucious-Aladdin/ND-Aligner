import random
from typing import Any

import librosa
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from tts.config.ndaligner.data_config import DataConfig

from .data_parser import LibriTTSParser, LJSpeechParser, VCTKParser
from .data_types import TrainBatch, TrainDatasetInstance, TrainItem
from .dataset import TTSDataset
from .quantile_bucket_sampler import QuantileDurationBatchSampler


class Collate:
    def __init__(
        self,
        spec_pad_value: float,
        recon_spec_pad_value: float,
    ):
        self.spec_pad_value = spec_pad_value
        self.recon_spec_pad_value = recon_spec_pad_value

    def __call__(
        self,
        batch: list[TrainDatasetInstance],
    ) -> TrainBatch:
        xs = [item.text for item in batch]
        ys = [item.spec for item in batch]
        y_recons = [item.recon_spec for item in batch]
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
        )

        y_recon_lengths = torch.tensor(
            [y.size(1) for y in y_recons],
            dtype=torch.long,
        )

        if not torch.equal(y_lengths, y_recon_lengths):
            raise RuntimeError(
                "spec_lengths and recon_spec_lengths must match. "
                + f"spec_lengths={y_lengths.tolist()}, "
                + f"recon_spec_lengths={y_recon_lengths.tolist()}"
            )

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

        y_recons_transposed = [y.transpose(0, 1) for y in y_recons]
        y_recon_padded = pad_sequence(
            y_recons_transposed,
            batch_first=True,
            padding_value=self.recon_spec_pad_value,
        )
        y_recon_padded = y_recon_padded.transpose(1, 2).contiguous()

        cond_batched = torch.stack(conds, dim=0)

        return TrainBatch(
            text=x_padded,
            text_lengths=x_lengths,
            spec=y_padded,
            spec_lengths=y_lengths,
            recon_spec=y_recon_padded,
            recon_spec_lengths=y_recon_lengths,
            cond=cond_batched,
            scripts=scripts,
            wav_paths=wav_paths,
            utt_ids=utt_ids,
            spk_ids=spk_ids,
            datasets=datasets,
        )


class DataFactory:
    def __init__(self, config: DataConfig):
        self.config = config
        self.dataset_cfg = config.dataset
        self.train_cfg = config.train

        # 1. Parse all selected datasets.
        all_train_items: list[TrainItem] = []
        all_test_items: list[TrainItem] = []

        for ds_name in self.dataset_cfg.dataset_list:
            ds_name = ds_name.lower()

            if ds_name == "ljspeech":
                parser = LJSpeechParser(
                    root_dir=self.dataset_cfg.ljspeech_root,
                    spk_encoder_tag=self.config.preprocess.spk_encoder_type,
                    num_test_samples=self.dataset_cfg.ljspeech_num_test_samples,
                    test_split_seed=self.dataset_cfg.seed,
                )
            elif ds_name == "vctk":
                parser = VCTKParser(
                    root_dir=self.dataset_cfg.vctk_root,
                    spk_encoder_tag=self.config.preprocess.spk_encoder_type,
                    test_speaker_ids=self.dataset_cfg.vctk_test_speakers,
                )
            elif ds_name == "libritts":
                parser = LibriTTSParser(
                    root_dir=self.dataset_cfg.libritts_root,
                    spk_encoder_tag=self.config.preprocess.spk_encoder_type,
                    subsets=self.dataset_cfg.libritts_subsets,
                )
            else:
                print(f"[Warning] Unknown dataset name: {ds_name}. Skipping.")
                continue

            parsed = parser.parse()

            train_items = parsed.train_items
            test_items = parsed.test_items or []

            print(
                f"[Info] Parsed {len(train_items)} train items "
                + f"and {len(test_items)} test items from {ds_name}"
            )

            all_train_items.extend(train_items)
            all_test_items.extend(test_items)

        if not all_train_items:
            raise ValueError("No train data items found. Check your dataset_list and root paths.")

        # 1.5 Filter train/test items by duration.
        print(
            f"[Info] Filtering items by duration "
            + f"({self.dataset_cfg.min_duration_sec}s ~ {self.dataset_cfg.max_duration_sec}s)..."
        )

        train_pairs = self._filter_items_by_duration(
            all_train_items,
            desc="Filtering train data",
        )

        test_pairs = self._filter_items_by_duration(
            all_test_items,
            desc="Filtering test data",
        )

        if not train_pairs:
            raise ValueError("No train items left after duration filtering.")

        # 2. Shuffle train pool and split train/valid.
        rng = random.Random(self.dataset_cfg.seed)
        rng.shuffle(train_pairs)

        num_total = len(train_pairs)
        num_valid = int(num_total * self.dataset_cfg.val_ratio)

        valid_pairs = train_pairs[:num_valid]
        train_pairs = train_pairs[num_valid:]

        self.train_items = [item for item, _duration in train_pairs]
        self.valid_items = [item for item, _duration in valid_pairs]
        self.test_items = [item for item, _duration in test_pairs]

        self.train_durations = [duration for _item, duration in train_pairs]
        self.valid_durations = [duration for _item, duration in valid_pairs]
        self.test_durations = [duration for _item, duration in test_pairs]

        print(
            f"[Info] Data split complete: "
            + f"{len(self.train_items)} train, "
            + f"{len(self.valid_items)} valid, "
            + f"{len(self.test_items)} test"
        )

        if not self.valid_items:
            print("[Warning] No validation items found. Check val_ratio.")

        if not self.test_items:
            print("[Warning] No test items found.")

        # 3. Create datasets.
        self.train_dataset = TTSDataset(self.train_items, config)
        self.valid_dataset = TTSDataset(self.valid_items, config)
        self.test_dataset = TTSDataset(self.test_items, config)

        self.collate_fn = Collate(
            spec_pad_value=self.train_dataset.spec_pad_value,
            recon_spec_pad_value=self.train_dataset.recon_spec_pad_value,
        )

    def _filter_items_by_duration(
        self,
        items: list[TrainItem],
        desc: str,
    ) -> list[tuple[TrainItem, float]]:
        filtered_pairs: list[tuple[TrainItem, float]] = []

        if not items:
            return filtered_pairs

        for item in tqdm(items, desc=desc):
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
            f"[Info] {desc}: filtered {len(items) - len(filtered_pairs)} items. "
            + f"Remaining: {len(filtered_pairs)}"
        )

        return filtered_pairs

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

    @property
    def test_loader(self) -> DataLoader[Any]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.train_cfg.val_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.train_cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
