import csv
import glob
import os
import random
from abc import ABC, abstractmethod
from typing import NamedTuple, override

from .data_types import TrainItem


class ParsedItems(NamedTuple):
    train_items: list[TrainItem]
    test_items: list[TrainItem] | None = None


class BaseDatasetParser(ABC):
    @abstractmethod
    def parse(self) -> ParsedItems:
        raise NotImplementedError()


class LJSpeechParser(BaseDatasetParser):
    def __init__(
        self,
        root_dir: str,
        spk_encoder_tag: str = "",
        num_test_samples: int = 0,
        test_split_seed: int | None = None,
    ):
        if num_test_samples < 0:
            raise ValueError(f"num_test_samples must be non-negative, got {num_test_samples}.")

        if num_test_samples != 0 and test_split_seed is None:
            raise ValueError("There exist test samples but split seed is not set.")

        self.root_dir = root_dir
        self.spk_encoder_tag = spk_encoder_tag
        self.num_test_samples = num_test_samples
        self.test_split_seed = test_split_seed

    @override
    def parse(self) -> ParsedItems:
        items: list[TrainItem] = []

        metadata_path = os.path.join(self.root_dir, "metadata.csv")
        wav_dir = os.path.join(self.root_dir, "wavs")

        if not os.path.exists(metadata_path):
            return ParsedItems(train_items=[])

        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")

            for row in reader:
                if len(row) < 3:
                    continue

                wav_id = row[0]
                text = row[2]  # normalized text

                audio_path = os.path.join(wav_dir, f"{wav_id}.wav")
                base_path = os.path.join(wav_dir, wav_id)
                spk_path = f"{base_path}_{self.spk_encoder_tag}_spk.pt"

                if not os.path.exists(spk_path):
                    spk_path = ""

                items.append(
                    TrainItem(
                        audio_path=audio_path,
                        spk_path=spk_path,
                        text=text,
                        utt_id=wav_id,
                        spk_id="LJ",
                        dataset="ljspeech",
                    )
                )

        if self.num_test_samples == 0:
            return ParsedItems(train_items=items)

        if self.num_test_samples > len(items):
            raise ValueError(
                f"num_test_samples={self.num_test_samples} is larger than "
                + f"the number of parsed LJSpeech items={len(items)}."
            )

        assert self.test_split_seed is not None

        rng = random.Random(self.test_split_seed)
        test_indices = set(rng.sample(range(len(items)), self.num_test_samples))

        train_items = [item for idx, item in enumerate(items) if idx not in test_indices]
        test_items = [item for idx, item in enumerate(items) if idx in test_indices]

        return ParsedItems(
            train_items=train_items,
            test_items=test_items,
        )


class VCTKParser(BaseDatasetParser):
    def __init__(
        self,
        root_dir: str,
        spk_encoder_tag: str = "",
        test_speaker_ids: list[str] | None = None,
    ):
        self.root_dir = root_dir
        self.spk_encoder_tag = spk_encoder_tag
        self.test_speaker_ids = set(test_speaker_ids or [])

    @override
    def parse(self) -> ParsedItems:
        train_items: list[TrainItem] = []
        test_items: list[TrainItem] = []

        wav_root = os.path.join(self.root_dir, "wav48_silence_trimmed")
        txt_root = os.path.join(self.root_dir, "txt")

        if not os.path.exists(wav_root):
            return ParsedItems(train_items=[])

        wav_paths = sorted(
            glob.glob(
                os.path.join(wav_root, "**/*.wav"),
                recursive=True,
            )
        )

        for audio_path in wav_paths:
            file_name = os.path.basename(audio_path)

            if "_mic2" in file_name:
                continue

            rel_path = os.path.relpath(audio_path, wav_root)
            base_rel_path = os.path.splitext(rel_path)[0]
            text_path = os.path.join(txt_root, base_rel_path + ".txt")

            spk_id = rel_path.split(os.sep)[0]
            utt_id = os.path.splitext(os.path.basename(audio_path))[0].replace("_mic1", "")

            # VCTK common case: audio has _mic1, but text does not.
            if not os.path.exists(text_path):
                alt_base_rel_path = base_rel_path.replace("_mic1", "")
                alt_text_path = os.path.join(txt_root, alt_base_rel_path + ".txt")

                if os.path.exists(alt_text_path):
                    text_path = alt_text_path

            if not os.path.exists(text_path):
                continue

            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            spk_path = os.path.splitext(audio_path)[0] + f"_{self.spk_encoder_tag}_spk.pt"
            if not os.path.exists(spk_path):
                spk_path = ""

            item = TrainItem(
                audio_path=audio_path,
                spk_path=spk_path,
                text=text,
                utt_id=utt_id,
                spk_id=spk_id,
                dataset="vctk",
            )

            if spk_id in self.test_speaker_ids:
                test_items.append(item)
            else:
                train_items.append(item)

        if not self.test_speaker_ids:
            return ParsedItems(train_items=train_items)

        found_test_speakers = {item.spk_id for item in test_items}
        missing_test_speakers = self.test_speaker_ids - found_test_speakers

        if missing_test_speakers:
            print(
                "[Warning] Some configured VCTK test speakers were not found: "
                + f"{sorted(missing_test_speakers)}"
            )

        leaked_test_speakers = {item.spk_id for item in train_items} & self.test_speaker_ids
        if leaked_test_speakers:
            raise RuntimeError(
                "[Error] VCTK test speaker leakage detected: " + f"{sorted(leaked_test_speakers)}"
            )

        return ParsedItems(
            train_items=train_items,
            test_items=test_items,
        )


class LibriTTSParser(BaseDatasetParser):
    def __init__(
        self,
        root_dir: str,
        spk_encoder_tag: str = "",
        subsets: list[str] | None = None,
    ):
        self.root_dir = root_dir
        self.spk_encoder_tag = spk_encoder_tag
        self.subsets = subsets

    @override
    def parse(self) -> ParsedItems:
        items = []

        if self.subsets is None:
            subsets = ["train-clean-100", "train-clean-360"]
        else:
            subsets = self.subsets

        for subset in subsets:
            subset_dir = os.path.join(self.root_dir, subset)
            if not os.path.exists(subset_dir):
                continue

            dataset_name = {
                "train-clean-100": "libri-100",
                "train-clean-360": "libri-360",
            }[subset]

            wav_paths = sorted(glob.glob(os.path.join(subset_dir, "**/*.wav"), recursive=True))
            for audio_path in wav_paths:
                base_path = os.path.splitext(audio_path)[0]
                text_path = base_path + ".normalized.txt"

                if not os.path.exists(text_path):
                    continue

                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                spk_path = base_path + f"_{self.spk_encoder_tag}_spk.pt"
                if not os.path.exists(spk_path):
                    spk_path = ""

                rel_path = os.path.relpath(audio_path, subset_dir)
                parts = rel_path.split(os.sep)

                spk_id = parts[0] if len(parts) > 0 else ""
                utt_id = os.path.splitext(os.path.basename(audio_path))[0]

                items.append(
                    TrainItem(
                        audio_path=audio_path,
                        spk_path=spk_path,
                        text=text,
                        utt_id=utt_id,
                        spk_id=spk_id,
                        dataset=dataset_name,
                    )
                )
        return ParsedItems(
            train_items=items,
            test_items=None,
        )


if __name__ == "__main__":
    vctk_parser = VCTKParser(
        root_dir="/shared/data_zfs/blue2959/VCTK-preprocessed",
        test_speaker_ids=["p229"],
    )
    items = vctk_parser.parse()
    print(items.train_items[:3])
    print(items.test_items[:3])  # type: ignore

    # libritts_parser = LibriTTSParser(root_dir="/shared/data_zfs/blue2959/LibriTTS-preprocessed")
    # items = libritts_parser.parse()
    # print(items[:3])

    ljspeech_parser = LJSpeechParser(
        root_dir="/shared/data_zfs/blue2959/LJSpeech-1.1-preprocessed",
        num_test_samples=30,
        test_split_seed=2026,
    )
    items = ljspeech_parser.parse()
    print(items.train_items[:3])
    print(items.test_items[:3])  # type: ignore
