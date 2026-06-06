import csv
import glob
import os
from abc import ABC, abstractmethod
from typing import override

from .data_types import TTSItem


class BaseDatasetParser(ABC):
    @abstractmethod
    def parse(self) -> list[TTSItem]:
        raise NotImplementedError()


class LJSpeechParser(BaseDatasetParser):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    @override
    def parse(self) -> list[TTSItem]:
        items = []
        metadata_path = os.path.join(self.root_dir, "metadata.csv")
        wav_dir = os.path.join(self.root_dir, "wavs")

        if not os.path.exists(metadata_path):
            return items

        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if len(row) < 3:
                    continue
                wav_id = row[0]
                text = row[2]  # normalized text

                audio_path = os.path.join(wav_dir, f"{wav_id}.wav")
                base_path = os.path.join(wav_dir, wav_id)
                spk_path = f"{base_path}_spk.pt"

                if not os.path.exists(spk_path):
                    spk_path = ""

                items.append(
                    TTSItem(
                        audio_path=audio_path,
                        spk_path=spk_path,
                        text=text,
                        utt_id=wav_id,
                        spk_id="LJ",
                        dataset="ljspeech",
                    )
                )
        return items


class VCTKParser(BaseDatasetParser):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    @override
    def parse(self) -> list[TTSItem]:
        items = []
        wav_root = os.path.join(self.root_dir, "wav48_silence_trimmed")
        txt_root = os.path.join(self.root_dir, "txt")

        if not os.path.exists(wav_root):
            return items

        # Recursively find all wav files
        wav_paths = sorted(glob.glob(os.path.join(wav_root, "**/*.wav"), recursive=True))

        for audio_path in wav_paths:
            file_name = os.path.basename(audio_path)
            if "_mic2" in file_name:
                continue

            # Construct text path by changing ancestor folder and extension
            rel_path = os.path.relpath(audio_path, wav_root)
            base_rel_path = os.path.splitext(rel_path)[0]
            text_path = os.path.join(txt_root, base_rel_path + ".txt")

            spk_id = rel_path.split(os.sep)[0]
            utt_id = os.path.splitext(os.path.basename(audio_path))[0].replace("_mic1", "")

            # VCTK common case: audio has _mic1, but text doesn't
            if not os.path.exists(text_path):
                alt_base_rel_path = base_rel_path.replace("_mic1", "")
                alt_text_path = os.path.join(txt_root, alt_base_rel_path + ".txt")
                if os.path.exists(alt_text_path):
                    text_path = alt_text_path

            if not os.path.exists(text_path):
                continue

            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            spk_path = os.path.splitext(audio_path)[0] + "_spk.pt"
            if not os.path.exists(spk_path):
                spk_path = ""

            items.append(
                TTSItem(
                    audio_path=audio_path,
                    spk_path=spk_path,
                    text=text,
                    utt_id=utt_id,
                    spk_id=spk_id,
                    dataset="vctk",
                )
            )
        return items


class LibriTTSParser(BaseDatasetParser):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    @override
    def parse(self) -> list[TTSItem]:
        items = []
        subsets = ["train-clean-100", "train-clean-360"]

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

                spk_path = base_path + "_spk.pt"
                if not os.path.exists(spk_path):
                    spk_path = ""

                rel_path = os.path.relpath(audio_path, subset_dir)
                parts = rel_path.split(os.sep)

                spk_id = parts[0] if len(parts) > 0 else ""
                utt_id = os.path.splitext(os.path.basename(audio_path))[0]

                items.append(
                    TTSItem(
                        audio_path=audio_path,
                        spk_path=spk_path,
                        text=text,
                        utt_id=utt_id,
                        spk_id=spk_id,
                        dataset=dataset_name,
                    )
                )
        return items


if __name__ == "__main__":
    vctk_parser = VCTKParser(root_dir="/shared/data_zfs/blue2959/VCTK-preprocessed")
    items = vctk_parser.parse()
    print(items[:3])

    libritts_parser = LibriTTSParser(root_dir="/shared/data_zfs/blue2959/LibriTTS-preprocessed")
    items = libritts_parser.parse()
    print(items[:3])

    ljspeech_parser = LJSpeechParser(root_dir="/shared/data_zfs/blue2959/LJSpeech-1.1-preprocessed")
    items = ljspeech_parser.parse()
    print(items[:3])
