# make_libritts_long_json.py

import argparse
import json
import random
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def get_duration_sec(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return float(info.frames) / float(info.samplerate)


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/shared/data_zfs/blue2959/LibriTTS-preprocessed",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="libritts_long_25s_500.json",
    )
    parser.add_argument("--min_sec", type=float, default=25.0)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=["train-clean-100", "train-clean-360", "test-clean"],
    )
    args = parser.parse_args()

    root = Path(args.root)
    rng = random.Random(args.seed)

    candidates = []

    for subset in args.subsets:
        subset_dir = root / subset
        if not subset_dir.exists():
            print(f"[skip] subset not found: {subset_dir}")
            continue

        wav_paths = sorted(subset_dir.rglob("*.wav"))
        print(f"[scan] {subset}: {len(wav_paths)} wavs")

        for wav_path in tqdm(wav_paths, desc=f"Scanning {subset}", unit="wav"):
            normalized_path = wav_path.with_suffix(".normalized.txt")

            if not normalized_path.exists():
                continue

            try:
                duration = get_duration_sec(wav_path)
            except Exception as e:
                print(f"[warn] failed duration: {wav_path} | {e}")
                continue

            if duration < args.min_sec:
                continue

            try:
                text = read_text(normalized_path)
            except Exception as e:
                print(f"[warn] failed text: {normalized_path} | {e}")
                continue

            if not text:
                continue

            candidates.append(
                {
                    "wav_path": str(wav_path),
                    "text": text,
                    "duration": duration,
                }
            )

    print(f"[found] >= {args.min_sec:.1f}s: {len(candidates)}")

    rng.shuffle(candidates)
    selected = candidates[: args.num_samples]

    output = [[item["wav_path"], item["text"]] for item in selected]

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[saved] {out_path}")
    print(f"[selected] {len(output)}")

    if selected:
        durations = [x["duration"] for x in selected]
        print(
            f"[duration] "
            + f"min={min(durations):.2f}s "
            + f"max={max(durations):.2f}s "
            + f"avg={sum(durations) / len(durations):.2f}s"
        )


if __name__ == "__main__":
    main()
