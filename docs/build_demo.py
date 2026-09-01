"""
Build the demo page samples from aligned utterances.

Takes a directory of paired `.wav` and `.TextGrid` files, and for each one
writes a figure, a full-utterance audio file, and one clip per word, into
`docs/samples/<id>/`. The manifest it prints goes into `docs/index.html`.

    python scripts/build_demo.py samples_in docs/samples --corpus TIMIT

Requires ffmpeg on PATH for the audio encoding.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import textgrid as tg_mod

# Words shorter than this are not worth a clip button; a listener cannot judge
# a boundary from 40 ms of audio.
MIN_CLIP_SECONDS = 0.06

# Padding around a word clip, so that the onset is audible rather than clipped
# at the very first sample.
CLIP_PAD_SECONDS = 0.02


@dataclass
class Word:
    label: str
    start: float
    end: float
    clip: str | None


@dataclass
class Sample:
    id: str
    corpus: str
    transcript: str
    duration: float
    audio: str
    figure: str
    words: list[Word]


def read_tiers(path: Path) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]]:
    """Return the word and phone intervals of a TextGrid, in that order."""
    grid = tg_mod.TextGrid.fromFile(str(path))

    def tier(name: str) -> list[tuple[float, float, str]]:
        for item in grid.tiers:
            if item.name.lower() == name:
                return [(iv.minTime, iv.maxTime, iv.mark) for iv in item]
        raise SystemExit(f"{path.name}: no tier named {name!r}")

    return tier("words"), tier("phones")


def draw_figure(
    audio: np.ndarray,
    sample_rate: int,
    words: list[tuple[float, float, str]],
    phones: list[tuple[float, float, str]],
    out_path: Path,
) -> None:
    """Draw the waveform with a word tier and a phone tier beneath it."""
    duration = len(audio) / sample_rate
    times = np.arange(len(audio)) / sample_rate

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(min(2.2 * duration, 24), 3.4),
        gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.05},
        sharex=True,
    )

    axes[0].plot(times, audio, linewidth=0.4)
    axes[0].set_yticks([])
    axes[0].set_xlim(0, duration)
    for spine in axes[0].spines.values():
        spine.set_visible(False)

    for ax, intervals in ((axes[1], words), (axes[2], phones)):
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlim(0, duration)

        for start, end, label in intervals:
            ax.axvline(start, color="0.3", linewidth=0.6)
            if label.strip():
                ax.text(
                    (start + end) / 2,
                    0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        ax.axvline(duration, color="0.3", linewidth=0.6)

    axes[2].set_xlabel("Time (s)")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def encode(
    source: Path, target: Path, start: float | None = None, end: float | None = None
) -> None:
    """Encode a whole file or a slice of it, via ffmpeg."""
    command = ["ffmpeg", "-y", "-loglevel", "error"]

    if start is not None:
        command += ["-ss", f"{start:.3f}"]
    if end is not None and start is not None:
        command += ["-t", f"{end - start:.3f}"]

    command += ["-i", str(source), "-ac", "1", "-b:a", "96k", str(target)]

    subprocess.run(command, check=True)


def build_sample(
    wav_path: Path,
    grid_path: Path,
    out_dir: Path,
    sample_id: str,
    corpus: str,
) -> Sample:
    """Write one sample directory and return its manifest entry."""
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, sample_rate = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = len(audio) / sample_rate
    words, phones = read_tiers(grid_path)

    draw_figure(audio, sample_rate, words, phones, out_dir / "grid.png")
    encode(wav_path, out_dir / "audio.mp3")

    entries: list[Word] = []
    index = 0

    for start, end, label in words:
        label = label.strip()

        if not label:
            continue

        clip = None

        if end - start >= MIN_CLIP_SECONDS:
            clip_name = f"w{index:02d}.mp3"
            encode(
                wav_path,
                out_dir / clip_name,
                start=max(0.0, start - CLIP_PAD_SECONDS),
                end=min(duration, end + CLIP_PAD_SECONDS),
            )
            clip = clip_name
            index += 1

        entries.append(Word(label=label, start=start, end=end, clip=clip))

    return Sample(
        id=sample_id,
        corpus=corpus,
        transcript=" ".join(word.label for word in entries),
        duration=duration,
        audio="audio.mp3",
        figure="grid.png",
        words=entries,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])  # pyright: ignore
    parser.add_argument("source", type=Path, help="directory of .wav and .TextGrid pairs")
    parser.add_argument("destination", type=Path, help="docs/samples")
    parser.add_argument("--corpus", default="", help="label shown on the card, e.g. TIMIT")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="where to write samples.json. Default: <destination>/../samples.json",
    )
    args = parser.parse_args()

    pairs = []
    for wav_path in sorted(args.source.glob("*.wav")):
        grid_path = wav_path.with_suffix(".TextGrid")
        if grid_path.exists():
            pairs.append((wav_path, grid_path))
        else:
            print(f"skipping {wav_path.name}: no TextGrid")

    if not pairs:
        raise SystemExit(f"No .wav/.TextGrid pairs under {args.source}")

    manifest_path = args.manifest or args.destination.parent / "samples.json"

    existing: list[dict] = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())

    by_id = {entry["id"]: entry for entry in existing}

    for wav_path, grid_path in pairs:
        sample_id = wav_path.stem
        print(f"building {sample_id}")

        sample = build_sample(
            wav_path=wav_path,
            grid_path=grid_path,
            out_dir=args.destination / sample_id,
            sample_id=sample_id,
            corpus=args.corpus,
        )

        by_id[sample_id] = asdict(sample)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(list(by_id.values()), indent=2, ensure_ascii=False))

    print(f"\nWrote {len(pairs)} samples; manifest now holds {len(by_id)} ({manifest_path})")


if __name__ == "__main__":
    main()
