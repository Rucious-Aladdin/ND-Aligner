from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import soundfile as sf
from praatio import textgrid as tgio
from tqdm.auto import tqdm

SPECIAL_LABELS = {
    "spn",
    "sil",
    "noise",
    "vocnoise",
    "iver",
    "laugh",
    "unknown",
    "exclude",
    "cutoff",
    "error",
    "ext",
}

ANGLE_TOKEN_PATTERN = re.compile(r"^<[^>]*>$")
BRACE_TOKEN_PATTERN = re.compile(r"^\{[^}]*\}$")


def entry_values(entry) -> tuple[float, float, str]:
    """Support both praatio Interval objects and tuple-like entries."""
    if hasattr(entry, "start"):
        return float(entry.start), float(entry.end), str(entry.label)

    return float(entry[0]), float(entry[1]), str(entry[2])


def is_special_token(label: str) -> bool:
    """
    Return True for non-lexical/special Buckeye labels.

    Examples:
        <EXCLUDE>
        <CUTOFF>
        <UNKNOWN>
        <LAUGH>
        {B_TRANS}
        spn
    """
    token = label.strip()
    lowered = token.lower()

    if not token:
        return True

    if ANGLE_TOKEN_PATTERN.fullmatch(token):
        return True

    if BRACE_TOKEN_PATTERN.fullmatch(token):
        return True

    return lowered in SPECIAL_LABELS


def find_tier_name(
    tier_names: tuple[str, ...] | list[str],
    speaker: str,
    tier_type: str,
) -> str:
    """
    Find the utterance or word tier.

    Expected Buckeye benchmark names:
        s01
        s01 - words
        s01 - phones
    """
    if tier_type == "utterance":
        expected = speaker
        if expected in tier_names:
            return expected

        candidates = [
            name
            for name in tier_names
            if "word" not in name.lower() and "phone" not in name.lower()
        ]

    elif tier_type == "words":
        expected = f"{speaker} - words"
        if expected in tier_names:
            return expected

        candidates = [name for name in tier_names if "word" in name.lower()]

    else:
        raise ValueError(f"Unsupported tier type: {tier_type}")

    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not uniquely determine {tier_type} tier for {speaker}. "
            f"Available tiers: {tier_names}"
        )

    return candidates[0]


def seconds_to_sample(time_sec: float, sample_rate: int) -> int:
    return int(round(time_sec * sample_rate))


def select_words_for_utterance(
    word_entries,
    utterance_start: float,
    utterance_end: float,
) -> list[tuple[float, float, str]]:
    """
    Select words whose midpoint lies inside the utterance interval.

    Padding silence around each utterance therefore does not introduce
    neighboring words.
    """
    selected = []

    for entry in word_entries:
        start, end, label = entry_values(entry)
        midpoint = start + (end - start) / 2.0

        if midpoint < utterance_start:
            continue

        if midpoint > utterance_end:
            break

        selected.append((start, end, label.strip()))

    return selected


def write_chunk(
    *,
    audio,
    sample_rate: int,
    subtype: str,
    crop_start: int,
    crop_end: int,
    words: list[tuple[float, float, str]],
    output_stem: Path,
) -> None:
    chunk_audio = audio[crop_start:crop_end]
    num_samples = len(chunk_audio)

    wav_path = output_stem.with_suffix(".wav")
    txt_path = output_stem.with_suffix(".TXT")
    wrd_path = output_stem.with_suffix(".WRD")

    sf.write(
        wav_path,
        chunk_audio,
        sample_rate,
        subtype=subtype,
    )

    transcript = " ".join(label for _, _, label in words)

    # TIMIT-style TXT:
    # <start sample> <end sample> <transcript>
    txt_path.write_text(
        f"0 {num_samples} {transcript}\n",
        encoding="utf-8",
    )

    wrd_lines = []

    for word_start_sec, word_end_sec, label in words:
        word_start = seconds_to_sample(word_start_sec, sample_rate) - crop_start
        word_end = seconds_to_sample(word_end_sec, sample_rate) - crop_start

        word_start = max(0, min(word_start, num_samples))
        word_end = max(0, min(word_end, num_samples))

        if word_end <= word_start:
            continue

        wrd_lines.append(f"{word_start} {word_end} {label}")

    wrd_path.write_text(
        "\n".join(wrd_lines) + "\n",
        encoding="utf-8",
    )


def process_recording(
    textgrid_path: Path,
    wav_path: Path,
    grid_out_dir: Path,
    min_words: int,
) -> Counter:
    speaker = textgrid_path.parent.name
    recording_id = textgrid_path.stem

    output_speaker_dir = grid_out_dir / speaker
    output_speaker_dir.mkdir(parents=True, exist_ok=True)

    tg = tgio.openTextgrid(
        textgrid_path,
        includeEmptyIntervals=False,
    )

    tier_names = list(tg.tierNames)

    utterance_tier_name = find_tier_name(
        tier_names,
        speaker,
        "utterance",
    )
    word_tier_name = find_tier_name(
        tier_names,
        speaker,
        "words",
    )

    utterance_entries = tg.getTier(utterance_tier_name).entries
    word_entries = tg.getTier(word_tier_name).entries

    info = sf.info(wav_path)
    audio, sample_rate = sf.read(
        wav_path,
        always_2d=True,
    )

    if sample_rate != info.samplerate:
        raise RuntimeError(f"Unexpected sample-rate mismatch: {wav_path}")

    total_samples = len(audio)
    stats = Counter()

    output_index = 0

    for source_index, utterance_entry in enumerate(utterance_entries):
        utterance_start, utterance_end, utterance_label = entry_values(utterance_entry)

        utterance_label = utterance_label.strip()

        if not utterance_label:
            stats["empty_utterance"] += 1
            continue

        words = select_words_for_utterance(
            word_entries,
            utterance_start,
            utterance_end,
        )

        if not words:
            stats["no_words"] += 1
            continue

        # 특수 토큰이 하나라도 있으면 utterance 전체 제거.
        special_words = [label for _, _, label in words if is_special_token(label)]

        # Utterance tier에도 특수 토큰이 남아 있는지 추가 확인.
        utterance_tokens = utterance_label.split()
        special_utterance_tokens = [token for token in utterance_tokens if is_special_token(token)]

        if special_words or special_utterance_tokens:
            stats["special_token"] += 1
            continue

        # MFA 3.0 benchmark protocol:
        # utterances with three words or fewer are removed.
        if len(words) < min_words:
            stats["too_few_words"] += 1
            continue

        crop_start = seconds_to_sample(utterance_start, sample_rate)
        crop_end = seconds_to_sample(utterance_end, sample_rate)

        crop_start = max(0, min(crop_start, total_samples))
        crop_end = max(0, min(crop_end, total_samples))

        if crop_end <= crop_start:
            stats["invalid_crop"] += 1
            continue

        output_stem = output_speaker_dir / (f"{recording_id}_chunk_{output_index:04d}")

        write_chunk(
            audio=audio,
            sample_rate=sample_rate,
            subtype=info.subtype,
            crop_start=crop_start,
            crop_end=crop_end,
            words=words,
            output_stem=output_stem,
        )

        output_index += 1
        stats["written"] += 1

    stats["source_utterances"] += len(utterance_entries)
    return stats


def convert_buckeye_benchmark(
    benchmark_dir: Path,
    grid_out_dir: Path,
    min_words: int,
) -> None:
    benchmark_dir = benchmark_dir.resolve()
    grid_out_dir = grid_out_dir.resolve()

    if not benchmark_dir.is_dir():
        raise NotADirectoryError(f"Benchmark directory does not exist: {benchmark_dir}")

    grid_out_dir.mkdir(parents=True, exist_ok=True)

    textgrid_paths = sorted(benchmark_dir.glob("s*/s*.TextGrid"))

    if not textgrid_paths:
        raise FileNotFoundError(f"No TextGrid files found under: {benchmark_dir}")

    total_stats = Counter()

    progress = tqdm(
        textgrid_paths,
        desc="Converting Buckeye",
        unit="recording",
        dynamic_ncols=True,
    )

    for textgrid_path in progress:
        relative_path = textgrid_path.relative_to(benchmark_dir)

        progress.set_postfix_str(str(relative_path))

        wav_path = textgrid_path.with_suffix(".wav")

        if not wav_path.exists():
            tqdm.write(f"[SKIP] Missing WAV: {wav_path}")
            total_stats["missing_wav"] += 1
            continue

        stats = process_recording(
            textgrid_path=textgrid_path,
            wav_path=wav_path,
            grid_out_dir=grid_out_dir,
            min_words=min_words,
        )
        total_stats.update(stats)

        progress.set_postfix(
            file=textgrid_path.stem,
            written=total_stats["written"],
            special=total_stats["special_token"],
            short=total_stats["too_few_words"],
        )

    print("\n" + "=" * 70)
    print("Buckeye benchmark conversion completed")
    print("=" * 70)
    print(f"Input directory : {benchmark_dir}")
    print(f"Output directory: {grid_out_dir}")
    print(f"Minimum words   : {min_words}")
    print("-" * 70)

    for key in [
        "source_utterances",
        "written",
        "special_token",
        "too_few_words",
        "empty_utterance",
        "no_words",
        "invalid_crop",
        "missing_wav",
    ]:
        print(f"{key:20s}: {total_stats[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the MFA Buckeye benchmark into TIMIT-style "
            "utterance-level WAV, TXT, and WRD files."
        )
    )

    parser.add_argument(
        "benchmark_dir",
        type=Path,
        help="Buckeye-benchmark directory containing s01/.../s0101a.TextGrid and WAV files.",
    )
    parser.add_argument(
        "grid_out_dir",
        type=Path,
        help="Output directory for TIMIT-style utterance chunks.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=4,
        help="Minimum number of words per utterance. Default: 4.",
    )

    args = parser.parse_args()

    convert_buckeye_benchmark(
        benchmark_dir=args.benchmark_dir,
        grid_out_dir=args.grid_out_dir,
        min_words=args.min_words,
    )


if __name__ == "__main__":
    main()
