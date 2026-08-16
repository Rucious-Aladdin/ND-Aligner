from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

# Charsiu demo uses the local repository source.
sys.path.append("src/")
from Charsiu import charsiu_forced_aligner  # type: ignore

# ============================================================
# Config
# ============================================================

TIMIT_ROOT = Path("/shared/data_zfs/blue2959/TIMIT/TEST")
OUTPUT_PATH = Path("./charsiu_timit_efficiency.json")

CHARSIU_MODEL = "charsiu/en_w2v2_fc_10ms"

WARMUP_SAMPLES = 200
NUM_RUNS = 5


# ============================================================
# Utilities
# ============================================================


def read_timit_text(txt_path: Path) -> str:
    """
    TIMIT TXT:
        <start_sample> <end_sample> <text>
    """
    with txt_path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()

    parts = line.split(maxsplit=2)

    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
        return parts[2]

    return line


def collect_timit_samples(
    root_dir: Path,
) -> list[tuple[Path, str]]:
    """
    Returns:
        [(wav_path, text), ...]
    """
    wav_paths = sorted(list(root_dir.rglob("*.WAV")) + list(root_dir.rglob("*.wav")))

    samples: list[tuple[Path, str]] = []

    for wav_path in wav_paths:
        txt_path = wav_path.with_suffix(".TXT")

        if not txt_path.exists():
            txt_path = wav_path.with_suffix(".txt")

        if not txt_path.exists():
            continue

        text = read_timit_text(txt_path)
        samples.append((wav_path, text))

    return samples


def get_audio_duration_sec(
    wav_path: Path,
) -> float:
    info = sf.info(str(wav_path))
    return float(info.frames) / float(info.samplerate)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_unique_parameters(
    *modules: torch.nn.Module,
) -> int:
    """
    Count unique Parameters across one or more torch modules.
    """
    seen: set[int] = set()
    total = 0

    for module in modules:
        for param in module.parameters():
            param_id = id(param)

            if param_id in seen:
                continue

            seen.add(param_id)
            total += param.numel()

    return total


# ============================================================
# Load Charsiu
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

charsiu = charsiu_forced_aligner(
    aligner=CHARSIU_MODEL,
)

# The demo/API exposes the actual neural alignment model as .aligner.
if not hasattr(charsiu, "aligner"):
    raise RuntimeError(
        "Charsiu object has no '.aligner' attribute. "
        + "Inspect vars(charsiu) and update parameter counting."
    )

if not isinstance(charsiu.aligner, torch.nn.Module):
    raise TypeError(f"charsiu.aligner is not torch.nn.Module: {type(charsiu.aligner)}")

charsiu.aligner.eval()

print(f"[Efficiency] Device: {device}")
print(f"[Efficiency] Charsiu model: {CHARSIU_MODEL}")


# ============================================================
# Dataset
# ============================================================

samples = collect_timit_samples(TIMIT_ROOT)

if not samples:
    raise RuntimeError(f"No TIMIT samples found under: {TIMIT_ROOT}")

print(f"[Efficiency] Found {len(samples)} samples.")


# ------------------------------------------------------------
# Audio duration
#
# Calculated outside the timed region.
# ------------------------------------------------------------

audio_durations: list[float] = []

for wav_path, _ in tqdm(
    samples,
    desc="Reading audio durations",
):
    audio_durations.append(get_audio_duration_sec(wav_path))

total_audio_sec = sum(audio_durations)

print(
    f"[Efficiency] Total audio duration: "
    + f"{total_audio_sec:.2f} sec "
    + f"({total_audio_sec / 3600.0:.3f} h)"
)


# ============================================================
# Parameters
# ============================================================

# Charsiu's Wav2Vec2 forced-alignment network, including its
# frame-classification head.
num_params = count_unique_parameters(
    charsiu.aligner,
)

num_params_m = num_params / 1e6

print(f"[Efficiency] Parameters: " + f"{num_params:,} " + f"({num_params_m:.3f} M)")


# ============================================================
# Warm-up
# ============================================================

warmup_samples = samples[
    : min(
        WARMUP_SAMPLES,
        len(samples),
    )
]

print(f"[Efficiency] Warm-up: " + f"{len(warmup_samples)} samples")

with torch.inference_mode():
    for wav_path, text in tqdm(
        warmup_samples,
        desc="Warm-up",
    ):
        _ = charsiu.align(
            audio=str(wav_path),
            text=text,
        )

cuda_sync()


# ============================================================
# Benchmark
# ============================================================

run_results: list[dict[str, float | int]] = []

for run_idx in range(NUM_RUNS):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        cuda_sync()
        torch.cuda.reset_peak_memory_stats()

    cuda_sync()

    start_time = time.perf_counter()

    with torch.inference_mode():
        for wav_path, text in tqdm(
            samples,
            desc=f"Benchmark {run_idx + 1}/{NUM_RUNS}",
        ):
            _ = charsiu.align(
                audio=str(wav_path),
                text=text,
            )

    cuda_sync()

    elapsed_sec = time.perf_counter() - start_time

    rtf = elapsed_sec / total_audio_sec

    throughput = len(samples) / elapsed_sec

    if torch.cuda.is_available():
        peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)

        peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
    else:
        peak_allocated_gb = float("nan")
        peak_reserved_gb = float("nan")

    run_result = {
        "run": run_idx + 1,
        "elapsed_sec": elapsed_sec,
        "rtf": rtf,
        "utterances_per_sec": throughput,
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
    }

    run_results.append(run_result)

    print()
    print(f"[Run {run_idx + 1}]")
    print(f"  elapsed          : " + f"{elapsed_sec:.3f} sec")
    print(f"  RTF              : " + f"{rtf:.6f}")
    print(f"  utterances/sec   : " + f"{throughput:.3f}")
    print(f"  peak allocated   : " + f"{peak_allocated_gb:.3f} GB")
    print(f"  peak reserved    : " + f"{peak_reserved_gb:.3f} GB")


# ============================================================
# Aggregate
# ============================================================


def compute_mean_std(
    key: str,
) -> tuple[float, float]:
    values = torch.tensor(
        [float(x[key]) for x in run_results],
        dtype=torch.float64,
    )

    mean = values.mean().item()

    std = values.std(unbiased=True).item() if len(values) > 1 else 0.0

    return mean, std


mean_elapsed_sec, std_elapsed_sec = compute_mean_std("elapsed_sec")

mean_rtf, std_rtf = compute_mean_std("rtf")

mean_throughput, std_throughput = compute_mean_std("utterances_per_sec")

mean_peak_allocated_gb, std_peak_allocated_gb = compute_mean_std("peak_allocated_gb")

mean_peak_reserved_gb, std_peak_reserved_gb = compute_mean_std("peak_reserved_gb")


summary = {
    "model": CHARSIU_MODEL,
    "device": str(device),
    "num_samples": len(samples),
    "num_runs": NUM_RUNS,
    "warmup_samples": len(warmup_samples),
    "total_audio_sec": total_audio_sec,
    "total_audio_hours": (total_audio_sec / 3600.0),
    "num_params": num_params,
    "params_m": num_params_m,
    "mean_elapsed_sec": mean_elapsed_sec,
    "std_elapsed_sec": std_elapsed_sec,
    "mean_rtf": mean_rtf,
    "std_rtf": std_rtf,
    "mean_utterances_per_sec": mean_throughput,
    "std_utterances_per_sec": std_throughput,
    "mean_peak_allocated_gb": mean_peak_allocated_gb,
    "std_peak_allocated_gb": std_peak_allocated_gb,
    "mean_peak_reserved_gb": mean_peak_reserved_gb,
    "std_peak_reserved_gb": std_peak_reserved_gb,
    "runs": run_results,
}


# ============================================================
# Print final result
# ============================================================

print()
print("=" * 70)
print("Charsiu TIMIT Efficiency")
print("=" * 70)

print(f"Samples           : " + f"{len(samples)}")

print(
    f"Audio duration    : " + f"{total_audio_sec:.2f} sec " + f"({total_audio_sec / 3600.0:.3f} h)"
)

print(f"Params            : " + f"{num_params_m:.3f} M")

print(f"Elapsed           : " + f"{mean_elapsed_sec:.3f} ± " + f"{std_elapsed_sec:.3f} sec")

print(f"RTF               : " + f"{mean_rtf:.6f} ± " + f"{std_rtf:.6f}")

print(f"Throughput        : " + f"{mean_throughput:.3f} ± " + f"{std_throughput:.3f} utt/s")

print(
    f"Peak GPU memory   : " + f"{mean_peak_allocated_gb:.3f} ± " + f"{std_peak_allocated_gb:.3f} GB"
)

print(
    f"Peak reserved     : " + f"{mean_peak_reserved_gb:.3f} ± " + f"{std_peak_reserved_gb:.3f} GB"
)

print("=" * 70)


# ============================================================
# Save
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"[Efficiency] Saved to: " + f"{OUTPUT_PATH}")
