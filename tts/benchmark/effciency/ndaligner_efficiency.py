from __future__ import annotations

import json
import os
import time
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from tts.config.ndaligner.training_module_config import NDAlignerTrainingModuleConfigs
from tts.config.utils.io import load_config
from tts.models.ndaligner import init_nd_aligner_training_module

# ============================================================
# Config
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

aligner_training_module_cfg_path = "/home/blue2959/monotonic_tts/runs/nd_aligner_main_vctk+full+sr16k+hop10ms+win25ms_20260807-084839/model_config.json"
aligner_training_module_ckpt_path = "/home/blue2959/monotonic_tts/runs/nd_aligner_main_vctk+full+sr16k+hop10ms+win25ms_20260807-084839/checkpoints_timit_bae/best_step_timit_bae_0.019883_step_1201000_epoch_329.pth"

TIMIT_ROOT = Path("/shared/data_zfs/blue2959/TIMIT/TEST")

OUTPUT_PATH = Path("./nd_aligner_timit_efficiency.json")

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

        samples.append(
            (
                wav_path,
                text,
            )
        )

    return samples


def get_audio_duration_sec(
    wav_path: Path,
) -> float:
    info = sf.info(str(wav_path))

    return float(info.frames) / float(info.samplerate)


def count_unique_parameters(
    *modules,  # type: ignore
) -> int:
    """
    여러 module에 같은 Parameter가 중복 등록되어 있어도
    한 번만 계산한다.

    aligner + input_maker를 함께 전달하면
    speaker encoder 등을 포함한 실제 inference neural
    components의 parameter 수를 셀 수 있다.
    """
    seen: set[int] = set()
    total = 0

    for module in modules:
        if module is None:
            continue

        parameters_fn = getattr(
            module,
            "parameters",
            None,
        )

        if parameters_fn is None:
            continue

        for param in parameters_fn():
            param_id = id(param)

            if param_id in seen:
                continue

            seen.add(param_id)
            total += param.numel()

    return total


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# Load model
# ============================================================

model_config = load_config(
    aligner_training_module_cfg_path,
    NDAlignerTrainingModuleConfigs,
)

aligner_training_module = init_nd_aligner_training_module(
    config=model_config,
    device=device,
)

aligner_training_module.load_checkpoint(
    ckpt_path=aligner_training_module_ckpt_path,
    device=device,
)

aligner = aligner_training_module.nd_aligner.eval()

assert aligner.input_maker is not None

aligner.input_maker.to(device=device)
aligner.input_maker.eval()


# ============================================================
# Dataset
# ============================================================

samples = collect_timit_samples(TIMIT_ROOT)

if not samples:
    raise RuntimeError(f"No TIMIT samples found under: {TIMIT_ROOT}")

print(f"[Efficiency] Found {len(samples)} samples.")


from silero_vad import load_silero_vad

assert aligner.input_maker is not None

aligner.input_maker.silero_model = load_silero_vad(onnx=True)
aligner.input_maker.trim_nonspeech_region = aligner.input_maker.zero_nonspeech_region = True

# ------------------------------------------------------------
# Audio duration
#
# IMPORTANT:
# 이것은 timed region 밖에서 계산한다.
# ------------------------------------------------------------

audio_durations = []

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

num_params = count_unique_parameters(
    aligner,
    aligner.input_maker,
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
        _ = aligner.inference_from_wavs(
            wav_paths=[str(wav_path)],
            texts=[text],
        )

cuda_sync()


# ============================================================
# Benchmark
# ============================================================

run_results = []

for run_idx in range(NUM_RUNS):

    # --------------------------------------------------------
    # Reset peak stats AFTER warm-up.
    #
    # 현재 GPU에 올라가 있는 model weights 등의 allocation은
    # baseline으로 유지되고, 이후 inference peak가 기록된다.
    # --------------------------------------------------------

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
            _ = aligner.inference_from_wavs(
                wav_paths=[str(wav_path)],
                texts=[text],
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
        [x[key] for x in run_results],
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
    "device": str(device),
    "num_samples": len(samples),
    "num_runs": NUM_RUNS,
    "warmup_samples": len(warmup_samples),
    "total_audio_sec": total_audio_sec,
    "total_audio_hours": total_audio_sec / 3600.0,
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
print("ND-Aligner TIMIT Efficiency")
print("=" * 70)

print(f"Samples           : " + f"{len(samples)}")

print(
    f"Audio duration    : " + f"{total_audio_sec:.2f} sec " + f"({total_audio_sec / 3600.0:.3f} h)"
)

print(f"Params            : " + f"{num_params_m:.3f} M")

print(f"Elapsed           : " + f"{mean_elapsed_sec:.3f} ± {std_elapsed_sec:.3f} sec")

print(f"RTF               : " + f"{mean_rtf:.6f} ± {std_rtf:.6f}")

print(f"Throughput        : " + f"{mean_throughput:.3f} ± {std_throughput:.3f} utt/s")

print(
    f"Peak GPU memory   : " + f"{mean_peak_allocated_gb:.3f} ± " + f"{std_peak_allocated_gb:.3f} GB"
)

print(
    f"Peak reserved     : " + f"{mean_peak_reserved_gb:.3f} ± " + f"{std_peak_reserved_gb:.3f} GB"
)

print("=" * 70)
