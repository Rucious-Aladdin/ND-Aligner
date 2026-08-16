from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from omegaconf import OmegaConf  # type: ignore
from tqdm import tqdm

# ============================================================
# NeMo Forced Aligner import path
# ============================================================

_THIS_DIR = Path(__file__).resolve().parent

_NFA_DIR = _THIS_DIR / "tools" / "nemo_forced_aligner"

if not _NFA_DIR.exists():
    raise RuntimeError(f"NeMo Forced Aligner directory not found: {_NFA_DIR}")

sys.path.insert(
    0,
    str(_NFA_DIR),
)


from align import AlignmentConfig  # type: ignore
from nemo.collections.asr.models.ctc_models import EncDecCTCModel  # type: ignore
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import (  # type: ignore
    EncDecHybridRNNTCTCModel,
)
from nemo.collections.asr.parts.utils.aligner_utils import (  # type: ignore
    add_t_start_end_to_utt_obj,
    get_batch_variables,
    viterbi_decoding,
)
from nemo.collections.asr.parts.utils.transcribe_utils import setup_model  # type: ignore

TIMIT_ROOT = Path("/shared/data_zfs/blue2959/TIMIT/TEST")
OUTPUT_PATH = Path("./nemo_timit_efficiency.json")

PRETRAINED_NAME = "stt_en_conformer_ctc_medium"

WARMUP_SAMPLES = 200
NUM_RUNS = 5
BATCH_SIZE = 1
USE_LOCAL_ATTENTION = True
SEGMENT_SEPARATORS = [".", "?", "!", "..."]


def read_timit_text(txt_path: Path) -> str:
    with txt_path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()

    parts = line.split(maxsplit=2)
    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
        return parts[2]
    return line


def collect_timit_samples(root_dir: Path) -> list[tuple[Path, str]]:
    wav_paths = sorted(list(root_dir.rglob("*.WAV")) + list(root_dir.rglob("*.wav")))

    samples: list[tuple[Path, str]] = []

    for wav_path in wav_paths:
        txt_path = wav_path.with_suffix(".TXT")
        if not txt_path.exists():
            txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue

        samples.append(
            (
                wav_path,
                read_timit_text(txt_path),
            )
        )

    return samples


def get_audio_duration_sec(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return float(info.frames) / float(info.samplerate)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_unique_parameters(*modules: torch.nn.Module) -> int:
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


def compute_mean_std(
    run_results: list[dict[str, float | int]],
    key: str,
) -> tuple[float, float]:
    values = torch.tensor(
        [float(x[key]) for x in run_results],
        dtype=torch.float64,
    )

    mean = values.mean().item()
    std = values.std(unbiased=True).item() if len(values) > 1 else 0.0
    return mean, std


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this benchmark.")

transcribe_device = torch.device("cuda")
viterbi_device = torch.device("cuda")

cfg = OmegaConf.structured(
    AlignmentConfig(
        pretrained_name=PRETRAINED_NAME,
        model_path=None,
        manifest_filepath=None,
        output_dir=None,
        align_using_pred_text=False,
        transcribe_device="cuda",
        viterbi_device="cuda",
        batch_size=BATCH_SIZE,
        use_local_attention=USE_LOCAL_ATTENTION,
        additional_segment_grouping_separator=SEGMENT_SEPARATORS,
        audio_filepath_parts_in_utt_id=1,
        use_buffered_chunked_streaming=False,
        simulate_cache_aware_streaming=False,
    )
)

model, _ = setup_model(
    cfg,
    transcribe_device,
)

model.eval()

if isinstance(model, EncDecHybridRNNTCTCModel):
    model.change_decoding_strategy(decoder_type="ctc")

if USE_LOCAL_ATTENTION:
    model.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=[64, 64],
    )

if not isinstance(
    model,
    (EncDecCTCModel, EncDecHybridRNNTCTCModel),
):
    raise TypeError(f"Unsupported model type: {type(model)}")


samples = collect_timit_samples(TIMIT_ROOT)

if not samples:
    raise RuntimeError(f"No TIMIT samples found under: {TIMIT_ROOT}")

print(f"[Efficiency] Found {len(samples)} samples.")

audio_durations = [
    get_audio_duration_sec(wav_path)
    for wav_path, _ in tqdm(
        samples,
        desc="Reading audio durations",
    )
]

total_audio_sec = sum(audio_durations)

print(
    f"[Efficiency] Total audio duration: "
    + f"{total_audio_sec:.2f} sec "
    + f"({total_audio_sec / 3600.0:.3f} h)"
)


num_params = count_unique_parameters(model)
num_params_m = num_params / 1e6

print(f"[Efficiency] Parameters: " + f"{num_params:,} " + f"({num_params_m:.3f} M)")


def align_one(
    wav_path: Path,
    text: str,
    output_timestep_duration: float | None,
) -> tuple[object, float]:
    (
        log_probs_batch,
        y_batch,
        T_batch,
        U_batch,
        utt_obj_batch,
        output_timestep_duration,
    ) = get_batch_variables(
        audio=[str(wav_path)],
        model=model,
        segment_separators=SEGMENT_SEPARATORS,
        align_using_pred_text=False,
        audio_filepath_parts_in_utt_id=1,
        gt_text_batch=[text],
        output_timestep_duration=output_timestep_duration,
        simulate_cache_aware_streaming=False,
        use_buffered_chunked_streaming=False,
        buffered_chunk_params={},
    )

    alignments_batch = viterbi_decoding(
        log_probs_batch,
        y_batch,
        T_batch,
        U_batch,
        viterbi_device,
    )

    utt_obj = add_t_start_end_to_utt_obj(
        utt_obj_batch[0],
        alignments_batch[0],
        output_timestep_duration,
    )

    return utt_obj, output_timestep_duration  # type: ignore


warmup_samples = samples[
    : min(
        WARMUP_SAMPLES,
        len(samples),
    )
]

print(f"[Efficiency] Warm-up: " + f"{len(warmup_samples)} samples")

output_timestep_duration = None

with torch.inference_mode():
    for wav_path, text in tqdm(
        warmup_samples,
        desc="Warm-up",
    ):
        _, output_timestep_duration = align_one(
            wav_path,
            text,
            output_timestep_duration,
        )

cuda_sync()


run_results: list[dict[str, float | int]] = []

for run_idx in range(NUM_RUNS):
    torch.cuda.empty_cache()
    cuda_sync()
    torch.cuda.reset_peak_memory_stats()

    cuda_sync()

    run_output_timestep_duration = output_timestep_duration

    start_time = time.perf_counter()

    with torch.inference_mode():
        for wav_path, text in tqdm(
            samples,
            desc=f"Benchmark {run_idx + 1}/{NUM_RUNS}",
        ):
            _, run_output_timestep_duration = align_one(
                wav_path,
                text,
                run_output_timestep_duration,
            )

    cuda_sync()

    elapsed_sec = time.perf_counter() - start_time

    rtf = elapsed_sec / total_audio_sec
    throughput = len(samples) / elapsed_sec

    peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)

    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)

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
    print(f"  elapsed          : {elapsed_sec:.3f} sec")
    print(f"  RTF              : {rtf:.6f}")
    print(f"  utterances/sec   : {throughput:.3f}")
    print(f"  peak allocated   : {peak_allocated_gb:.3f} GB")
    print(f"  peak reserved    : {peak_reserved_gb:.3f} GB")


mean_elapsed_sec, std_elapsed_sec = compute_mean_std(
    run_results,
    "elapsed_sec",
)

mean_rtf, std_rtf = compute_mean_std(
    run_results,
    "rtf",
)

mean_throughput, std_throughput = compute_mean_std(
    run_results,
    "utterances_per_sec",
)

mean_peak_allocated_gb, std_peak_allocated_gb = compute_mean_std(
    run_results,
    "peak_allocated_gb",
)

mean_peak_reserved_gb, std_peak_reserved_gb = compute_mean_std(
    run_results,
    "peak_reserved_gb",
)


summary = {
    "model": PRETRAINED_NAME,
    "device": "cuda",
    "batch_size": BATCH_SIZE,
    "use_local_attention": USE_LOCAL_ATTENTION,
    "attention_context": [64, 64] if USE_LOCAL_ATTENTION else None,
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


print()
print("=" * 70)
print("NeMo Forced Aligner TIMIT Efficiency")
print("=" * 70)

print(f"Model             : {PRETRAINED_NAME}")
print(f"Samples           : {len(samples)}")
print(
    f"Audio duration    : " + f"{total_audio_sec:.2f} sec " + f"({total_audio_sec / 3600.0:.3f} h)"
)
print(f"Params            : {num_params_m:.3f} M")
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
