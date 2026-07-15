import argparse
import glob
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Any

import librosa
import torch
from tqdm import tqdm

from tts.config.utils.io import load_config
from tts.models.modules.spk_encoder import ECAPASpeakerEncoder
from tts.preprocess.audio_preprocessor import AudioPreprocessor

from ..config.preprocess.preprocess_config import PreprocessConfigs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess audio dataset and extract speaker embeddings."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to preprocess config JSON file",
    )
    parser.add_argument(
        "-j",
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel audio preprocessing workers. Default: 8.",
    )
    parser.add_argument(
        "--skip_audio",
        action="store_true",
        help="Skip audio preprocessing and only extract speaker embeddings.",
    )
    parser.add_argument(
        "--skip_spk",
        action="store_true",
        help="Skip speaker embedding extraction.",
    )
    parser.add_argument(
        "--skip_existing_spk",
        action="store_true",
        help="Skip speaker embedding files that already exist.",
    )
    return parser.parse_args()


def print_config_report(preprocess_config: PreprocessConfigs, total_files: int) -> None:
    report = [
        "\n" + "=" * 50,
        " PREPROCESSING CONFIGURATION REPORT ",
        "=" * 50,
        f" Total Files to Scan    : {total_files}",
        f" Data Root Directory    : {preprocess_config.data_root_dir}",
        f" Preprocessed Directory : {preprocess_config.preprocessed_dir}",
        f" Target Sample Rate     : {preprocess_config.resample_sr} Hz",
        "-" * 50,
        f" Silence Trimming       : {preprocess_config.silence_trim}",
        f"   - Denoise before VAD : {preprocess_config.denoise_before_vad}",
        f"   - LPF before VAD     : {preprocess_config.lpf_before_vad} "
        + f"(Cutoff: {preprocess_config.lpf_cutoff_freq}Hz)",
        f"   - Silence Margin     : {preprocess_config.silence_margin_sec} sec",
        "-" * 50,
        f" Peak Normalization     : {preprocess_config.peak_normalize}",
        f"   - Target Peak        : {preprocess_config.peak_target}",
        "=" * 50 + "\n",
    ]
    print("\n".join(report))


def _init_worker_tqdm(lock: Any) -> None:
    tqdm.set_lock(lock)


def _split_evenly(items: list[str], n: int) -> list[list[str]]:
    return [items[i::n] for i in range(n)]


def _collect_all_files(root_dir: str) -> list[str]:
    all_files: list[str] = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def _process_file_shard(
    worker_id: int,
    files: list[str],
    preprocess_config: PreprocessConfigs,
) -> dict[str, int]:
    """
    Run audio preprocessing on one file shard.

    AudioPreprocessor is created inside each process.
    This avoids pickling Silero VAD and gives each worker its own model.
    """
    preprocessor = AudioPreprocessor(preprocess_config)
    audio_extensions = {".wav", ".flac"}

    stats = {
        "scanned": 0,
        "audio_processed": 0,
        "audio_skipped": 0,
        "copied": 0,
        "errors": 0,
    }

    pbar = tqdm(
        files,
        desc=f"Audio Worker {worker_id:02d}",
        unit="file",
        position=worker_id,
        leave=True,
        dynamic_ncols=True,
    )

    for p in pbar:
        stats["scanned"] += 1

        rel_path = os.path.relpath(p, preprocess_config.data_root_dir)
        ext = os.path.splitext(p)[1].lower()

        if ext in audio_extensions:
            base_rel_path, _ = os.path.splitext(rel_path)
            out_path = os.path.join(
                preprocess_config.preprocessed_dir,
                base_rel_path + ".wav",
            )

            try:
                success = preprocessor.process_audio(p, out_path)
                if success:
                    stats["audio_processed"] += 1
                else:
                    stats["audio_skipped"] += 1
                    tqdm.write(
                        f"[Warning][W{worker_id:02d}] "
                        + f"{rel_path}: no valid speech/audio after preprocessing"
                    )
            except Exception as e:
                stats["errors"] += 1
                tqdm.write(f"[Error][W{worker_id:02d}] Failed to process {rel_path}: {e}")

        else:
            out_path = os.path.join(preprocess_config.preprocessed_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            try:
                shutil.copy2(p, out_path)
                stats["copied"] += 1
            except Exception as e:
                stats["errors"] += 1
                tqdm.write(f"[Error][W{worker_id:02d}] Failed to copy {rel_path}: {e}")

    pbar.close()
    return stats


def run_audio_preprocessing(
    preprocess_config: PreprocessConfigs,
    num_workers: int,
) -> dict[str, int]:
    all_files = _collect_all_files(preprocess_config.data_root_dir)

    if not all_files:
        print(f"[Info] No files found in directory: '{preprocess_config.data_root_dir}'")
        return {
            "scanned": 0,
            "audio_processed": 0,
            "audio_skipped": 0,
            "copied": 0,
            "errors": 0,
        }

    print_config_report(preprocess_config, len(all_files))

    os.makedirs(preprocess_config.preprocessed_dir, exist_ok=True)

    num_workers = max(1, int(num_workers))
    num_workers = min(num_workers, len(all_files))

    print(
        f">>> Starting audio preprocessing with {num_workers} workers: "
        + f"{preprocess_config.data_root_dir} -> {preprocess_config.preprocessed_dir}"
    )

    if preprocess_config.silence_trim:
        print(">>> Each worker loads its own VAD model.")
    else:
        print(">>> Silence trimming is off.")

    file_shards = _split_evenly(all_files, num_workers)

    ctx = get_context("spawn")
    tqdm_lock = ctx.RLock()

    total_stats = {
        "scanned": 0,
        "audio_processed": 0,
        "audio_skipped": 0,
        "copied": 0,
        "errors": 0,
    }

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_init_worker_tqdm,
        initargs=(tqdm_lock,),
    ) as executor:
        futures = [
            executor.submit(
                _process_file_shard,
                worker_id,
                shard,
                preprocess_config,
            )
            for worker_id, shard in enumerate(file_shards)
            if shard
        ]

        for future in as_completed(futures):
            stats = future.result()
            for key, value in stats.items():
                total_stats[key] += value

    print("\n" + "=" * 50)
    print(" AUDIO PREPROCESSING SUMMARY ")
    print("=" * 50)
    print(f" Scanned files       : {total_stats['scanned']}")
    print(f" Audio processed     : {total_stats['audio_processed']}")
    print(f" Audio skipped       : {total_stats['audio_skipped']}")
    print(f" Non-audio copied    : {total_stats['copied']}")
    print(f" Errors              : {total_stats['errors']}")
    print("=" * 50)

    return total_stats


def run_speaker_embedding_extraction(
    preprocess_config: PreprocessConfigs,
    *,
    skip_existing: bool = False,
) -> dict[str, int]:
    print("\n>>> Initializing SpeakerEncoder (ECAPA-TDNN)...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        speaker_encoder = ECAPASpeakerEncoder(device=device)
        speaker_encoder.eval()
    except Exception as e:
        raise RuntimeError(
            f"Error loading SpeakerEncoder: {e}. Cannot proceed without a valid model."
        ) from e

    print(f">>> Extracting speaker embeddings from: {preprocess_config.preprocessed_dir}")

    wav_paths = glob.glob(
        os.path.join(preprocess_config.preprocessed_dir, "**", "*.wav"),
        recursive=True,
    )
    wav_paths = sorted(wav_paths)

    if not wav_paths:
        print(f"[Info] No WAV files found in directory: '{preprocess_config.preprocessed_dir}'")
        return {
            "wav_scanned": 0,
            "spk_written": 0,
            "spk_skipped": 0,
            "errors": 0,
        }

    stats = {
        "wav_scanned": 0,
        "spk_written": 0,
        "spk_skipped": 0,
        "errors": 0,
    }

    for wav_path in tqdm(wav_paths, desc="Extracting Speaker Embeddings", unit="wav"):
        stats["wav_scanned"] += 1

        base_path, _ = os.path.splitext(wav_path)
        output_path = f"{base_path}_spk.pt"

        if skip_existing and os.path.exists(output_path):
            stats["spk_skipped"] += 1
            continue

        try:
            wav_np, _ = librosa.load(wav_path, sr=16000, mono=True)
            waveform = torch.from_numpy(wav_np).float().unsqueeze(0)

            with torch.no_grad():
                embedding = speaker_encoder(waveform)

            embedding = embedding.squeeze().detach().cpu().float()

            if embedding.dim() != 1:
                raise RuntimeError(
                    f"Expected 1D speaker embedding, got shape={tuple(embedding.shape)}"
                )

            torch.save(embedding, output_path)
            stats["spk_written"] += 1

        except Exception as e:
            stats["errors"] += 1
            tqdm.write(f"[Error] Failed to extract speaker embedding for {wav_path}: {e}")

    print("\n" + "=" * 50)
    print(" SPEAKER EMBEDDING SUMMARY ")
    print("=" * 50)
    print(f" WAV scanned         : {stats['wav_scanned']}")
    print(f" Embeddings written  : {stats['spk_written']}")
    print(f" Embeddings skipped  : {stats['spk_skipped']}")
    print(f" Errors              : {stats['errors']}")
    print("=" * 50)

    return stats


def main(args: Any):
    if args.config:
        print(f"📖 Loading preprocess config from: {args.config}")
        preprocess_config = load_config(args.config, PreprocessConfigs)
    else:
        preprocess_config = PreprocessConfigs()

    total_errors = 0

    if not args.skip_audio:
        audio_stats = run_audio_preprocessing(
            preprocess_config=preprocess_config,
            num_workers=args.num_workers,
        )
        total_errors += audio_stats["errors"]
    else:
        print(">>> Skipping audio preprocessing.")

    if not args.skip_spk:
        spk_stats = run_speaker_embedding_extraction(
            preprocess_config=preprocess_config,
            skip_existing=args.skip_existing_spk,
        )
        total_errors += spk_stats["errors"]
    else:
        print(">>> Skipping speaker embedding extraction.")

    print("\n" + "=" * 50)
    print(" FULL PREPROCESS PIPELINE SUMMARY ")
    print("=" * 50)
    print(f" Total errors        : {total_errors}")
    print("=" * 50)

    if total_errors == 0:
        print("\n>>> All preprocessing tasks completed successfully!")
    else:
        print("\n>>> Finished with errors. Check logs above.")


if __name__ == "__main__":
    main(parse_args())
