import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Any

from tqdm import tqdm

from nd_aligner.config.utils.io import load_config
from nd_aligner.preprocess.audio_preprocessor import AudioPreprocessor

from ..config.preprocess.preprocess_config import PreprocessConfigs


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


def _process_file_shard(
    worker_id: int,
    files: list[str],
    preprocess_config: PreprocessConfigs,
) -> dict[str, int]:
    """
    Run preprocessing on one shard.

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
        desc=f"Worker {worker_id:02d}",
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


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio dataset")
    parser.add_argument("-c", "--config", type=str, help="Path to preprocess config JSON file")
    parser.add_argument(
        "-j",
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel preprocessing workers. Default: 16.",
    )
    args = parser.parse_args()

    if args.config:
        print(f"📖 Loading preprocess config from: {args.config}")
        preprocess_config = load_config(args.config, PreprocessConfigs)
    else:
        preprocess_config = PreprocessConfigs()

    all_files: list[str] = []
    for root, _, files in os.walk(preprocess_config.data_root_dir):
        for file in files:
            all_files.append(os.path.join(root, file))

    if not all_files:
        print(f"[Info] No files found in directory: '{preprocess_config.data_root_dir}'")
        return

    print_config_report(preprocess_config, len(all_files))

    os.makedirs(preprocess_config.preprocessed_dir, exist_ok=True)

    num_workers = max(1, int(args.num_workers))
    num_workers = min(num_workers, len(all_files))

    print(
        f">>> Starting dataset preprocess with {num_workers} workers: "
        + f"{preprocess_config.data_root_dir} -> {preprocess_config.preprocessed_dir}"
    )

    if preprocess_config.silence_trim:
        print(">>> Each worker loads its own VAD model.")
    else:
        print(">>> Silence trimming is off. VAD is not used for trimming.")

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
    print(" PREPROCESSING SUMMARY ")
    print("=" * 50)
    print(f" Scanned files       : {total_stats['scanned']}")
    print(f" Audio processed     : {total_stats['audio_processed']}")
    print(f" Audio skipped       : {total_stats['audio_skipped']}")
    print(f" Non-audio copied    : {total_stats['copied']}")
    print(f" Errors              : {total_stats['errors']}")
    print("=" * 50)

    if total_stats["errors"] == 0:
        print("\n>>> All tasks completed successfully!")
    else:
        print("\n>>> Finished with errors. Check logs above.")


if __name__ == "__main__":
    main()
