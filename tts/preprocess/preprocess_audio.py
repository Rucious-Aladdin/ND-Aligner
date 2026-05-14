import argparse
import os
import shutil

from tqdm import tqdm

from tts.audio.audio_preprocessor import AudioPreprocessor
from tts.config.utils.io import load_config

from .config import PreprocessConfig


def print_config_report(preprocess_config: PreprocessConfig, total_files: int) -> None:
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
        f"   - LPF before VAD     : {preprocess_config.lpf_before_vad} (Cutoff: {preprocess_config.lpf_cutoff_freq}Hz)",
        f"   - Silence Margin     : {preprocess_config.silence_margin_sec} sec",
        f" RMS Normalization      : {preprocess_config.rms_normalize}",
        f"   - Target RMS         : {preprocess_config.rms_target}",
        "=" * 50 + "\n",
    ]
    print("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio dataset")
    parser.add_argument("-c", "--config", type=str, help="Path to data config JSON file")
    args = parser.parse_args()

    # Load global data config
    if args.config:
        print(f"📖 Loading data config from: {args.config}")
        preprocess_config = load_config(args.config, PreprocessConfig)
    else:
        preprocess_config = PreprocessConfig()

    # 1. Find ALL files recursively
    all_files = []
    for root, _, files in os.walk(preprocess_config.data_root_dir):
        for file in files:
            all_files.append(os.path.join(root, file))

    if not all_files:
        print(f"[Info] No files found in directory: '{preprocess_config.data_root_dir}'")
        return

    print_config_report(preprocess_config, len(all_files))

    print(">>> Loading VAD model...")
    preprocessor = AudioPreprocessor(preprocess_config)

    os.makedirs(preprocess_config.preprocessed_dir, exist_ok=True)

    audio_extensions = {".wav", ".flac"}

    print(
        f">>> Starting dataset preprocess: {preprocess_config.data_root_dir} -> {preprocess_config.preprocessed_dir}"
    )

    for p in tqdm(all_files, desc="Preprocessing dataset", unit="file"):
        rel_path = os.path.relpath(p, preprocess_config.data_root_dir)
        ext = os.path.splitext(p)[1].lower()

        if ext in audio_extensions:
            # 1. Process Audio
            base_rel_path, _ = os.path.splitext(rel_path)
            out_path = os.path.join(preprocess_config.preprocessed_dir, base_rel_path + ".wav")

            try:
                success = preprocessor.process_audio(p, out_path)
                if not success:
                    tqdm.write(f"[Warning] {rel_path}: no valid values in audio (skipped)")
            except Exception as e:
                tqdm.write(f"[Error] Failed to process {rel_path}: {e}")
            pass
        else:
            # 2. Copy Non-Audio files
            out_path = os.path.join(preprocess_config.preprocessed_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            try:
                shutil.copy2(p, out_path)
            except Exception as e:
                tqdm.write(f"[Error] Failed to copy {rel_path}: {e}")

    print("\n>>> All tasks completed successfully!")


if __name__ == "__main__":
    main()
