import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tts.config.ndaligner.data_config import DataConfig
from tts.config.utils.io import load_config
from tts.data.tts_datafactory import TTSDataFactory


def main():
    parser = argparse.ArgumentParser(
        description="Compute Mel-spectrogram statistics (sigma_data) for EDM"
    )
    parser.add_argument("-c", "--config", type=str, help="Path to data config JSON file")
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=1000,
        help="Number of mel samples to process (default: 1000)",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=64, help="Batch size for calculation (default: 64)"
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for dataloader (default: 8)",
    )
    args = parser.parse_args()

    # 1. Load Configuration
    if args.config:
        print(f"📖 Loading config from: {args.config}")
        config = load_config(args.config, DataConfig)
    else:
        print("ℹ️ No config provided, using default DataConfig.")
        config = DataConfig()

    try:
        # 2. Initialize Data Factory
        print(f"🏗️ Initializing data factory for datasets: {config.dataset.dataset_list}")
        factory = TTSDataFactory(config)

        # 3. Setup DataLoader (Optimized for calculation)
        loader = DataLoader(
            factory.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=factory.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        all_values = []
        samples_count = 0

        print(f"🚀 Sampling {args.num_samples} mel-spectrograms to compute sigma_data...")
        pbar = tqdm(total=args.num_samples, desc="Processing Mel")

        # 4. Accumulate statistics
        for batch in loader:
            # batch is TTSBatch
            # spec: (B, n_mels, T)
            # spec_lengths: (B,)

            spec = batch.spec
            lengths = batch.spec_lengths

            for i in range(spec.size(0)):
                if samples_count >= args.num_samples:
                    break

                valid_len = int(lengths[i].item())
                # Extract valid frames only
                valid_mel = spec[i, :, :valid_len]  # (n_mels, T_valid)

                # We flatten to (n_mels * T_valid) to compute global std
                all_values.append(valid_mel.reshape(-1))

                samples_count += 1
                pbar.update(1)

            if samples_count >= args.num_samples:
                break

        pbar.close()

        if not all_values:
            print("❌ No data found. Please check your dataset paths.")
            return

        # 5. Calculate Final Statistics
        print("📊 Calculating mean and standard deviation...")
        all_values_tensor = torch.cat(all_values)
        mean = all_values_tensor.mean().item()
        std = all_values_tensor.std().item()

        print("\n" + "=" * 45)
        print(f"MEL-SPECTROGRAM STATISTICS (N={samples_count})")
        print("-" * 45)
        print(f"{'Global Mean:':<25} {mean:.6f}")
        print(f"{'Global Std (sigma_data):':<25} {std:.6f}")
        print("=" * 45)
        print(f"\n💡 Recommendation: Update your EDM plan with sigma_data = {std:.4f}")

    except Exception as e:
        print(f"💥 Error during calculation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
    main()
