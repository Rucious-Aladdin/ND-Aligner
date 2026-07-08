import argparse
import glob
import os
from typing import Any

import librosa
import torch
from tqdm import tqdm

from tts.config.utils.io import load_config
from tts.models.modules.spk_encoder import ECAPASpeakerEncoder

from ..config.preprocess.preprocess_config import PreprocessConfigs


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess speaker embeddings")
    parser.add_argument("-c", "--config", type=str, help="Path to data config JSON file")
    args = parser.parse_args()
    return args


def main(args: Any):
    # Load global data config
    if args.config:
        print(f"📖 Loading data config from: {args.config}")
        preprocess_config = load_config(args.config, PreprocessConfigs)
    else:
        preprocess_config = PreprocessConfigs()

    print(">>> Initializing SpeakerEncoder (ECAPA-TDNN)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        speaker_encoder = ECAPASpeakerEncoder(device=device)
    except Exception as e:
        print(f"Error loading SpeakerEncoder: {e}. Cannot proceed without a valid model.")
        return  # Exit if model cannot be loaded

    print(f">>> Extracting speaker embeddings from: {preprocess_config.preprocessed_dir}")

    # Recursively find all .wav files in preprocessed directory
    wav_paths = glob.glob(
        os.path.join(preprocess_config.preprocessed_dir, f"**/*.wav"),
        recursive=True,
    )

    if not wav_paths:
        print(f"[Info] No WAV files found in directory: '{preprocess_config.preprocessed_dir}'")
        return

    for wav_path in tqdm(wav_paths, desc="Extracting Speaker Embeddings"):
        # Save as {wav_name}_spk.pt in the same folder as the wav file
        base_path, _ = os.path.splitext(wav_path)
        output_path = f"{base_path}_spk.pt"

        try:
            # Load with librosa and force 16kHz (SpeakerEncoder target)
            # Preprocessed files might already be at target SR, but SpeakerEncoder needs 16k
            wav_np, _ = librosa.load(wav_path, sr=16000)

            # Convert to torch tensor
            waveform = torch.from_numpy(wav_np).float()

            # SpeakerEncoder expects (B, T)
            waveform = waveform.unsqueeze(0)

            with torch.no_grad():
                # SpeakerEncoder.forward handles movement to device and squeezing logic
                embedding = speaker_encoder(waveform)

                # Save as squeezed 1D tensor (192,)
                torch.save(embedding.squeeze().cpu(), output_path)

        except Exception as e:
            print(f"Error processing {wav_path}: {e}")

    print(">>> Speaker embedding extraction completed.")


if __name__ == "__main__":
    main(parse_args())
