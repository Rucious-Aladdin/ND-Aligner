import argparse
from typing import Any

import librosa
import torch
from scipy.io.wavfile import write

from tts.config.stage1.data_config import DataConfig
from tts.config.stage1.model_config import MonotonicTTSConfigs
from tts.config.utils.io import load_config
from tts.models.init_monotonic_tts import init_monotonic_tts
from tts.tokenizer.text_tokenizer import TextTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Speaker Inference for Monotonic TTS")

    parser.add_argument(
        "--model_ckpt_path",
        type=str,
        required=True,
        help="checkpoint path for tts model",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        help="Path to model config JSON file. If not provided, default config is used.",
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        help="Path to data config JSON file. If not provided, default config is used.",
    )
    parser.add_argument(
        "--ref_audio_path",
        type=str,
        required=True,
        help="Path to reference audio file for speaker conditioning (16kHz mono recommended).",
    )
    parser.add_argument(
        "--script",
        type=str,
        help="text script for tts synthesize",
        default="Hello, My name is Monotonic TTS. Nice to meet you.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="output path for synthesized wav",
        default="./output.wav",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.667,
        help="noise scale for stochastic duration predictor",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use",
    )

    return parser.parse_args()


def main(args: Any):
    device = torch.device(args.device)
    print(f"🚀 Using device: {device}")

    # 1. Load Configurations
    if args.model_config_path:
        print(f"📖 Loading model config from: {args.model_config_path}")
        model_config = load_config(args.model_config_path, MonotonicTTSConfigs)
    else:
        model_config = MonotonicTTSConfigs()

    if args.data_config_path:
        print(f"📖 Loading data config from: {args.data_config_path}")
        data_config = load_config(args.data_config_path, DataConfig)
    else:
        data_config = DataConfig()

    # 2. Initialize Model
    print("🏗️ Initializing model...")
    # Load only necessary modules for inference
    model = init_monotonic_tts(
        config=model_config,
        load_spec_encoder=False,
        load_aligner=False,
        load_aux_decoder=True,
        load_vocoder=True,
        load_speaker_encoder=True,
        device=args.device,
    ).to(device)

    # 3. Load Checkpoint
    print(f"🔄 Loading TTS checkpoint from {args.model_ckpt_path}...")
    model.load_checkpoint(args.model_ckpt_path, device=device)
    model.eval()

    # 4. Prepare Reference Audio
    print(f"🎧 Loading reference audio from {args.ref_audio_path}...")
    # SpeakerEncoder (ECAPA-TDNN) expects 16kHz mono
    ref_wav, _ = librosa.load(args.ref_audio_path, sr=16000)
    ref_wav_tensor = torch.from_numpy(ref_wav).float().unsqueeze(0).to(device)

    # 5. Tokenize Script
    print(f"📝 Tokenizing script: '{args.script}'")
    tokenizer = TextTokenizer()
    x = tokenizer(args.script).to(device)
    x_lengths = torch.LongTensor([x.size(1)]).to(device)

    # 6. Inference
    print("🚀 Synthesizing...")
    with torch.no_grad():
        # Uses the newly added method that combines speaker embedding extraction and synthesis
        output = model.inference_from_ref_audio(
            x=x,
            x_lengths=x_lengths,
            ref_waveform=ref_wav_tensor,
            noise_scale=args.noise_scale,
            use_aux_dec=True,
        )

    # 7. Save Output
    if output.wav_hat is not None:
        wav = output.wav_hat.squeeze().cpu().numpy()
        # Scale if necessary, but Generator output is usually float32 in [-1, 1]
        write(args.output_path, data_config.audio.sr, wav)
        print(f"✅ Synthesized audio saved to {args.output_path}")
    else:
        print("❌ Error: Vocoder failed to generate waveform.")


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
