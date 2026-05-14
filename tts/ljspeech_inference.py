import argparse
import os
from typing import Any

import torch
import torchaudio
from scipy.io.wavfile import write

from tts.config.stage1.data_config import DataConfig
from tts.config.stage1.model_config import MonotonicTTSConfigs
from tts.models.init_monotonic_tts import init_monotonic_tts
from tts.tokenizer.text_tokenizer import TextTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for LJSpeech Dataset")

    parser.add_argument(
        "--model_ckpt_path",
        type=str,
        help="checkpoint path for tts model",
    )
    parser.add_argument(
        "--script",
        type=str,
        help="text script for tts synthesize",
        default="Hello, My name is Monotonic TTS.",
    )

    parser.add_argument(
        "--vocoder_ckpt_path",
        type=str,
        help="hifigan-vocoder checkpoint path",
        default="./checkpoints/hifigan/generator_v1",
    )
    parser.add_argument(
        "--vocoder_config_path",
        type=str,
        help="hifigan-vocoder config path (json)",
        default="./checkpoints/hifigan/config.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="output path for synthesized wav",
        default="./output.wav",
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
    config = DataConfig()
    model_config = MonotonicTTSConfigs()

    # 2. Compatibility Check
    print("🔍 Checking compatibility...")
    import json
    from dataclasses import replace

    # Update vocoder paths in config if provided via args
    vocoder_cfg = replace(
        model_config.vocoder, config_path=args.vocoder_config_path, ckpt_path=args.vocoder_ckpt_path
    )
    model_config = replace(model_config, vocoder=vocoder_cfg)

    # Check mel channels consistency
    with open(args.vocoder_config_path, "r") as f:
        v_cfg = json.load(f)
        v_mels = v_cfg.get("num_mels", 80)
        if v_mels != config.audio.n_mels:
            raise ValueError(
                f"Vocoder mel channels ({v_mels}) does not match TTS mel channels ({config.audio.n_mels})"
            )
        print(f"✅ Compatibility check passed: {v_mels} mel channels.")

    # 3. Initialize Model
    print("🏗️ Initializing model...")
    # Load only necessary modules for inference
    model = init_monotonic_tts(
        config=model_config,
        load_spec_encoder=False,
        load_aligner=False,
        load_aux_decoder=False,
        load_vocoder=True,
    ).to(device)

    # 4. Load Checkpoint
    print(f"🔄 Loading TTS checkpoint from {args.model_ckpt_path}...")
    model.load_checkpoint(args.model_ckpt_path, device=device)
    model.eval()

    # 5. Tokenize Script
    print(f"📝 Tokenizing script: '{args.script}'")
    tokenizer = TextTokenizer()
    x = tokenizer(args.script).to(device)
    x_lengths = torch.LongTensor([x.size(1)]).to(device)

    # 6. Prepare dummy speaker conditioning (LJSpeech is single speaker)
    # Based on model_config.py, SPK_COND_DIM = 192
    cond = torch.zeros(1, 192).to(device)

    # 7. Inference
    print("🚀 Synthesizing...")
    with torch.no_grad():
        output = model.inference(x, x_lengths, cond)

    # 8. Save Output
    if output.wav_hat is not None:
        wav = output.wav_hat.squeeze().cpu().numpy()
        # Scale if necessary, but Generator output is usually float32 in [-1, 1]
        write(args.output_path, config.audio.sr, wav)
        print(f"✅ Synthesized audio saved to {args.output_path}")
    else:
        print("❌ Error: Vocoder failed to generate waveform.")


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
