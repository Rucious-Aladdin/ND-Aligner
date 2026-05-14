# Monotonic TTS: Fast and Robust Text-to-Speech with Monotonic Alignment

### Robust and efficient end-to-end TTS architecture featuring Monotonic Alignment Search (MAS) and FiLM-conditioned text encoding.

> This project implements a high-performance Text-to-Speech (TTS) system designed for stability and speed. By utilizing a Monotonic Alignment Network alongside a flexible Text Encoder (supporting Conv, Transformer, and Conformer architectures), the model achieves precise character-to-spectrogram alignment. The architecture incorporates FiLM (Feature-wise Linear Modulation) for effective speaker conditioning and a stochastic duration predictor to capture the natural variance of human speech.

## TODO
- [x] Implementation of Monotonic Alignment Network
- [x] FiLM-conditioned Conv Text Encoder with configurable kernel sizes
- [x] Multi-speaker support with Speaker Encoder (ECAPA-TDNN)
- [x] Integration with HiFi-GAN vocoder for high-quality audio synthesis
- [ ] Support for streaming inference
- [ ] Distributed Data Parallel (DDP) training optimization

## Pre-requisites
1. Python >= 3.12 (Optimized for 3.13)
2. Clone this repository:
```bash
git clone https://github.com/your-username/monotonic_tts.git
cd monotonic_tts
```
3. Install dependencies using [uv](https://github.com/astral-sh/uv):
```bash
uv sync
```
4. Install system dependencies for audio processing (phonemizer requirement):
```bash
sudo apt-get install espeak-ng
```

## Dataset Preparation
The project supports LJSpeech, VCTK, and LibriTTS datasets. 
- Update `DATA_PARENT_DIR` in `tts/config/data_config.py` to point to your preprocessed data root.
- The training script expects preprocessed datasets (mel-spectrograms, etc.) at the specified paths.

## Training
The training process uses JSON configuration files for flexible setup.

### Training Stage 1 (Alignment and Basic Synthesis)
```bash
# Using default configurations
python -m tts.train_first

# Using custom configurations
python -m tts.train_first -c path/to/data_config.json -m path/to/model_config.json
```

### Training Stage 2 (Planned: Diffusion-based Decoding)
In the next stage, we plan to implement a diffusion-based refinement process:
- **Freeze**: `MonotonicAlignmentNet`, `TextEncoder`, and `SpecEncoder`.
- **Train**: A high-fidelity Mel-Decoder using **Karras SDE/ODE** (Score-based Diffusion Models) for improved naturalness and detail.
Checkpoints and Tensorboard logs will be saved in the `./runs` directory (or as specified in your config). Monitor progress with:
```bash
tensorboard --logdir ./runs
```

### Important Configurations
In `tts/config/model_config.py`:
- `encoder_type`: `"conv"`, `"transformer"`, or `"conformer"`.
- `kernel_sizes`: For the `"conv"` encoder, use a list like `[3, 3, 3, 1, 1]`.

In `tts/config/data_config.py`:
- `batch_size`: Adjust based on GPU VRAM.
- `fp16_run`: Set to `True` for mixed precision training.

## Inference
Synthesize speech using the specialized inference scripts.

### Single-speaker (LJSpeech)
```bash
python -m tts.ljspeech_inference \
    --model_ckpt_path ./checkpoints/ckpt_step_31000.pth \
    --script "Hello, this is a monotonic tts system." \
    --output_path ./output_ljspeech.wav
```

### Multi-speaker (Zero-shot via Reference Audio)
The multi-speaker model uses a reference audio to extract speaker embeddings (ECAPA-TDNN).
```bash
python -m tts.multi_speaker_inference \
    --model_ckpt_path ./checkpoints/ckpt_step_31000.pth \
    --ref_audio_path ./path/to/reference_audio.wav \
    --script "Welcome to the world of speech synthesis." \
    --output_path ./output_multi.wav
```

## References
- [StyleTTS 2](https://github.com/yl4579/StyleTTS2)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [Monotonic Alignment Search (VITS)](https://github.com/jaywalnut310/vits)

## License
MIT License
