# NDAligner

## Pre-requisites

1. Python 3.14
2. Clone this repository:
   ```bash
   $ git clone https://github.com/Rucious-Aladdin/ND-Aligner.git && cd ND-Aligner
   ```
3. Install dependencies using [uv](https://docs.astral.sh/uv/):
   ```bash
   $ uv sync --locked
   ```
4. Install system dependencies for audio processing (phonemizer requirement):
   ```bash
   $ sudo apt-get install espeak-ng
   ```
5. Build C/CUDA extensions:
   ```bash
   $ scripts/build_fb_kernel.sh
   $ scripts/build_mas_dp.sh
   ```

## Dataset Preparation

The project supports LJSpeech, VCTK, and LibriTTS datasets. (and evaluations for TIMIT, BuckEye.)

- Update `DATA_PARENT_DIR` in `nd_aligner/config/ndaligner/data_config.py` to point to your preprocessed data root.
- The training script expects preprocessed datasets (mel-spectrograms, etc.) at the specified paths.

## Preprocessing

```bash
$ uv run python -m nd_aligner.preprocess.preprocess_audio

# If you use multi-speaker settings, MUST DO:
$ uv run python -m nd_aligner.preprocess.preprocess_spk_embeddings
```

## Training

The training process uses JSON configuration files for flexible setup.

```bash
# Using default configurations
$ uv run python -m nd_aligner.train.train
```

## License

MIT License
