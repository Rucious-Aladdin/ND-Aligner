# NDAligner


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
The project supports LJSpeech, VCTK, and LibriTTS datasets. (and evaluations for TIMIT, BuckEye.)
- Update `DATA_PARENT_DIR` in `tts/config/data_config.py` to point to your preprocessed data root.
- The training script expects preprocessed datasets (mel-spectrograms, etc.) at the specified paths.

## Preprocessing
```bash
uv run python -m tts.preprocess.preprocess_audio

# If you use multi-speaker settings, MUST DO:
uv run python -m tts.prepreocess.preprocess_spk_embeddings
```


## Training
The training process uses JSON configuration files for flexible setup.

```bash
# Using default configurations
uv run python -m tts.train.train_nd_aligner
```

## License
MIT License
