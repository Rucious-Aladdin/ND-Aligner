# ND-Aligner

A standalone neural forced aligner trained from paired speech and text, without
frame-level boundary labels.

## Requirements

- Python 3.13
- CUDA GPU and the **CUDA toolkit, including `nvcc`** — see below
- espeak-ng, for phonemization

```bash
git clone https://github.com/Rucious-Aladdin/ND-Aligner.git
cd ND-Aligner

sudo apt-get install espeak-ng
uv sync
```

### Compiled kernels

Two parts of the aligner are compiled rather than written in PyTorch.

The monotone forward–backward recursion is sequential over speech frames, so a
PyTorch implementation has to launch one kernel per frame. This repository
instead ships a CUDA extension (`nd_aligner/models/modules/forward_backward/`)
that keeps the entire recursion in a single launch, with one block per batch item
and threads parallelizing over the text axis. It also provides explicit backward
kernels for both `log_alpha` and `log_beta`, which avoid atomics by gathering
rather than scattering gradients. End to end this makes training about **7×
faster**. Viterbi decoding has a smaller C helper
(`nd_aligner/models/modules/mas/`) built with `gcc`.

Both are compiled on first use and then cached. PyTorch fallbacks exist for
each and are selected automatically when a build fails or when tensors are on
CPU; they are there for reference and debugging, and training on them is not
practical.

The CUDA build needs `nvcc` on `PATH`, at a version matching the one PyTorch was
built against:

```bash
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

If `nvcc` is missing, install the CUDA toolkit — the driver alone is not enough —
and point `CUDA_HOME` at it:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

See `nd_aligner/models/modules/forward_backward/README.md` for the recursions,
the gradient derivations, and the kernel layout.

## Data preparation

LJSpeech, VCTK, and LibriTTS are supported for training; TIMIT and Buckeye are
used for evaluation.

Point `DATA_PARENT_DIR` in `nd_aligner/config/data_config.py` at your data root,
then:

```bash
uv run python -m nd_aligner.preprocess.preprocess_audio

# Required for multi-speaker training
uv run python -m nd_aligner.preprocess.preprocess_spk_embeddings
```

Audio is resampled to 16 kHz and stored as 80-bin mel-spectrograms with a 10 ms
hop and a 25 ms window. Speaker embeddings come from Resemblyzer, so a single
utterance is enough to condition the model — no speaker-level grouping is
required at inference.

## Training

```bash
uv run python -m nd_aligner.train.train
```

Configuration is read from JSON files; see `nd_aligner/config/` for the
available fields. The diagonal prior weight is annealed to zero over the first
30k steps, after which the alignment is shaped by the CRF and reconstruction
objectives alone.

## Aligning speech

See `notebooks/alignment_demo.ipynb`.

## Evaluation

Baseline aligners are not vendored here. Each was run in its own environment and
scored with the same procedure; the notebooks under `notebooks/baselines/`
record how, and carry setup instructions at the top:

- `charsiu_en_w2v2_fc_10ms.ipynb`
- `maps_eval.ipynb`
- `mfa3_0.ipynb`
