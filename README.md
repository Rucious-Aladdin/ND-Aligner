# ND-Aligner

A neural forced aligner trained from paired speech and text, without frame-level boundary labels.

## Results

Word-boundary accuracy on the TIMIT test partition and on Buckeye. **WBE** is the mean absolute distance in milliseconds from each reference word boundary to the predicted one; **P25** and **P50** are the percentages of reference boundaries within 25 ms and 50 ms. **BL** is whether boundary labels were used in training: manual, pseudo-labels from another aligner, or none.

| System                            | BL     | TIMIT WBE ↓ | P25 ↑    | P50 ↑    | Buckeye WBE ↓ | P25 ↑    | P50 ↑    |
| --------------------------------- | ------ | ----------- | -------- | -------- | ------------- | -------- | -------- |
| MFA ARPA 3.0 (HMM–GMM)            | none   | 18.7        | 76.6     | 92.4     | 21.5          | 76.7     | 91.7     |
| MAPS                              | manual | 21.4        | 76.8     | 89.9     | –             | –        | –        |
| Charsiu (W2V2-FC-10ms)            | pseudo | 25.1        | 66.8     | 88.3     | 29.2          | 69.0     | 87.4     |
| **ND-Aligner** (VCTK)             | none   | 17.4        | 79.3     | 93.1     | 28.9          | 71.8     | 87.2     |
| **ND-Aligner** (VCTK+LibriSpeech) | none   | **16.6**    | **79.9** | **93.4** | **19.7**      | **79.0** | **92.3** |

Trained on paired speech and text alone, ND-Aligner is ahead of MFA 3.0 by 2.1 ms on TIMIT and 1.8 ms on Buckeye. MAPS is trained on manual TIMIT boundaries and is reported on TIMIT only, as it uses Buckeye in training; Charsiu is trained on MFA alignments. ND-Aligner scores 98.0% of TIMIT and 98.2% of Buckeye reference boundaries against essentially 100% for the baselines, because eSpeak's connected-speech rules occasionally merge adjacent words.

Both halves of the asymmetry earn their place. Widening the decoder's receptive field from 5 to 13 frames costs 4.1 ms of word-boundary error, and restricting the alignment scorer to 1×1 kernels — removing its context while keeping it learned — costs 4.9 ms.

## Alignment examples

**<https://rucious-aladdin.github.io/ND-Aligner/>** shows word and phone alignments the model produced for TIMIT, Buckeye, and LibriSpeech utterances, with each word clipped to its predicted boundaries so you can listen to them one at a time. Nothing there is hand-corrected. The page is built from `docs/` by `docs/build_demo.py`.

## Requirements

- Python 3.14
- CUDA GPU and the **CUDA toolkit, including `nvcc`** — see below
- A C compiler (`clang` or `gcc`)
- espeak-ng, for phonemization

```bash
git clone https://github.com/Rucious-Aladdin/ND-Aligner.git
cd ND-Aligner

sudo apt-get install espeak-ng
uv sync --locked

scripts/build_fb_kernel.sh
scripts/build_mas_dp.sh
```

### Compiled kernels for Training & Inference Acceleration

Two parts of the aligner are compiled rather than written in PyTorch.

The monotone forward–backward recursion is sequential over speech frames, so a PyTorch implementation has to launch one kernel per frame. This repository instead ships a CUDA extension (`nd_aligner/models/modules/forward_backward/`) that keeps the entire recursion in a single launch, with one block per batch item and threads parallelizing over the text axis. It also provides explicit backward kernels for both `log_alpha` and `log_beta`, which avoid atomics by gathering rather than scattering gradients. End to end this makes training about **7× faster**. Viterbi decoding has a smaller C helper (`nd_aligner/models/modules/mas/`).

PyTorch fallbacks exist for both and are selected automatically when a build is unavailable or when tensors are on CPU. They are there for reference and debugging; training on them is not practical.

`scripts/build_fb_kernel.sh` builds the CUDA extension. It resolves `CUDA_HOME` from the `nvidia-cu13` wheel installed by `uv sync`, so a system CUDA toolkit is not required, and it reads the target architectures from `nvidia-smi` unless `TORCH_CUDA_ARCH_LIST` is already set. Pass `--test` to run the kernel test suite
immediately after building:

```bash
scripts/build_fb_kernel.sh --test
```

`scripts/build_mas_dp.sh` compiles `viterbi_dp.c` into a shared library and loads it once to check the symbol. It uses `clang` if available and `gcc` otherwise; override with `CC` or adjust flags with `MAS_CFLAGS`:
```bash
CC=gcc MAS_CFLAGS="-O2" scripts/build_mas_dp.sh
```

If the CUDA build fails, check that `nvcc` is reachable and that its version matches the one PyTorch was built against:

```bash
nvcc --version
uv run python -c "import torch; print(torch.version.cuda)"
```

See `nd_aligner/models/modules/forward_backward/README.md` for what the kernels compute and how to call them, and the derivation note alongside it for the recursions and their gradients.

## Data preparation

LJSpeech, VCTK, and LibriTTS are supported for training; TIMIT and Buckeye are used for evaluation.

Point `DATA_PARENT_DIR` in `nd_aligner/config/ndaligner/data_config.py` at your data root, then:

```bash
uv run python -m nd_aligner.preprocess.preprocess_audio

# Required for multi-speaker training
uv run python -m nd_aligner.preprocess.preprocess_spk_embeddings
```

## Training

> **Training evaluates on TIMIT by default.** Every `train_time_eval_per_step` steps it aligns a held-out TIMIT subset, logs the
> word-boundary error, and keeps checkpoints by that metric. If you do not have TIMIT, set `train_time_eval_logging = False` in `ExperimentConfigs`
> (`nd_aligner/config/ndaligner/data_config.py`) before starting, or training will fail on the missing `timit_val_root_dir`. Training itself does not use
> TIMIT; the paths there are only for this monitoring.

```bash
uv run python -m nd_aligner.train.train
```

Configuration is read from JSON files; see `nd_aligner/config/` for the available fields.

Losses go to TensorBoard throughout training, along with the TIMIT word-boundary metrics when the evaluation above is enabled.

![Training curves and TIMIT word-boundary metrics](assets/train.png)

## Pretrained checkpoints

The two models from the results table are in `checkpoints/ndaligner/v2.3/`, each with the config files needed to rebuild it:

| Directory           | Trained on                 | TIMIT WBE |
| ------------------- | -------------------------- | --------- |
| `VCTK/`             | VCTK (44 h)                | 17.4 ms   |
| `VCTK+LibriSpeech/` | VCTK + LibriSpeech (960 h) | 16.6 ms   |

Use `VCTK+LibriSpeech/` unless you have a reason not to; it is the stronger of the two, and by a wide margin on spontaneous speech.

## Aligning speech

`demo/alignment_demo.ipynb` loads one of those checkpoints, runs it on a waveform, and writes a Praat TextGrid with a word tier and a phone tier.

![Word and phone alignment for a TIMIT utterance](assets/word_grid.png)

Word boundaries are recovered from the token sequence, so the phone tier carries the character-level IPA units the model aligns rather than a phone inventory.

## Evaluation

`nd_aligner/benchmark/alignment_benchmark_test.ipynb` scores a checkpoint against TIMIT-style boundary annotations and reports the WBE and Pn numbers above. Buckeye has to be converted into that format first; `nd_aligner/benchmark/buckeye/README.md` walks through the preparation, which follows the Montreal Forced Aligner project's benchmark pipeline so that the scores are comparable to theirs.

Baseline aligners are not vendored here. Each was run in its own environment and scored with the same procedure; the notebooks under `nd_aligner/benchmark/baseline_evals/` record how, and carry setup instructions at the top:

- `charsiu_en_w2v2_fc_10ms_eval.ipynb`
- `maps_eval.ipynb`
- `mfa3.0_arpa_eval.ipynb`

## License

MIT — see [LICENSE](LICENSE).
