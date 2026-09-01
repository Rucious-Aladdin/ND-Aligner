# Buckeye data preparation

> **This procedure comes from the Montreal Forced Aligner project.**
>
> The pipeline follows
> **<https://github.com/MontrealCorpusTools/mfa-models/tree/main/scripts/alignment_benchmarks/data_prep>**,
> and `create_buckeye_benchmark.py` is taken from there unmodified. The
> corrections in `buckeye_fixes.patch` are theirs as well, redone here where
> their patch did not apply cleanly against our copy of the corpus.
>
> We add two scripts that the procedure needed to run end to end here:
> `flatten_buckeye.py`, since the patch assumes a flat corpus that the download
> does not produce, and `buckeye_to_timit_eval_format.py`, since our benchmarker
> scores TIMIT-style files.

The Buckeye Corpus is available from <https://buckeyecorpus.osu.edu/>, which requires an account. Download and extract the speaker zip files, then run the four steps below to reach the layout the TIMIT benchmarker expects.

```
buckeye corpus  ──flatten──▶  flat  ──patch──▶  corrected
                                                    │
                                       create_buckeye_benchmark.py
                                                    ▼
                                        benchmark/ + reference/
                                                    │
                                    buckeye_to_timit_eval_format.py
                                                    ▼
                                          TIMIT-style chunks
```

## 1. Flatten

Extraction leaves the corpus nested under speaker directories, with `.zip.extracted` markers alongside the recordings. The annotation patch is written against a flat layout, so flatten first:

```bash
python flatten_buckeye.py /path/to/buckeye/corpus /path/to/flat
```

Only `.wav`, `.words`, `.phones`, `.txt`, and `.log` files whose names match a Buckeye recording (`s0101a` and so on) are carried over. The script refuses to overwrite an existing file rather than silently merging two sources.

## 2. Apply the annotation patch

> **Use `buckeye_fixes.patch` from this directory, not the patch shipped with
> the MFA scripts.** Theirs did not apply cleanly against our copy of the
> corpus — the context lines failed to match on roughly thirty recordings — so
> those hunks were reapplied by hand and the whole thing re-exported. The
> corrections are the same ones; only the patch file differs.

```bash
cd /path/to/flat
git init
git add .
git commit -m "Initial commit"
git apply /path/to/buckeye_fixes.patch
```

The corrections are the MFA project's: formatting errors, misaligned sections, typos, and a reorganization of laughter and noise tags. Their README explains the reasoning, which is worth reading before using the corpus for anything other than alignment — they are tuned for alignment work and make choices a phonetic study might not want.

Committing before patching is what makes `git apply` work, and it also leaves the corrections inspectable afterwards with `git diff`.

## 3. Build the MFA benchmark layout

`create_buckeye_benchmark.py` is the MFA script, unmodified. It splits each recording into utterances and separates the annotations from the audio, producing a `benchmark/` directory to align and a `reference/` directory to score against. Both are grouped by speaker.

It needs `pysoundfile` and `praatio`:

```bash
python create_buckeye_benchmark.py \
    /path/to/flat \
    /path/to/benchmark \
    /path/to/reference
```

`benchmark/` is what you pass to `mfa align`; the two together are the input to
[MFA's alignment evaluation](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/implementations/alignment_evaluation.html#alignment-evaluation).

## 4. Convert to the TIMIT evaluation format (optional)

This step is ours and is not part of the MFA workflow. It exists only so that Buckeye can be scored by the same benchmarker as TIMIT, so use it or not as suits your setup.

```bash
python buckeye_to_timit_eval_format.py \
    /path/to/benchmark \
    /path/to/grid \
    --min-words 4
```

Each utterance in the benchmark TextGrids becomes its own chunk, written as a TIMIT-style triple:

```
grid/
  s01/
    s0101a_chunk_0000.wav
    s0101a_chunk_0000.txt
    s0101a_chunk_0000.WRD
    ...
```

Utterances containing special tokens, or shorter than `--min-words`, are dropped. Chunk timestamps in the `.WRD` files are relative to the chunk, not to the source recording, matching how TIMIT stores them.
