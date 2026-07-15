from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import torch
import torchaudio.functional as AF
from tqdm import tqdm

from tts.models.modules.hifigan_vocoder import Generator
from tts.models.ndaligner import AlignerFeatures, NDAligner
from tts.models.utils.input_maker import AlignerInputMaker
from tts.models.utils.lev_words_mapper import (
    LevensteinWordsMapper,
    MatchedWords,
    WordSegment,
)

from ..utils.entropy import compute_framewise_entropy
from ..utils.mcd_dtw import compute_mcd_dtw


class TIMITMetrics(NamedTuple):
    word_boundary_error: float  # MAE in seconds
    p_word_10ms: float  # Accuracy % within 10ms
    p_word_25ms: float  # Accuracy % within 25ms
    p_word_50ms: float  # Accuracy % within 50ms
    p_word_100ms: float  # Accuracy % within 100ms

    mcd_dtw: float
    posterior_entropy: float


class TIMITSampleResult(NamedTuple):
    boundary_errors: list[float]
    mcd_dtw: float | None
    posterior_entropy: float | None


class TIMITBenchMarker:
    """
    TIMIT word-boundary benchmark for ND-Aligner.

    This refactored version delegates all model-input preprocessing to
    AlignerInputMaker. The benchmark itself only performs:
      - TIMIT WRD/TXT bookkeeping
      - aligner execution
      - token-to-word mapping
      - boundary-error, entropy, and optional MCD-DTW evaluation
    """

    def __init__(
        self,
        root_dir: str | Path,
        ref_audio_sr: int,
        hyp_audio_sr: int,
        hyp_hop_length: int,
        input_maker: AlignerInputMaker,
        hyp_ignore_symbols: set[str] | None = None,
        max_ref_words_per_hyp_word: int = 5,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.ref_audio_sr = int(ref_audio_sr)
        self.hyp_audio_sr = int(hyp_audio_sr)
        self.hyp_hop_length = int(hyp_hop_length)
        self.input_maker = input_maker
        self.tokenizer = input_maker.tokenizer

        self.word_mapper = LevensteinWordsMapper(
            tokenizer=self.tokenizer,
            hyp_ignore_symbols=hyp_ignore_symbols,
            max_ref_words_per_hyp_word=max_ref_words_per_hyp_word,
        )
        self.seed = seed

        wav_paths = sorted(list(self.root_dir.rglob("*.WAV")) + list(self.root_dir.rglob("*.wav")))
        triplets: list[tuple[Path, Path, Path]] = []
        for wav_path in wav_paths:
            wrd_path = wav_path.with_suffix(".WRD")
            if not wrd_path.exists():
                wrd_path = wav_path.with_suffix(".wrd")

            txt_path = wav_path.with_suffix(".TXT")
            if not txt_path.exists():
                txt_path = wav_path.with_suffix(".txt")

            if wrd_path.exists() and txt_path.exists():
                triplets.append((wrd_path, wav_path, txt_path))

        random.Random(self.seed).shuffle(triplets)
        self.triplets = triplets
        print(f"[TIMITBenchMarker] Found {len(self.triplets)} valid (WRD, WAV, TXT) triplets.")

    @torch.no_grad()
    def __call__(
        self,
        aligner: NDAligner,
        vocoder: Generator | None = None,
        max_test_samples: int | None = None,
        save_align_figure: bool = False,
        align_figure_dir: str | Path | None = None,
        is_test: bool = False,
    ) -> TIMITMetrics:
        was_aligner_training = aligner.training
        was_input_maker_training = self.input_maker.training
        was_vocoder_training = vocoder.training if vocoder is not None else None

        # speaker_encoder is accepted only for backward-compatible call sites.
        # Preprocessing now uses self.input_maker.speaker_encoder internally.

        device = next(aligner.parameters()).device
        self.input_maker.to(device=device)

        try:
            aligner.eval()
            self.input_maker.eval()
            if vocoder is not None:
                vocoder.eval()

            triplets = (
                self.triplets if max_test_samples is None else self.triplets[:max_test_samples]
            )

            all_boundary_errors: list[float] = []
            all_mcd_dtw: list[float] = []
            all_entropy: list[float] = []

            for wrd_path, wav_path, txt_path in tqdm(
                triplets,
                desc="Computing Alignments",
                total=len(triplets),
            ):
                result = self.run_one(
                    txt_path=txt_path,
                    wrd_path=wrd_path,
                    wav_path=wav_path,
                    aligner=aligner,
                    vocoder=vocoder,
                    save_align_figure=save_align_figure,
                    align_figure_dir=align_figure_dir,
                    is_test=is_test,
                )

                all_boundary_errors.extend(result.boundary_errors)
                if result.posterior_entropy is not None:
                    all_entropy.append(result.posterior_entropy)
                if result.mcd_dtw is not None:
                    all_mcd_dtw.append(result.mcd_dtw)

            if not all_boundary_errors:
                return TIMITMetrics(
                    word_boundary_error=0.0,
                    p_word_10ms=0.0,
                    p_word_25ms=0.0,
                    p_word_50ms=0.0,
                    p_word_100ms=0.0,
                    mcd_dtw=float("nan"),
                    posterior_entropy=float("nan"),
                )

            err_tensor = torch.tensor(all_boundary_errors, dtype=torch.float32)
            mae = err_tensor.mean().item()
            p_10ms = (err_tensor <= 0.010).float().mean().item() * 100.0
            p_25ms = (err_tensor <= 0.025).float().mean().item() * 100.0
            p_50ms = (err_tensor <= 0.050).float().mean().item() * 100.0
            p_100ms = (err_tensor <= 0.100).float().mean().item() * 100.0

            mcd_dtw = (
                torch.tensor(all_mcd_dtw, dtype=torch.float32).mean().item()
                if all_mcd_dtw
                else float("nan")
            )
            posterior_entropy = (
                torch.tensor(all_entropy, dtype=torch.float32).mean().item()
                if all_entropy
                else float("nan")
            )

            return TIMITMetrics(
                word_boundary_error=mae,
                p_word_10ms=p_10ms,
                p_word_25ms=p_25ms,
                p_word_50ms=p_50ms,
                p_word_100ms=p_100ms,
                mcd_dtw=mcd_dtw,
                posterior_entropy=posterior_entropy,
            )

        finally:
            if was_aligner_training:
                aligner.train()
            if was_input_maker_training:
                self.input_maker.train()
            if vocoder is not None and was_vocoder_training:
                vocoder.train()

    @torch.no_grad()
    def run_one(
        self,
        txt_path: Path | str,
        wrd_path: Path | str,
        wav_path: Path | str,
        aligner: NDAligner,
        vocoder: Generator | None = None,
        save_align_figure: bool = False,
        align_figure_dir: str | Path | None = None,
        is_test: bool = False,
    ) -> TIMITSampleResult:
        aligner.eval()
        self.input_maker.eval()

        txt_path = Path(txt_path)
        wrd_path = Path(wrd_path)
        wav_path = Path(wav_path)

        wrd_segments = self._read_wrd(wrd_path)
        ref_words = [seg.word for seg in wrd_segments]
        text = self._read_txt(txt_path)

        batch = self.input_maker.make_with_audio(
            wav_paths=[wav_path],
            scripts=[text],
        )

        wav_16k_start_offset = int(batch.wav_16k_start_offset[0].item())
        _wav_16k_end_offset = int(batch.wav_16k_end_offset[0].item())

        cond = (
            torch.tensor(
                0.0,
                device=batch.x.device,
            )
            if batch.cond is None
            else batch.cond
        )

        features = aligner.compute_alignments(
            x=batch.x,
            x_lengths=batch.x_lengths,
            y=batch.y,
            y_lengths=batch.y_lengths,
            cond=cond,
            compute_soft_path=is_test,
            compute_hard_path=True,
        )

        assert features.hard_dur is not None
        hard_dur = features.hard_dur[0].detach().cpu().tolist()

        text_len = int(batch.x_lengths[0].item())
        token_ids_1d = batch.x[0, :text_len].detach().cpu()
        hyp_symbols_raw = self.tokenizer.decode_to_symbols(token_ids_1d)

        if len(hyp_symbols_raw) > 0 and isinstance(hyp_symbols_raw[0], list):
            raise ValueError("Expected 1D token symbols, got batched symbols.")

        hyp_symbols = [str(sym) for sym in hyp_symbols_raw]
        if len(hyp_symbols) != len(hard_dur):
            raise RuntimeError(
                "Token-symbol / duration length mismatch.\n"
                + f"len(hyp_symbols)={len(hyp_symbols)}, len(hard_dur)={len(hard_dur)}\n"
                + f"decoded={self.tokenizer.decode(token_ids_1d)!r}\n"
                + f"symbols={hyp_symbols}\n"
            )

        matched = self.word_mapper(ref_seqs=ref_words, hyp_seqs=hyp_symbols)

        if save_align_figure:
            save_path = self._make_alignment_figure_path(
                wav_path=wav_path,
                align_figure_dir=align_figure_dir,
            )
            self._save_alignment_figure(
                save_path=save_path,
                wrd_segments=wrd_segments,
                matched_words=matched,
                hyp_symbols=hyp_symbols,
                hard_dur=hard_dur,
                features=features,
                t_mel=batch.y.size(2),
                wav_16k_start_offset=wav_16k_start_offset,
            )

        boundary_errors = self._get_eval_instance(
            wrd_segments=wrd_segments,
            matched_words=matched,
            hard_dur=hard_dur,
            wav_16k_start_offset=wav_16k_start_offset,
        )

        spec_mask = self._make_spec_mask(
            y_lengths=batch.y_lengths,
            t_spec=batch.y.size(2),
            dtype=batch.y.dtype,
            device=batch.y.device,
        )

        mcd_dtw = posterior_entropy = None
        if is_test:
            posterior_entropy = compute_framewise_entropy(
                attn=features.soft_attn,
                mask=spec_mask,
            ).item()

            if vocoder is not None:
                if self.ref_audio_sr != 16000:
                    raise ValueError(
                        "This simplified TIMIT path assumes ref_audio_sr == 16000 "
                        + f"for MCD-DTW, got {self.ref_audio_sr}."
                    )

                mcd_dtw = self._compute_reconstruction_mcd_dtw(
                    aligner=aligner,
                    vocoder=vocoder,
                    features=features,
                    spec_mask=spec_mask,
                    ref_wav_tensor=batch.wav_16k,
                    cond=batch.cond,
                    device=str(batch.y.device),
                )

        return TIMITSampleResult(
            boundary_errors=boundary_errors,
            mcd_dtw=mcd_dtw,
            posterior_entropy=posterior_entropy,
        )

    @torch.no_grad()
    def _compute_reconstruction_mcd_dtw(
        self,
        aligner: NDAligner,
        vocoder: Generator,
        features: AlignerFeatures,
        spec_mask: torch.Tensor,
        ref_wav_tensor: torch.Tensor,
        cond: torch.Tensor | None,
        device: str,
    ) -> float:
        def normalize_wav_shape(wav: torch.Tensor) -> torch.Tensor:
            if wav.dim() == 3:
                if wav.size(1) == 1:
                    wav = wav.squeeze(1)
                elif wav.size(2) == 1:
                    wav = wav.squeeze(2)
                else:
                    raise ValueError(f"Expected mono vocoder output, got {tuple(wav.shape)}.")
            if wav.dim() != 2:
                raise ValueError(f"wav must have shape (B, T), got {tuple(wav.shape)}.")
            return wav

        assert aligner.spec_decoder is not None

        vocoder = vocoder.to(device)
        vocoder.eval()

        decoder_input = torch.bmm(
            features.soft_attn,
            features.h_text.transpose(1, 2),
        )  # (B, T_mel, C_text)

        recon_mel = aligner.spec_decoder(
            x=decoder_input,
            cond=cond,
            mask=spec_mask,
        )  # (B, n_mels, T_mel)

        hyp_wav = vocoder(recon_mel)
        hyp_wav = normalize_wav_shape(hyp_wav)

        # HiFi-GAN vocoder is assumed to output 22050 Hz.
        hyp_wav = AF.resample(
            waveform=hyp_wav,
            orig_freq=22050,
            new_freq=self.ref_audio_sr,
        )

        ref_wav = normalize_wav_shape(ref_wav_tensor)
        ref_mask = torch.ones_like(ref_wav, dtype=ref_wav.dtype, device=ref_wav.device)
        hyp_mask = torch.ones_like(hyp_wav, dtype=hyp_wav.dtype, device=hyp_wav.device)

        score = compute_mcd_dtw(
            ref_wavs=ref_wav,
            ref_wav_mask=ref_mask,
            hyp_wavs=hyp_wav,
            hyp_wav_mask=hyp_mask,
            sampling_rate=16000,
            exclude_c0=True,
        )
        return score.mean().item()

    def _get_eval_instance(
        self,
        wrd_segments: list[WordSegment],
        matched_words: MatchedWords,
        hard_dur: list[int],
        wav_16k_start_offset: int,
    ) -> list[float]:
        sec_per_frame = self.hyp_hop_length / self.hyp_audio_sr

        # wav_16k offset은 항상 16-kHz sample 좌표이다.
        start_offset_sec = wav_16k_start_offset / 16000.0

        token_starts: list[float] = []
        token_ends: list[float] = []
        current_frame = 0

        for dur in hard_dur:
            start_sec = max(0.0, current_frame * sec_per_frame)
            current_frame += int(dur)
            end_sec = max(0.0, current_frame * sec_per_frame)

            token_starts.append(start_sec)
            token_ends.append(end_sec)

        errors: list[float] = []

        for ref_slice, hyp_slice in zip(
            matched_words.ref_matched_indices,
            matched_words.hyp_matched_indices,
            strict=True,
        ):
            if ref_slice.start is None or ref_slice.stop is None:
                raise ValueError(f"Invalid ref slice: {ref_slice}")

            if hyp_slice.start is None or hyp_slice.stop is None:
                raise ValueError(f"Invalid hyp slice: {hyp_slice}")

            gt_start = (
                wrd_segments[ref_slice.start].start_sample / self.ref_audio_sr - start_offset_sec
            )
            gt_end = (
                wrd_segments[ref_slice.stop - 1].end_sample / self.ref_audio_sr - start_offset_sec
            )

            pred_start = token_starts[hyp_slice.start]
            pred_end = token_ends[hyp_slice.stop - 1]

            errors.append(abs(pred_start - gt_start))
            errors.append(abs(pred_end - gt_end))

        return errors

    @staticmethod
    def _read_wrd(wrd_path: Path) -> list[WordSegment]:
        segments: list[WordSegment] = []
        with open(wrd_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=2)
                segments.append(
                    WordSegment(
                        start_sample=int(parts[0]),
                        end_sample=int(parts[1]),
                        word=parts[2],
                    )
                )
        return segments

    @staticmethod
    def _read_txt(txt_path: Path) -> str:
        """
        Read a TIMIT TXT transcript.

        TIMIT TXT lines have the form:
            <start_sample> <end_sample> <text>
        Only <text> is passed to AlignerInputMaker.
        """
        with open(txt_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()

        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            return parts[2]
        return line

    ## Optional visualization private methods ##

    def _make_alignment_figure_path(
        self,
        wav_path: Path,
        align_figure_dir: str | Path | None,
    ) -> Path:
        figure_dir = (
            Path(align_figure_dir)
            if align_figure_dir is not None
            else Path("./timit_align_figures")
        )

        try:
            rel_stem = wav_path.relative_to(self.root_dir).with_suffix("").as_posix()
        except ValueError:
            rel_stem = wav_path.with_suffix("").name

        safe_stem = rel_stem.replace("/", "__").replace("\\", "__")
        return figure_dir / f"{safe_stem}_alignment.png"

    def _save_alignment_figure(
        self,
        save_path: str | Path,
        wrd_segments: list[WordSegment],
        matched_words: MatchedWords,
        hyp_symbols: list[str],
        hard_dur: list[int],
        features: AlignerFeatures,
        t_mel: int,
        wav_16k_start_offset: int,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        raw_align = self._extract_viterbi_alignment_matrix(
            features=features,
            hard_dur=hard_dur,
            t_text=len(hyp_symbols),
            t_mel=t_mel,
        )

        gt_mat, gt_labels = self._build_ground_truth_word_matrix(
            wrd_segments=wrd_segments,
            t_mel=t_mel,
            wav_16k_start_offset=wav_16k_start_offset,
        )

        collapsed_mat, collapsed_labels = self._build_collapsed_word_matrix(
            raw_align=raw_align,
            hyp_symbols=hyp_symbols,
            matched_words=matched_words,
        )

        raw_labels = [self._pretty_token_label(sym) for sym in hyp_symbols]
        total_rows = len(gt_labels) + len(collapsed_labels) + len(raw_labels)
        fig_height = max(9.0, min(36.0, 0.18 * total_rows))

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(18.0, fig_height),
            constrained_layout=True,
        )

        self._plot_alignment_matrix(
            ax=axes[0],
            mat=gt_mat,
            y_labels=gt_labels,
            title="Ground Truth (TIMIT words)",
        )
        self._plot_alignment_matrix(
            ax=axes[1],
            mat=collapsed_mat,
            y_labels=collapsed_labels,
            title="Model Alignment (matched word spans collapsed)",
        )
        self._plot_alignment_matrix(
            ax=axes[2],
            mat=raw_align,
            y_labels=raw_labels,
            title="Raw Model Alignment (TextTokenizer output)",
        )
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _extract_viterbi_alignment_matrix(
        self,
        features: AlignerFeatures,
        hard_dur: list[int],
        t_text: int,
        t_mel: int,
    ) -> torch.Tensor:
        for attr_name in ("viterbi_path", "hard_path", "path", "attn_hard"):
            value = getattr(features, attr_name, None)
            mat = self._coerce_path_tensor_to_alignment_matrix(
                value=value,
                t_text=t_text,
                t_mel=t_mel,
            )
            if mat is not None:
                return mat

        return self._hard_dur_to_alignment_matrix(
            hard_dur=hard_dur,
            t_text=t_text,
            t_mel=t_mel,
        )

    @staticmethod
    def _coerce_path_tensor_to_alignment_matrix(
        value: object,
        t_text: int,
        t_mel: int,
    ) -> torch.Tensor | None:
        if not isinstance(value, torch.Tensor):
            return None

        path = value.detach().float().cpu()
        while path.dim() > 2 and path.size(0) == 1:
            path = path.squeeze(0)

        if path.dim() == 2:
            if path.shape == (t_text, t_mel):
                return path
            if path.shape == (t_mel, t_text):
                return path.transpose(0, 1)
            if path.size(0) == 1 and path.numel() == t_text * t_mel:
                return path.reshape(t_text, t_mel)
            if path.size(1) == 1 and path.numel() == t_text * t_mel:
                return path.reshape(t_text, t_mel)

        if path.dim() == 1 and path.numel() == t_mel:
            indices = path.long().clamp(min=0, max=t_text - 1)
            mat = torch.zeros(t_text, t_mel, dtype=torch.float32)
            mat[indices, torch.arange(t_mel)] = 1.0
            return mat

        return None

    @staticmethod
    def _hard_dur_to_alignment_matrix(
        hard_dur: list[int],
        t_text: int,
        t_mel: int,
    ) -> torch.Tensor:
        mat = torch.zeros(t_text, t_mel, dtype=torch.float32)
        cur = 0
        for text_idx, dur in enumerate(hard_dur[:t_text]):
            dur = int(dur)
            if dur <= 0:
                continue
            end = min(t_mel, cur + dur)
            if cur < t_mel and end > cur:
                mat[text_idx, cur:end] = 1.0
            cur += dur
        return mat

    def _build_ground_truth_word_matrix(
        self,
        wrd_segments: list[WordSegment],
        t_mel: int,
        wav_16k_start_offset: int,
    ) -> tuple[torch.Tensor, list[str]]:
        sec_per_frame = self.hyp_hop_length / self.hyp_audio_sr
        start_offset_sec = wav_16k_start_offset / 16000.0
        total_duration_sec = t_mel * sec_per_frame

        mat = torch.zeros(
            len(wrd_segments),
            t_mel,
            dtype=torch.float32,
        )
        labels: list[str] = []

        for row_idx, seg in enumerate(wrd_segments):
            # Original TIMIT coordinates -> trimmed waveform coordinates
            start_sec = seg.start_sample / self.ref_audio_sr - start_offset_sec
            end_sec = seg.end_sample / self.ref_audio_sr - start_offset_sec

            labels.append(seg.word)

            # VAD trim 범위 밖에 완전히 놓인 segment는 표시하지 않는다.
            if end_sec <= 0.0 or start_sec >= total_duration_sec:
                continue

            start_frame = max(
                0,
                int(math.floor(start_sec / sec_per_frame)),
            )
            end_frame = min(
                t_mel,
                int(math.ceil(end_sec / sec_per_frame)),
            )

            if end_frame <= start_frame:
                end_frame = min(t_mel, start_frame + 1)

            mat[row_idx, start_frame:end_frame] = 1.0

        return mat, labels

    def _build_collapsed_word_matrix(
        self,
        raw_align: torch.Tensor,
        hyp_symbols: list[str],
        matched_words: MatchedWords,
    ) -> tuple[torch.Tensor, list[str]]:
        t_text, t_mel = raw_align.shape
        collapsed_labels: list[str] = []
        token_to_collapsed_row: dict[int, int] = {}
        match_by_hyp_start: dict[int, tuple[slice, slice]] = {}
        matched_token_indices: set[int] = set()

        for ref_slice, hyp_slice in zip(
            matched_words.ref_matched_indices,
            matched_words.hyp_matched_indices,
            strict=True,
        ):
            if ref_slice.start is None or ref_slice.stop is None:
                raise ValueError(f"Invalid ref slice: {ref_slice}")
            if hyp_slice.start is None or hyp_slice.stop is None:
                raise ValueError(f"Invalid hyp slice: {hyp_slice}")

            match_by_hyp_start[hyp_slice.start] = (ref_slice, hyp_slice)
            matched_token_indices.update(range(hyp_slice.start, hyp_slice.stop))

        token_idx = 0
        while token_idx < t_text:
            if token_idx in match_by_hyp_start:
                ref_slice, hyp_slice = match_by_hyp_start[token_idx]
                label = " ".join(matched_words.ref_seqs[ref_slice.start : ref_slice.stop])

                row_idx = len(collapsed_labels)
                collapsed_labels.append(label)

                for idx in range(hyp_slice.start, hyp_slice.stop):
                    if 0 <= idx < t_text:
                        token_to_collapsed_row[idx] = row_idx

                token_idx = hyp_slice.stop
                continue

            if token_idx in matched_token_indices:
                token_idx += 1
                continue

            row_idx = len(collapsed_labels)
            collapsed_labels.append(self._pretty_token_label(hyp_symbols[token_idx]))
            token_to_collapsed_row[token_idx] = row_idx
            token_idx += 1

        collapsed_mat = torch.zeros(len(collapsed_labels), t_mel, dtype=torch.float32)
        for token_idx in range(t_text):
            row_idx = token_to_collapsed_row.get(token_idx)
            if row_idx is None:
                continue
            collapsed_mat[row_idx] = torch.maximum(collapsed_mat[row_idx], raw_align[token_idx])

        return collapsed_mat, collapsed_labels

    @staticmethod
    def _plot_alignment_matrix(
        ax,
        mat: torch.Tensor,
        y_labels: list[str],
        title: str,
    ) -> None:
        ax.imshow(
            mat.cpu().numpy(),
            aspect="auto",
            interpolation="nearest",
            origin="upper",
        )
        ax.set_title(title)
        ax.set_xlabel("Mel frame")
        ax.set_ylabel("Text")

        if not y_labels:
            ax.set_yticks([])
            return

        max_ticks = 80
        step = max(1, math.ceil(len(y_labels) / max_ticks))
        ticks = list(range(0, len(y_labels), step))
        ax.set_yticks(ticks)
        ax.set_yticklabels([y_labels[i] for i in ticks], fontsize=6)

    @staticmethod
    def _pretty_token_label(token: str) -> str:
        if token == " ":
            return "<sp>"
        if token == "^":
            return "<bos>"
        if token == "~":
            return "<eos>"
        if token == "\n":
            return "<nl>"
        if token == "\t":
            return "<tab>"
        return token

    @staticmethod
    def _make_spec_mask(
        y_lengths: torch.Tensor,
        t_spec: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        idx = torch.arange(t_spec, device=device).unsqueeze(0)
        return (idx < y_lengths.unsqueeze(1)).to(dtype=dtype)


@dataclass(frozen=True)
class TIMITAnalysisRow:
    sample_id: str
    wav_path: str
    txt_path: str
    wrd_path: str
    text: str

    num_boundaries: int

    mean_wbe_ms: float
    median_wbe_ms: float
    p90_wbe_ms: float
    p95_wbe_ms: float
    max_wbe_ms: float

    p_10ms: float
    p_25ms: float
    p_50ms: float
    p_100ms: float

    posterior_entropy: float | None
    mcd_dtw: float | None


class TIMITErrorAnalyzer(TIMITBenchMarker):
    """
    Error-analysis extension of TIMITBenchMarker.

    Outputs:
        analysis_dir/
            sample_metrics.csv
            summary.json
            figures/
                sample_wbe_histogram.png
                boundary_error_histogram.png
                sample_wbe_ecdf.png
                worst_samples.png
                wbe_vs_entropy.png
            alignments/
                <worst-sample alignment figures>
    """

    def __init__(
        self,
        root_dir: str | Path,
        ref_audio_sr: int,
        hyp_audio_sr: int,
        hyp_hop_length: int,
        input_maker: AlignerInputMaker,
        analysis_dir: str | Path,
        hyp_ignore_symbols: set[str] | None = None,
        max_ref_words_per_hyp_word: int = 5,
        seed: int = 42,
    ):
        super().__init__(
            root_dir=root_dir,
            ref_audio_sr=ref_audio_sr,
            hyp_audio_sr=hyp_audio_sr,
            hyp_hop_length=hyp_hop_length,
            input_maker=input_maker,
            hyp_ignore_symbols=hyp_ignore_symbols,
            max_ref_words_per_hyp_word=max_ref_words_per_hyp_word,
            seed=seed,
        )

        self.analysis_dir = Path(analysis_dir)
        self.figure_dir = self.analysis_dir / "figures"
        self.alignment_dir = self.analysis_dir / "alignments"

        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.alignment_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def analyze(
        self,
        aligner: NDAligner,
        vocoder: Generator | None = None,
        max_test_samples: int | None = None,
        top_k_alignments: int = 30,
        compute_entropy: bool = True,
        compute_mcd_dtw: bool = False,
    ) -> list[TIMITAnalysisRow]:
        """
        Run sample-level error analysis.

        Args:
            aligner:
                Trained ND-Aligner.

            vocoder:
                Required only when compute_mcd_dtw=True.

            max_test_samples:
                Optional limit on the number of TIMIT samples.

            top_k_alignments:
                Number of worst-WBE samples for which alignment figures
                are generated.

            compute_entropy:
                Compute posterior entropy. This requires soft alignment.

            compute_mcd_dtw:
                Compute reconstruction MCD-DTW. This is relatively expensive
                and requires vocoder.

        Returns:
            Rows sorted by descending sample-level mean WBE.
        """
        if compute_mcd_dtw and vocoder is None:
            raise ValueError("compute_mcd_dtw=True requires a vocoder.")

        was_aligner_training = aligner.training
        was_input_maker_training = self.input_maker.training
        was_vocoder_training = vocoder.training if vocoder is not None else None

        device = next(aligner.parameters()).device
        self.input_maker.to(device=device)

        triplets = self.triplets if max_test_samples is None else self.triplets[:max_test_samples]

        rows: list[TIMITAnalysisRow] = []
        all_boundary_errors_ms: list[float] = []

        try:
            aligner.eval()
            self.input_maker.eval()

            if vocoder is not None:
                vocoder.eval()

            run_test_metrics = compute_entropy or compute_mcd_dtw
            analysis_vocoder = vocoder if compute_mcd_dtw else None

            for wrd_path, wav_path, txt_path in tqdm(
                triplets,
                desc="Analyzing TIMIT errors",
                total=len(triplets),
            ):
                result = super().run_one(
                    txt_path=txt_path,
                    wrd_path=wrd_path,
                    wav_path=wav_path,
                    aligner=aligner,
                    vocoder=analysis_vocoder,
                    save_align_figure=False,
                    align_figure_dir=None,
                    is_test=run_test_metrics,
                )

                boundary_errors_ms = [
                    float(error_sec) * 1000.0 for error_sec in result.boundary_errors
                ]

                if not boundary_errors_ms:
                    continue

                error_tensor = torch.tensor(
                    boundary_errors_ms,
                    dtype=torch.float32,
                )

                all_boundary_errors_ms.extend(boundary_errors_ms)

                row = TIMITAnalysisRow(
                    sample_id=self._make_sample_id(wav_path),
                    wav_path=str(wav_path),
                    txt_path=str(txt_path),
                    wrd_path=str(wrd_path),
                    text=self._read_txt(txt_path),
                    num_boundaries=len(boundary_errors_ms),
                    mean_wbe_ms=error_tensor.mean().item(),
                    median_wbe_ms=error_tensor.median().item(),
                    p90_wbe_ms=torch.quantile(
                        error_tensor,
                        0.90,
                    ).item(),
                    p95_wbe_ms=torch.quantile(
                        error_tensor,
                        0.95,
                    ).item(),
                    max_wbe_ms=error_tensor.max().item(),
                    p_10ms=((error_tensor <= 10.0).float().mean().item() * 100.0),
                    p_25ms=((error_tensor <= 25.0).float().mean().item() * 100.0),
                    p_50ms=((error_tensor <= 50.0).float().mean().item() * 100.0),
                    p_100ms=((error_tensor <= 100.0).float().mean().item() * 100.0),
                    posterior_entropy=result.posterior_entropy,
                    mcd_dtw=result.mcd_dtw,
                )
                rows.append(row)

            rows.sort(
                key=lambda row: row.mean_wbe_ms,
                reverse=True,
            )

            self._save_sample_csv(rows)
            self._save_analysis_summary(
                rows=rows,
                all_boundary_errors_ms=all_boundary_errors_ms,
            )
            self._save_analysis_figures(
                rows=rows,
                all_boundary_errors_ms=all_boundary_errors_ms,
            )

            self._save_worst_alignment_figures(
                rows=rows,
                aligner=aligner,
                top_k=top_k_alignments,
                compute_soft_path=compute_entropy,
            )

            return rows

        finally:
            if was_aligner_training:
                aligner.train()

            if was_input_maker_training:
                self.input_maker.train()

            if vocoder is not None and was_vocoder_training:
                vocoder.train()

    def _make_sample_id(self, wav_path: Path) -> str:
        try:
            relative_path = wav_path.relative_to(self.root_dir).with_suffix("")
            return relative_path.as_posix().replace("/", "__")
        except ValueError:
            return wav_path.stem

    def _save_sample_csv(
        self,
        rows: list[TIMITAnalysisRow],
    ) -> None:
        save_path = self.analysis_dir / "sample_metrics.csv"

        if not rows:
            save_path.write_text("", encoding="utf-8")
            return

        fieldnames = list(asdict(rows[0]).keys())

        with open(
            save_path,
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writeheader()

            for row in rows:
                writer.writerow(asdict(row))

    def _save_analysis_summary(
        self,
        rows: list[TIMITAnalysisRow],
        all_boundary_errors_ms: list[float],
    ) -> None:
        save_path = self.analysis_dir / "summary.json"

        if not rows or not all_boundary_errors_ms:
            summary = {
                "num_samples": 0,
                "num_boundaries": 0,
            }
        else:
            sample_wbe = torch.tensor(
                [row.mean_wbe_ms for row in rows],
                dtype=torch.float32,
            )
            boundary_wbe = torch.tensor(
                all_boundary_errors_ms,
                dtype=torch.float32,
            )

            summary = {
                "num_samples": len(rows),
                "num_boundaries": len(all_boundary_errors_ms),
                "sample_mean_wbe_ms": sample_wbe.mean().item(),
                "sample_median_wbe_ms": sample_wbe.median().item(),
                "sample_p90_wbe_ms": torch.quantile(
                    sample_wbe,
                    0.90,
                ).item(),
                "sample_p95_wbe_ms": torch.quantile(
                    sample_wbe,
                    0.95,
                ).item(),
                "sample_max_wbe_ms": sample_wbe.max().item(),
                "boundary_mean_wbe_ms": boundary_wbe.mean().item(),
                "boundary_median_wbe_ms": boundary_wbe.median().item(),
                "boundary_p90_wbe_ms": torch.quantile(
                    boundary_wbe,
                    0.90,
                ).item(),
                "boundary_p95_wbe_ms": torch.quantile(
                    boundary_wbe,
                    0.95,
                ).item(),
                "boundary_max_wbe_ms": boundary_wbe.max().item(),
                "p_10ms": ((boundary_wbe <= 10.0).float().mean().item() * 100.0),
                "p_25ms": ((boundary_wbe <= 25.0).float().mean().item() * 100.0),
                "p_50ms": ((boundary_wbe <= 50.0).float().mean().item() * 100.0),
                "p_100ms": ((boundary_wbe <= 100.0).float().mean().item() * 100.0),
            }

        with open(
            save_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def _save_analysis_figures(
        self,
        rows: list[TIMITAnalysisRow],
        all_boundary_errors_ms: list[float],
    ) -> None:
        if not rows:
            return

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        sample_wbe = torch.tensor(
            [row.mean_wbe_ms for row in rows],
            dtype=torch.float32,
        )

        # --------------------------------------------------------------
        # Sample-level WBE histogram
        # --------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(
            sample_wbe.numpy(),
            bins=40,
        )
        ax.axvline(
            sample_wbe.mean().item(),
            linestyle="--",
            label=f"Mean: {sample_wbe.mean().item():.2f} ms",
        )
        ax.axvline(
            sample_wbe.median().item(),
            linestyle=":",
            label=f"Median: {sample_wbe.median().item():.2f} ms",
        )
        ax.set_title("Sample-level mean WBE distribution")
        ax.set_xlabel("Mean WBE per sample (ms)")
        ax.set_ylabel("Number of samples")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            self.figure_dir / "sample_wbe_histogram.png",
            dpi=200,
        )
        plt.close(fig)

        # --------------------------------------------------------------
        # Boundary-level error histogram
        # --------------------------------------------------------------
        if all_boundary_errors_ms:
            boundary_errors = torch.tensor(
                all_boundary_errors_ms,
                dtype=torch.float32,
            )

            # Extremely large outliers can make the histogram unreadable.
            x_max = torch.quantile(
                boundary_errors,
                0.99,
            ).item()

            clipped = boundary_errors[boundary_errors <= x_max]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(
                clipped.numpy(),
                bins=60,
            )
            ax.axvline(10.0, linestyle="--", label="10 ms")
            ax.axvline(25.0, linestyle="--", label="25 ms")
            ax.axvline(50.0, linestyle="--", label="50 ms")
            ax.axvline(100.0, linestyle="--", label="100 ms")
            ax.set_title("Boundary-error distribution " + "(up to the 99th percentile)")
            ax.set_xlabel("Absolute boundary error (ms)")
            ax.set_ylabel("Number of boundaries")
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                self.figure_dir / "boundary_error_histogram.png",
                dpi=200,
            )
            plt.close(fig)

        # --------------------------------------------------------------
        # Sample-level ECDF
        # --------------------------------------------------------------
        sorted_wbe = torch.sort(sample_wbe).values
        cumulative = torch.arange(
            1,
            len(sorted_wbe) + 1,
            dtype=torch.float32,
        ) / len(sorted_wbe)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            sorted_wbe.numpy(),
            cumulative.numpy() * 100.0,
        )
        ax.axvline(25.0, linestyle="--", label="25 ms")
        ax.axvline(50.0, linestyle="--", label="50 ms")
        ax.set_title("Sample-level WBE empirical CDF")
        ax.set_xlabel("Mean WBE per sample (ms)")
        ax.set_ylabel("Samples at or below threshold (%)")
        ax.set_ylim(0.0, 100.0)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            self.figure_dir / "sample_wbe_ecdf.png",
            dpi=200,
        )
        plt.close(fig)

        # --------------------------------------------------------------
        # Worst samples
        # --------------------------------------------------------------
        worst_rows = rows[: min(30, len(rows))]
        labels = [row.sample_id for row in reversed(worst_rows)]
        values = [row.mean_wbe_ms for row in reversed(worst_rows)]

        fig_height = max(7.0, 0.35 * len(worst_rows))
        fig, ax = plt.subplots(
            figsize=(12, fig_height),
        )
        ax.barh(labels, values)
        ax.set_title("Worst samples by mean WBE")
        ax.set_xlabel("Mean WBE (ms)")
        ax.set_ylabel("Sample")
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        fig.savefig(
            self.figure_dir / "worst_samples.png",
            dpi=200,
        )
        plt.close(fig)

        # --------------------------------------------------------------
        # WBE versus posterior entropy
        # --------------------------------------------------------------
        entropy_rows = [
            row
            for row in rows
            if row.posterior_entropy is not None and math.isfinite(row.posterior_entropy)
        ]

        if entropy_rows:
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.scatter(
                [row.posterior_entropy for row in entropy_rows],  # type: ignore
                [row.mean_wbe_ms for row in entropy_rows],
                alpha=0.65,
            )
            ax.set_title("Posterior entropy versus sample WBE")
            ax.set_xlabel("Posterior entropy")
            ax.set_ylabel("Mean WBE (ms)")
            fig.tight_layout()
            fig.savefig(
                self.figure_dir / "wbe_vs_entropy.png",
                dpi=200,
            )
            plt.close(fig)

    @torch.no_grad()
    def _save_worst_alignment_figures(
        self,
        rows: list[TIMITAnalysisRow],
        aligner: NDAligner,
        top_k: int,
        compute_soft_path: bool,
    ) -> None:
        if top_k <= 0:
            return

        worst_rows = rows[: min(top_k, len(rows))]

        for rank, row in enumerate(
            tqdm(
                worst_rows,
                desc="Saving worst alignments",
            ),
            start=1,
        ):
            rank_dir = self.alignment_dir / f"rank_{rank:03d}_wbe_{row.mean_wbe_ms:.2f}ms"

            super().run_one(
                txt_path=Path(row.txt_path),
                wrd_path=Path(row.wrd_path),
                wav_path=Path(row.wav_path),
                aligner=aligner,
                vocoder=None,
                save_align_figure=True,
                align_figure_dir=rank_dir,
                is_test=compute_soft_path,
            )
