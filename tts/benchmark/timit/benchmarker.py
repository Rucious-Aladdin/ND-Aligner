from __future__ import annotations

import math
import random
from pathlib import Path
from typing import NamedTuple

import torch
import torchaudio.functional as AF
from tqdm import tqdm

from tts.models.modules.hifigan_vocoder import Generator
from tts.models.ndaligner import AlignerFeatures, NDAligner
from tts.models.utils.input_maker import AlignerInputMaker

from ..utils.entropy import compute_framewise_entropy
from ..utils.mcd_dtw import compute_mcd_dtw
from .word_mapper import MatchedWords, TimitWordSegment, WordsMapper


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

        self.word_mapper = WordsMapper(
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
            wav_paths=wav_path,
            scripts=[text],
        )

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
            )

        boundary_errors = self._get_eval_instance(
            wrd_segments=wrd_segments,
            matched_words=matched,
            hard_dur=hard_dur,
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
        wrd_segments: list[TimitWordSegment],
        matched_words: MatchedWords,
        hard_dur: list[int],
    ) -> list[float]:
        sec_per_frame = self.hyp_hop_length / self.hyp_audio_sr

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

            gt_start = wrd_segments[ref_slice.start].start_sample / self.ref_audio_sr
            gt_end = wrd_segments[ref_slice.stop - 1].end_sample / self.ref_audio_sr

            pred_start = token_starts[hyp_slice.start]
            pred_end = token_ends[hyp_slice.stop - 1]

            errors.append(abs(pred_start - gt_start))
            errors.append(abs(pred_end - gt_end))

        return errors

    @staticmethod
    def _read_wrd(wrd_path: Path) -> list[TimitWordSegment]:
        segments: list[TimitWordSegment] = []
        with open(wrd_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=2)
                segments.append(
                    TimitWordSegment(
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
        wrd_segments: list[TimitWordSegment],
        matched_words: MatchedWords,
        hyp_symbols: list[str],
        hard_dur: list[int],
        features: AlignerFeatures,
        t_mel: int,
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
        wrd_segments: list[TimitWordSegment],
        t_mel: int,
    ) -> tuple[torch.Tensor, list[str]]:
        sec_per_frame = self.hyp_hop_length / self.hyp_audio_sr
        mat = torch.zeros(len(wrd_segments), t_mel, dtype=torch.float32)
        labels: list[str] = []

        for row_idx, seg in enumerate(wrd_segments):
            start_sec = seg.start_sample / self.ref_audio_sr
            end_sec = seg.end_sample / self.ref_audio_sr

            start_frame = max(0, int(math.floor(start_sec / sec_per_frame)))
            end_frame = min(t_mel, int(math.ceil(end_sec / sec_per_frame)))

            if start_frame >= t_mel:
                labels.append(seg.word)
                continue
            if end_frame <= start_frame:
                end_frame = min(t_mel, start_frame + 1)

            mat[row_idx, start_frame:end_frame] = 1.0
            labels.append(seg.word)

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
