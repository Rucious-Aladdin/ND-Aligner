from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import TypedDict
from collections.abc import Sequence

import torch

from tts.logger.metrics.mcd_dtw import compute_mcd_dtw
from tts.logger.metrics.entropy import compute_framewise_entropy
from tts.logger.metrics.path_agreement import compute_path_agreement_score
from tts.logger.metrics.duration_mae import compute_duration_mae


class Stage1EvalTensorDumpInstance(TypedDict):
    # Ground truth from dataset
    gt_wav: torch.Tensor  # (T_gt_wav,)
    gt_mel: torch.Tensor  # (n_mels, T_gt_mel)

    # Reconstructed from train-time alignment
    recon_wav: torch.Tensor  # (T_recon_wav,)
    recon_mel: torch.Tensor  # (n_mels, T_recon_mel)

    # Generated from inference path
    # duration predictor + shallow decoder
    gen_wav: torch.Tensor  # (T_gen_wav,)
    gen_mel: torch.Tensor  # (n_mels, T_gen_mel)

    # Alignment
    viterbi_alignment: torch.Tensor  # (T_mel, T_text)
    posterior_attention: torch.Tensor  # (T_mel, T_text)

    # Duration-related tensors
    viterbi_duration: torch.Tensor  # (T_text,)
    posterior_duration: torch.Tensor  # (T_text,)
    pred_duration: torch.Tensor  # (T_text,)


class Stage1EvalCSVRowInstance(TypedDict):
    # Identity / metadata
    utt_id: str
    dataset: str
    split: str
    speaker_id: str
    text: str
    gt_wav_path: str

    # Experiment identity
    variant: str  # full, wo_local_support, wo_progress, ...
    epoch: int
    step: int

    # Tensor dump path
    pt_file_path: str

    # Length metadata
    text_len: int
    gt_wav_len: int
    recon_wav_len: int
    gen_wav_len: int
    gt_mel_len: int
    recon_mel_len: int
    gen_mel_len: int

    # MCD-DTW metrics
    mcd_dtw_recon: float  # MCD-DTW(GT, Recon)
    mcd_dtw_gen: float  # MCD-DTW(GT, Gen)

    # Pred-Rec Gap used in ablation table
    mcd_dtw_pred_rec_gap: float  # mcd_dtw_gen - mcd_dtw_recon

    # Optional: direct recon/gen mismatch
    mcd_dtw_recon_gen: float  # MCD-DTW(Recon, Gen)

    # Alignment-aware metrics
    viterbi_posterior_agree_score: float
    framewise_entropy: float

    # Duration predictor-aware metrics
    dur_viterbi_mae: float
    dur_posterior_mae: float


class Stage1TrainTimeEvalLogger:
    def __init__(
        self,
        base_dir: str,
        experiment_name: str,
        exp_variant: str,
    ) -> None:
        self.dir = Path(base_dir) / experiment_name
        self.dir.mkdir(parents=True, exist_ok=True)

        self.pt_dir = self.dir / "tensor_dumps"
        self.pt_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.dir / "eval_metrics.csv"
        self._init_csv()

        self.exp_variant = exp_variant

    def __call__(
        self,
        *,
        epoch: int,
        step: int,
        # metadata
        datasets: list[str] | tuple[str],
        utt_ids: list[str] | tuple[str],
        split: str,
        speaker_ids: list[str] | tuple[str],
        texts: list[str] | tuple[str],
        gt_wav_paths: list[str] | tuple[str],
        # wav tensors
        gt_wav: torch.Tensor,  # (B, T_gt_wav)
        gt_wav_mask: torch.Tensor,  # (B, T_gt_wav)
        recon_wav: torch.Tensor,  # (B, T_recon_wav)
        recon_wav_mask: torch.Tensor,  # (B, T_recon_wav)
        gen_wav: torch.Tensor,  # (B, T_gen_wav)
        gen_wav_mask: torch.Tensor,  # (B, T_gen_wav)
        # mel tensors
        gt_mel: torch.Tensor,  # (B, n_mels, T_gt_mel)
        gt_mel_mask: torch.Tensor,  # (B, T_gt_mel)
        recon_mel: torch.Tensor,  # (B, n_mels, T_recon_mel)
        recon_mel_mask: torch.Tensor,  # (B, T_recon_mel)
        gen_mel: torch.Tensor,  # (B, n_mels, T_gen_mel)
        gen_mel_mask: torch.Tensor,  # (B, T_gen_mel)
        # alignment tensors
        viterbi_alignment: torch.Tensor,  # (B, T_mel, T_text)
        posterior_attention: torch.Tensor,  # (B, T_mel, T_text)
        spec_mask: torch.Tensor,  # (B, T_mel)
        text_mask: torch.Tensor,  # (B, T_text)
        # duration tensors
        viterbi_duration: torch.Tensor,  # (B, T_text)
        posterior_duration: torch.Tensor,  # (B, T_text)
        pred_duration: torch.Tensor,  # (B, T_text)
    ) -> list[Stage1EvalCSVRowInstance]:
        B = gt_wav.size(0)

        self._validate_batch_size(
            B=B,
            datasets=datasets,
            utt_ids=utt_ids,
            speaker_ids=speaker_ids,
            texts=texts,
            gt_wav_paths=gt_wav_paths,
        )

        # Metrics: (B,)
        mcd_dtw_recon = compute_mcd_dtw(
            ref_wavs=gt_wav,
            ref_wav_mask=gt_wav_mask,
            hyp_wavs=recon_wav,
            hyp_wav_mask=recon_wav_mask,
            exclude_c0=True,
        )

        mcd_dtw_gen = compute_mcd_dtw(
            ref_wavs=gt_wav,
            ref_wav_mask=gt_wav_mask,
            hyp_wavs=gen_wav,
            hyp_wav_mask=gen_wav_mask,
            exclude_c0=True,
        )

        mcd_dtw_recon_gen = compute_mcd_dtw(
            ref_wavs=recon_wav,
            ref_wav_mask=recon_wav_mask,
            hyp_wavs=gen_wav,
            hyp_wav_mask=gen_wav_mask,
            exclude_c0=True,
        )

        viterbi_posterior_agree = compute_path_agreement_score(
            attn_posterior=posterior_attention,
            attn_viterbi=viterbi_alignment,
            mask=spec_mask,
        )

        framewise_entropy = compute_framewise_entropy(
            attn=posterior_attention,
            mask=spec_mask,
        )

        dur_viterbi_mae = compute_duration_mae(
            pred_duration=pred_duration,
            target_duration=viterbi_duration,
            text_mask=text_mask,
        )

        dur_posterior_mae = compute_duration_mae(
            pred_duration=pred_duration,
            target_duration=posterior_duration,
            text_mask=text_mask,
        )

        rows: list[Stage1EvalCSVRowInstance] = []

        for b in range(B):
            text_len = self._mask_len(text_mask[b])
            gt_wav_len = self._mask_len(gt_wav_mask[b])
            recon_wav_len = self._mask_len(recon_wav_mask[b])
            gen_wav_len = self._mask_len(gen_wav_mask[b])

            gt_mel_len = self._mask_len(gt_mel_mask[b])
            recon_mel_len = self._mask_len(recon_mel_mask[b])
            gen_mel_len = self._mask_len(gen_mel_mask[b])

            pt_file_path = self._save_tensor_dump(
                utt_id=utt_ids[b],
                epoch=epoch,
                step=step,
                gt_wav=gt_wav[b, :gt_wav_len],
                gt_mel=gt_mel[b, :, :gt_mel_len],
                recon_wav=recon_wav[b, :recon_wav_len],
                recon_mel=recon_mel[b, :, :recon_mel_len],
                gen_wav=gen_wav[b, :gen_wav_len],
                gen_mel=gen_mel[b, :, :gen_mel_len],
                viterbi_alignment=viterbi_alignment[b, :gt_mel_len, :text_len],
                posterior_attention=posterior_attention[b, :gt_mel_len, :text_len],
                viterbi_duration=viterbi_duration[b, :text_len],
                posterior_duration=posterior_duration[b, :text_len],
                pred_duration=pred_duration[b, :text_len],
            )

            row: Stage1EvalCSVRowInstance = {
                "utt_id": utt_ids[b],
                "dataset": datasets[b],
                "split": split,
                "speaker_id": speaker_ids[b],
                "text": texts[b],
                "gt_wav_path": gt_wav_paths[b],
                "variant": self.exp_variant,
                "epoch": epoch,
                "step": step,
                "pt_file_path": str(pt_file_path),
                "text_len": text_len,
                "gt_wav_len": gt_wav_len,
                "recon_wav_len": recon_wav_len,
                "gen_wav_len": gen_wav_len,
                "gt_mel_len": gt_mel_len,
                "recon_mel_len": recon_mel_len,
                "gen_mel_len": gen_mel_len,
                "mcd_dtw_recon": float(mcd_dtw_recon[b].detach().cpu().item()),
                "mcd_dtw_gen": float(mcd_dtw_gen[b].detach().cpu().item()),
                "mcd_dtw_pred_rec_gap": float(
                    (mcd_dtw_gen[b] - mcd_dtw_recon[b]).detach().cpu().item()
                ),
                "mcd_dtw_recon_gen": float(mcd_dtw_recon_gen[b].detach().cpu().item()),
                "viterbi_posterior_agree_score": float(
                    viterbi_posterior_agree[b].detach().cpu().item()
                ),
                "framewise_entropy": float(framewise_entropy[b].detach().cpu().item()),
                "dur_viterbi_mae": float(dur_viterbi_mae[b].detach().cpu().item()),
                "dur_posterior_mae": float(dur_posterior_mae[b].detach().cpu().item()),
            }

            self._append_csv_row(row)
            rows.append(row)

        return rows

    def _init_csv(self) -> None:
        if self.csv_path.exists():
            return

        fieldnames = list(Stage1EvalCSVRowInstance.__annotations__.keys())

        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def _save_tensor_dump(
        self,
        *,
        utt_id: str,
        epoch: int,
        step: int,
        gt_wav: torch.Tensor,
        gt_mel: torch.Tensor,
        recon_wav: torch.Tensor,
        recon_mel: torch.Tensor,
        gen_wav: torch.Tensor,
        gen_mel: torch.Tensor,
        viterbi_alignment: torch.Tensor,
        posterior_attention: torch.Tensor,
        viterbi_duration: torch.Tensor,
        posterior_duration: torch.Tensor,
        pred_duration: torch.Tensor,
    ) -> Path:
        safe_utt_id = self._safe_name(utt_id)
        safe_variant = self._safe_name(self.exp_variant)

        pt_path = self.pt_dir / f"{safe_variant}_{safe_utt_id}_ep{epoch:04d}_st{step:08d}.pt"

        instance: Stage1EvalTensorDumpInstance = {
            "gt_wav": self._to_cpu(gt_wav),
            "gt_mel": self._to_cpu(gt_mel),
            "recon_wav": self._to_cpu(recon_wav),
            "recon_mel": self._to_cpu(recon_mel),
            "gen_wav": self._to_cpu(gen_wav),
            "gen_mel": self._to_cpu(gen_mel),
            "viterbi_alignment": self._to_cpu(viterbi_alignment),
            "posterior_attention": self._to_cpu(posterior_attention),
            "viterbi_duration": self._to_cpu(viterbi_duration),
            "posterior_duration": self._to_cpu(posterior_duration),
            "pred_duration": self._to_cpu(pred_duration),
        }

        torch.save(instance, pt_path)
        return pt_path

    def _append_csv_row(self, row: Stage1EvalCSVRowInstance) -> None:
        fieldnames = list(Stage1EvalCSVRowInstance.__annotations__.keys())

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    def _mask_len(self, mask: torch.Tensor) -> int:
        if mask.dim() != 1:
            mask = mask.view(-1)

        return int(mask.to(dtype=torch.long).sum().item())

    def _to_cpu(self, x: torch.Tensor) -> torch.Tensor:
        return x.detach().cpu().contiguous()

    def _safe_name(self, name: str) -> str:
        name = str(name)
        name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
        return name.strip("_")

    def _validate_batch_size(
        self,
        *,
        B: int,
        datasets: Sequence[str],
        utt_ids: Sequence[str],
        speaker_ids: Sequence[str],
        texts: Sequence[str],
        gt_wav_paths: Sequence[str],
    ) -> None:
        items = {
            "datasets": datasets,
            "utt_ids": utt_ids,
            "speaker_ids": speaker_ids,
            "texts": texts,
            "gt_wav_paths": gt_wav_paths,
        }

        for name, value in items.items():
            if len(value) != B:
                raise ValueError(f"{name} must have length {B}, got {len(value)}.")
