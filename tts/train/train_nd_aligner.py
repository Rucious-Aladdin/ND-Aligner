import argparse
import math
import os
from typing import Any, NamedTuple, cast, override

import torch
from torch.utils.data import DataLoader

from tts.benchmark.timit.benchmarker import TIMITBenchMarker
from tts.benchmark.timit.word_mapper import HYP_IGNORE_SYMBOLS
from tts.config.ndaligner.data_config import DataConfig, LossConfigs, TrainConfigs
from tts.config.ndaligner.training_module_config import NDAlignerTrainingModuleConfigs
from tts.config.utils.io import load_config, save_config
from tts.data.data_types import LossValues, TTSBatch
from tts.data.tts_datafactory import TTSDataFactory
from tts.logger.timit_logger import NDAlignerTimitLogger
from tts.logger.utils.plot_alignment import plot_alignment
from tts.logger.utils.plot_spectrogram import plot_spectrogram
from tts.models.init_ndaligner import init_nd_aligner_training_module
from tts.models.ndaligner import (
    AlignerForward,
    NDAlignerLossWeights,
    NDAlignerTrainingModule,
    NDAlignerTrainingModuleForward,
)
from tts.tokenizer.text_tokenizer import TextTokenizer
from tts.utils.anneal import get_linear_anneal_weight
from tts.utils.checkpoint_manager import CheckpointManager

from .base_trainer import BaseTrainer

TEXT_TOKENIZER = TextTokenizer()


class TimitCheckpointMetric(NamedTuple):
    timit_bae: float


# torch.autograd.set_detect_anomaly(True)


def prepare_masked_score_for_plot(
    x: torch.Tensor,
    *,
    neg_threshold: float = -1e8,
    fill_mode: str = "min",  # "mean", "min", "p05"
) -> torch.Tensor:
    """
    x: masked score map, e.g. (T_s, T_t), containing -1e9 invalid entries.

    Returns:
        plot_x: invalid entries replaced for visualization only.
    """
    x = x.detach().float().cpu()

    valid = torch.isfinite(x) & (x > neg_threshold)

    if not valid.any():
        return torch.zeros_like(x)

    valid_values = x[valid]

    if fill_mode == "mean":
        fill_value = valid_values.mean()
    elif fill_mode == "min":
        fill_value = valid_values.min()
    elif fill_mode == "p05":
        fill_value = torch.quantile(valid_values, 0.05)
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")

    plot_x = x.clone()
    plot_x[~valid] = fill_value

    return plot_x


class NDAlignerTrainer(
    BaseTrainer[
        DataConfig,
        NDAlignerTrainingModuleConfigs,
    ]
):
    @override
    def setup_model(
        self,
    ) -> tuple[
        torch.nn.Module,
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LRScheduler | None,
    ]:
        self.data_config: DataConfig

        print("🏗️ Initializing model...")
        model = init_nd_aligner_training_module(
            config=self.model_config,
            load_speaker_encoder=self.data_config.extra_exp.train_time_eval_logging,
            load_vocoder=self.data_config.extra_exp.train_time_eval_logging,
            device=str(self.device),
        )
        model.print_parameter_summary()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.train_cfg.lr,
            betas=cast(tuple[float, float], self.train_cfg.betas),
            eps=self.train_cfg.eps,
            weight_decay=self.train_cfg.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.train_cfg.lr_decay_rate,
        )
        return model, optimizer, scheduler

    @override
    def on_fit_start(self) -> None:
        self.data_config: DataConfig
        self.model_config: NDAlignerTrainingModuleConfigs

        self.train_time_eval_logger = None
        self.timit_ckpt_manager = None
        if self.data_config.extra_exp.train_time_eval_logging:
            from tts.preprocess.config import PreprocessConfig

            print("🧪 Train-time TIMIT eval is on.")
            timit_benchmarker = TIMITBenchMarker(
                root_dir=self.data_config.extra_exp.timit_root_dir,
                ref_audio_sr=self.data_config.extra_exp.timit_sr,
                hyp_audio_sr=self.data_config.audio.sr,
                hyp_hop_length=self.data_config.audio.hop_length,
                tokenizer=TEXT_TOKENIZER,
                audio_config=self.data_config.audio,
                hyp_ignore_symbols=HYP_IGNORE_SYMBOLS,
                max_ref_words_per_hyp_word=5,
                spk_cond_dim=self.model_config.nd_aligner.spec_enc.cond_dim,
                rms_normalize=PreprocessConfig.rms_normalize,
                rms_target=PreprocessConfig.rms_target,
            )

            self.train_time_eval_logger = NDAlignerTimitLogger(
                base_dir=self.data_config.extra_exp.base_dir,
                experiment_name=self.data_config.extra_exp.exp_name,
                exp_variant=self.data_config.extra_exp.exp_variant,
                timit_benchmarker=timit_benchmarker,
                max_test_samples=self.data_config.extra_exp.timit_max_num_test_samples,
            )

            self.timit_ckpt_manager = CheckpointManager(
                checkpoint_dir=os.path.join(self.run_dir, "checkpoints_timit_bae"),
                keep_best_epoch_count=self.train_cfg.keep_best_count,
                keep_best_step_count=self.train_cfg.keep_best_count,
                keep_last_count=0,
                monitor_loss="timit_bae",
            )

            exp_name = self.data_config.extra_exp.exp_name
            marker_path = os.path.join(self.run_dir, f"{exp_name}.txt")
            with open(marker_path, "a", encoding="utf-8"):
                pass

        else:
            print("💤 Train-time TIMIT eval is off.")

        save_config(self.model_config.nd_aligner, os.path.join(self.run_dir, "aligner_config.json"))
        print(f"📄 Aligner-Only Configurations backed up!")

    @override
    def setup_dataloader(self) -> tuple[DataLoader[Any], DataLoader[Any]]:
        print("📦 Initializing datasets...")
        data_factory = TTSDataFactory(self.data_config)
        return data_factory.train_loader, data_factory.valid_loader

    def _get_loss_weights(self, step: int) -> NDAlignerLossWeights:
        """Calculates current weighted objective coefficients for NDAlignerTrainingModule."""
        cfg: LossConfigs = self.data_config.loss

        return NDAlignerLossWeights(
            crf_loss_weight=get_linear_anneal_weight(
                step,
                cfg.crf_start_step,
                cfg.crf_end_step,
                cfg.crf_init_weight,
                cfg.crf_final_weight,
            ),
            diag_loss_weight=get_linear_anneal_weight(
                step,
                cfg.diag_start_step,
                cfg.diag_end_step,
                cfg.diag_init_weight,
                cfg.diag_final_weight,
            ),
            recon_loss_weight=get_linear_anneal_weight(
                step,
                cfg.recon_start_step,
                cfg.recon_end_step,
                cfg.recon_init_weight,
                cfg.recon_final_weight,
            ),
            viterbi_kl_loss_weight=get_linear_anneal_weight(
                step,
                cfg.viterbi_kl_start_step,
                cfg.viterbi_kl_end_step,
                cfg.viterbi_kl_init_weight,
                cfg.viterbi_kl_final_weight,
            ),
            viterbi_ot_loss_weight=get_linear_anneal_weight(
                step,
                cfg.viterbi_ot_start_step,
                cfg.viterbi_ot_end_step,
                cfg.viterbi_ot_init_weight,
                cfg.viterbi_ot_final_weight,
            ),
        )

    def _build_loss_values(self, out: AlignerForward) -> LossValues:
        return LossValues(
            recon=out.recon_loss.item(),
            crf=out.crf_loss.item(),
            diag=out.diag_loss.item(),
            viterbi_kl=out.viterbi_kl_loss.item(),
            viterbi_ot=out.viterbi_ot_loss.item(),
        )

    @override
    def train_step(
        self,
        batch: TTSBatch,
        epoch: int,
        step: int,
    ) -> tuple[torch.Tensor, LossValues, AlignerForward]:
        self.train_cfg: TrainConfigs
        self.model: NDAlignerTrainingModule

        x, x_lengths = batch.text, batch.text_lengths
        y, y_lengths = batch.spec, batch.spec_lengths
        cond = batch.cond

        loss_weights = self._get_loss_weights(step)

        compute_viterbi_loss = (loss_weights.viterbi_kl_loss_weight > 0.0) or (
            loss_weights.viterbi_ot_loss_weight > 0.0
        )
        compute_diagonal_loss = loss_weights.diag_loss_weight > 0.0

        forward_out = cast(
            NDAlignerTrainingModuleForward,
            self.model(
                x=x,
                x_lengths=x_lengths,
                y=y,
                y_lengths=y_lengths,
                cond=cond,
                loss_weights=loss_weights,
                compute_viterbi_loss=compute_viterbi_loss,
                compute_diagonal_loss=compute_diagonal_loss,
            ),
        )

        out = forward_out.aligner_output
        losses = self._build_loss_values(out)

        return forward_out.loss, losses, out

    @override
    def on_train_step_end(
        self,
        batch: TTSBatch,
        epoch: int,
        step: int,
        is_step_boundary: bool,
        weighted_loss: float,
        metrics: LossValues,
        output: AlignerForward,
    ):
        if is_step_boundary and (step % self.train_cfg.log_interval == 0 or step == 1):
            curr_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Step {step} | "
                + f"Total={weighted_loss:.4f} | "
                + f"{self._format_loss_values(metrics)} | "
                + f"LR={curr_lr:.2e}"
            )
            if self.logger:
                self.logger.log_metrics(
                    {
                        "Total_Loss": weighted_loss,
                        **{f"{k.capitalize()}_Loss": v for k, v in metrics._asdict().items()},
                    },
                    step,
                    prefix="Train",
                )
                self.logger.log_metrics(
                    {
                        f"{k.capitalize()}_Weight": v
                        for k, v in self._get_loss_weights(step)._asdict().items()
                    },
                    step,
                    prefix="Weights",
                )
                self.logger.log_learning_rate(curr_lr, step)

        if (
            is_step_boundary
            and self.logger
            and (step % self.train_cfg.img_log_interval == 0 or step == 1)
        ):
            self._log_visuals(batch, output, step, prefix="Train")

    @override
    def validation_step(
        self,
        batch: TTSBatch,
        epoch: int,
        step: int,
    ) -> tuple[float, LossValues, AlignerForward]:
        x, x_len = batch.text, batch.text_lengths
        y, y_len = batch.spec, batch.spec_lengths
        cond = batch.cond

        loss_weights = self._get_loss_weights(step)
        compute_diagonal_loss = loss_weights.diag_loss_weight > 0.0

        forward_out = cast(
            NDAlignerTrainingModuleForward,
            self.model(
                x=x,
                x_lengths=x_len,
                y=y,
                y_lengths=y_len,
                cond=cond,
                loss_weights=loss_weights,
                compute_viterbi_loss=True,
                compute_diagonal_loss=compute_diagonal_loss,
            ),
        )

        out = forward_out.aligner_output
        unweighted_losses = self._build_loss_values(out)

        return forward_out.loss.item(), unweighted_losses, out

    @override
    def on_validation_epoch_end(
        self,
        epoch: int,
        step: int,
        avg_val_loss: float,
        avg_metrics: LossValues,
        last_batch: TTSBatch | None,
        last_output: AlignerForward | None,
    ):
        if self.logger:
            log_dict = {
                "Total_Loss": avg_val_loss,
                **{f"{k.capitalize()}_Loss": v for k, v in avg_metrics._asdict().items()},
            }

            self.logger.log_metrics(
                log_dict,
                step,
                prefix="Valid",
            )

            if last_batch is not None and last_output is not None:
                self._log_visuals(
                    last_batch,
                    last_output,
                    step,
                    prefix="Valid",
                )

        if self.train_time_eval_logger is not None:
            self._run_train_time_timit_eval(
                epoch=epoch,
                step=step,
                is_test=False,
            )

    def _log_visuals(
        self,
        batch: TTSBatch,
        out: AlignerForward,
        step: int,
        prefix: str,
    ):
        assert self.logger is not None

        s_len = int(batch.spec_lengths[0].item())
        t_len = int(batch.text_lengths[0].item())
        tokens = self._decode_token_labels(batch.text[0, :t_len])

        self.logger.log_figure(
            f"{prefix}/GT_Mel",
            plot_spectrogram(batch.spec[0, :, :s_len]),
            step,
        )
        self.logger.log_figure(
            f"{prefix}/Mel_recon",
            plot_spectrogram(out.recon[0, :s_len].transpose(0, 1)),
            step,
        )

        # ------------------------------------------------------------------
        # Alignment maps
        # ------------------------------------------------------------------
        soft_attn_2d = out.soft_attn[0, :s_len, :t_len].detach()

        self.logger.log_figure(
            f"{prefix}/Soft_Alignment_Gamma",
            plot_alignment(soft_attn_2d, tokens=tokens),
            step,
        )

        if out.hard_attn is not None:
            hard_attn_2d = out.hard_attn[0, :s_len, :t_len].detach()
            self.logger.log_figure(
                f"{prefix}/Hard_Viterbi_Alignment",
                plot_alignment(hard_attn_2d, tokens=tokens),
                step,
            )

        # ------------------------------------------------------------------
        # Alpha / Beta / Gamma visualization
        # Raw exp(log_alpha) is usually not useful because values are tiny.
        # Use framewise-normalized distributions over reachable states.
        # ------------------------------------------------------------------
        valid_2d = self._strict_reachability_mask_2d(
            s_len=s_len,
            t_len=t_len,
            device=out.soft_attn.device,
        )

        log_alpha_2d = out.log_alpha[0, :s_len, :t_len].detach()
        log_beta_2d = out.log_beta[0, :s_len, :t_len].detach()

        alpha_prob = self._normalize_log_map_over_states(
            log_alpha_2d,
            valid_2d,
        ).exp()
        beta_prob = self._normalize_log_map_over_states(
            log_beta_2d,
            valid_2d,
        ).exp()

        self.logger.log_figure(
            f"{prefix}/Alpha_Distribution",
            plot_alignment(alpha_prob, tokens=tokens),
            step,
        )
        self.logger.log_figure(
            f"{prefix}/Beta_Distribution",
            plot_alignment(beta_prob, tokens=tokens),
            step,
        )

        # ------------------------------------------------------------------
        # Emission score maps
        # ------------------------------------------------------------------
        masked_raw_unary = out.masked_raw_unary[0, :s_len, :t_len].detach()

        plot_raw_unary = prepare_masked_score_for_plot(
            masked_raw_unary,
            fill_mode="min",  # or "p05", "mean"
        )

        self.logger.log_figure(
            f"{prefix}/Log_Unary_Raw",
            plot_alignment(plot_raw_unary, tokens=tokens),
            step,
        )

        log_b_2d = out.log_b[0, :s_len, :t_len].detach()

        plot_log_b = prepare_masked_score_for_plot(
            log_b_2d,
            fill_mode="min",
        )

        self.logger.log_figure(
            f"{prefix}/Log_Unary_Potential",
            plot_alignment(plot_log_b, tokens=tokens),
            step,
        )

        unary_support_prob = log_b_2d.float().exp().masked_fill(~valid_2d, 0.0)

        self.logger.log_figure(
            f"{prefix}/Unary_Sup_Probability",
            plot_alignment(unary_support_prob, tokens=tokens),
            step,
        )

        # ------------------------------------------------------------------
        # Audio logging
        # ------------------------------------------------------------------
        if self.model.vocoder is not None:
            hop_length = self.data_config.audio.hop_length
            audio_len = s_len * hop_length
            script = batch.scripts[0]

            wav_gt = self.model.mel2wav(batch.spec[:1])
            self.logger.log_audio(
                f"{prefix}/GT_Audio",
                wav_gt[0, :, :audio_len],
                step,
                self.data_config.audio.sr,
            )

            wav_hat = self.model.mel2wav(out.recon[:1])
            self.logger.log_audio(
                f"{prefix}/Predicted_Audio",
                wav_hat[0, :, :audio_len],
                step,
                self.data_config.audio.sr,
            )
            self.logger.log_text(f"{prefix}/Script", script, step)

    @override
    def on_training_end(self, epoch: int, step: int):
        if self.train_time_eval_logger is not None:
            from pathlib import Path

            from tts.preprocess.config import PreprocessConfig

            timit_benchmarker = TIMITBenchMarker(
                root_dir=self.data_config.extra_exp.timit_test_root_dir,
                ref_audio_sr=self.data_config.extra_exp.timit_sr,
                hyp_audio_sr=self.data_config.audio.sr,
                hyp_hop_length=self.data_config.audio.hop_length,
                tokenizer=TEXT_TOKENIZER,
                audio_config=self.data_config.audio,
                hyp_ignore_symbols=HYP_IGNORE_SYMBOLS,
                max_ref_words_per_hyp_word=5,
                spk_cond_dim=self.model_config.nd_aligner.spec_enc.cond_dim,
                rms_normalize=PreprocessConfig.rms_normalize,
                rms_target=PreprocessConfig.rms_target,
            )

            self.train_time_eval_logger = NDAlignerTimitLogger(
                base_dir=str(Path(self.data_config.extra_exp.base_dir) / "test"),
                experiment_name=self.data_config.extra_exp.exp_name,
                exp_variant=self.data_config.extra_exp.exp_variant,
                timit_benchmarker=timit_benchmarker,
                max_test_samples=None,
            )

            self._run_train_time_timit_eval(
                epoch=epoch,
                step=step,
                is_test=True,
            )

    def _format_loss_values(self, losses: LossValues) -> str:
        return (
            f"recon={losses.recon:.4f} | "
            + f"crf={losses.crf:.4f} | "
            + f"diag={losses.diag:.4f} | "
            + f"viterbi_kl={losses.viterbi_kl:.4f} | "
            + f"viterbi_ot={losses.viterbi_ot:.4f}"
        )

    def _metric_log_dict(
        self,
        metrics: LossValues,
    ) -> dict[str, float]:
        values = metrics._asdict()

        log_dict: dict[str, float] = {}

        for key, value in values.items():
            if key.endswith("_acc"):
                name = key.removesuffix("_acc")
                log_dict[f"{name.capitalize()}_Acc"] = value
            else:
                log_dict[f"{key.capitalize()}_Loss"] = value

        return log_dict

    @staticmethod
    def _normalize_log_map_over_states(
        log_map: torch.Tensor,
        valid_mask: torch.Tensor,
        neg_large: float = -1e9,
    ) -> torch.Tensor:
        """
        Framewise normalize a log-map over the text-token axis.

        Input:
            log_map    : (T, N)
            valid_mask : (T, N)

        Output:
            normalized_log_map[t, j]
            =
            log_map[t, j] - logsumexp_j log_map[t, j]
        """
        log_map = log_map.masked_fill(~valid_mask, neg_large)
        log_norm = torch.logsumexp(log_map, dim=-1, keepdim=True)
        out = log_map - log_norm
        out = out.masked_fill(~valid_mask, neg_large)
        return out

    @staticmethod
    def _masked_minmax_normalize(
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Normalize a 2D map to [0, 1] over valid positions only.
        Useful for visualizing raw evidence surfaces.
        """
        x_valid = x[valid_mask]
        if x_valid.numel() == 0:
            return torch.zeros_like(x)

        x_min = x_valid.min()
        x_max = x_valid.max()
        x_norm = (x - x_min) / (x_max - x_min + eps)
        x_norm = x_norm.masked_fill(~valid_mask, 0.0)
        return x_norm

    @staticmethod
    def _decode_token_labels(token_ids: torch.Tensor) -> list[str]:
        """
        Decode token ids into per-token IPA/BOS/EOS labels for y-axis plotting.

        Args:
            token_ids: (T,)

        Returns:
            labels: list[str], length T
        """
        token_ids = token_ids.detach().cpu().long().view(-1)

        return [
            cast(str, TEXT_TOKENIZER.decode(token_ids[i : i + 1])) for i in range(token_ids.numel())
        ]

    @staticmethod
    def _strict_reachability_mask_2d(
        s_len: int,
        t_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        t = torch.arange(s_len, device=device).view(s_len, 1)
        j = torch.arange(t_len, device=device).view(1, t_len)

        reachable_from_start = j <= t
        completable_to_end = (t_len - 1 - j) <= (s_len - 1 - t)

        return reachable_from_start & completable_to_end

    @torch.no_grad()
    def _run_train_time_timit_eval(
        self,
        *,
        epoch: int,
        step: int,
        is_test: bool = False,
    ) -> None:
        self.model: NDAlignerTrainingModule

        assert self.train_time_eval_logger is not None

        was_training = self.model.training

        self.model.eval()
        assert self.model.speaker_encoder is not None
        assert self.model.vocoder is not None

        if (
            self.timit_ckpt_manager is not None and is_test
        ):  # find-best-checkpoint and load best-aligner
            best_ckpt_path = self.timit_ckpt_manager.get_best_checkpoint_path()
            self.model.load_checkpoint(best_ckpt_path, device=self.device)

        row = self.train_time_eval_logger(
            epoch=epoch,
            step=step,
            aligner=self.model.nd_aligner,
            speaker_encoder=self.model.speaker_encoder,
            vocoder=self.model.vocoder,
        )

        print(
            "[TIMIT Eval] "
            + f"epoch={epoch} | "
            + f"step={step} | "
            + f"WBE={row['word_boundary_error'] * 1000.0:.2f}ms | "
            + f"P10={row['p_word_10ms']:.2f} | "
            + f"P25={row['p_word_25ms']:.2f} | "
            + f"P50={row['p_word_50ms']:.2f} | "
            + f"P100={row['p_word_100ms']:.2f} | "
            + f"MCD={row['mcd_dtw']:.4f} | "
            + f"Entropy={row['posterior_entropy']:.4f}"
        )

        if self.logger is not None:
            prefix = "TIMIT" if not is_test else "TIMIT-TEST"

            self.logger.log_metrics(
                {
                    "Word_Boundary_Error_ms": row["word_boundary_error"] * 1000.0,
                    "P_Word_10ms": row["p_word_10ms"],
                    "P_Word_25ms": row["p_word_25ms"],
                    "P_Word_50ms": row["p_word_50ms"],
                    "P_Word_100ms": row["p_word_100ms"],
                    "MCD_DTW": row["mcd_dtw"],
                    "Posterior_Entropy": row["posterior_entropy"],
                },
                step,
                prefix=prefix,
            )

        bae = float(row["word_boundary_error"])

        if (self.timit_ckpt_manager is not None) and (not is_test):
            self.timit_ckpt_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                step=step,
                epoch=epoch,
                loss_values=TimitCheckpointMetric(timit_bae=bae),
                save_periodic=False,
                save_best_step=True,
                save_best_epoch=False,
            )

        if was_training:
            self.model.train()


def main():
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description="Train Stage 1 Monotonic TTS")
    parser.add_argument("-c", "--data_config", type=str, help="Path to data config JSON")
    parser.add_argument("-m", "--train_module_config", type=str, help="Path to model config JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = load_config(args.data_config, DataConfig) if args.data_config else DataConfig()
    model_config = (
        load_config(args.train_module_config, NDAlignerTrainingModuleConfigs)
        if args.train_module_config
        else NDAlignerTrainingModuleConfigs()
    )

    trainer = NDAlignerTrainer(
        data_config=data_config,
        model_config=model_config,
        device=device,
    )
    trainer.run()


if __name__ == "__main__":
    main()
