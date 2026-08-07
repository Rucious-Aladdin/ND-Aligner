import argparse
import os
from typing import Any, NamedTuple, cast, override

import torch
from torch.utils.data import DataLoader

from tts.benchmark.timit.benchmarker import TIMITBenchMarker
from tts.config.ndaligner.data_config import DataConfig, LossConfigs, TrainConfigs
from tts.config.ndaligner.training_module_config import NDAlignerTrainingModuleConfigs
from tts.config.utils.io import load_config, save_config
from tts.data.data_types import LossValues, TrainBatch
from tts.data.datafactory import DataFactory
from tts.data.quantile_bucket_sampler import QuantileDurationBatchSampler
from tts.logger.timit_logger import NDAlignerTimitLogger
from tts.logger.utils.plot_alignment import plot_alignment
from tts.logger.utils.plot_spectrogram import plot_spectrogram
from tts.models.ndaligner import (
    AlignerForward,
    NDAlignerLossWeights,
    NDAlignerTrainingModule,
    NDAlignerTrainingModuleForward,
    init_nd_aligner_training_module,
)
from tts.train.utils.anneal import get_linear_anneal_weight
from tts.train.utils.checkpoint_manager import CheckpointManager
from tts.utils.set_seed import set_seed

from .utils.base_trainer import BaseTrainer


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
        assert self.model.nd_aligner.input_maker is not None
        self.data_config: DataConfig
        self.model_config: NDAlignerTrainingModuleConfigs

        self.train_time_eval_logger = None
        self.timit_ckpt_manager = None

        self.tokenizer = self.model.nd_aligner.input_maker.tokenizer

        if self.data_config.extra_exp.train_time_eval_logging:

            assert self.model.nd_aligner.input_maker is not None

            print("🧪 Train-time TIMIT eval is on.")
            timit_benchmarker = TIMITBenchMarker(
                root_dir=self.data_config.extra_exp.timit_root_dir,
                ref_audio_sr=self.data_config.extra_exp.timit_sr,
                hyp_audio_sr=self.data_config.audio.sr,
                hyp_hop_length=self.data_config.audio.hop_length,
                input_maker=self.model.nd_aligner.input_maker,
                hyp_ignore_symbols=self.tokenizer.ignore_symbols,
                seed=self.data_config.dataset.seed,
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
        data_factory = DataFactory(self.data_config)
        return data_factory.train_loader, data_factory.valid_loader

    @override
    def on_train_epoch_start(self, epoch: int) -> None:
        assert self.train_loader is not None

        batch_sampler = cast(
            QuantileDurationBatchSampler,
            self.train_loader.batch_sampler,
        )
        batch_sampler.set_epoch(epoch)

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
        )

    def _build_loss_values(self, out: AlignerForward) -> LossValues:
        return LossValues(
            recon=out.recon_loss.item(),
            crf=out.crf_loss.item(),
            diag=out.diag_loss.item(),
        )

    @override
    def train_step(
        self,
        batch: TrainBatch,
        epoch: int,
        step: int,
    ) -> tuple[torch.Tensor, LossValues, AlignerForward]:
        self.train_cfg: TrainConfigs
        self.model: NDAlignerTrainingModule

        x, x_lengths = batch.text, batch.text_lengths
        y, y_lengths = batch.spec, batch.spec_lengths  # mel or linspec.
        y_recon, y_recon_lengths = batch.recon_spec, batch.recon_spec_lengths  # always mel.

        cond = batch.cond

        loss_weights = self._get_loss_weights(step)

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
                y_recon=y_recon,
                y_recon_lengths=y_recon_lengths,
                compute_diagonal_loss=compute_diagonal_loss,
            ),
        )

        out = forward_out.aligner_output
        losses = self._build_loss_values(out)

        return forward_out.loss, losses, out

    @override
    def on_train_step_end(
        self,
        batch: TrainBatch,
        epoch: int,
        step: int,
        is_step_boundary: bool,
        weighted_loss: float,
        metrics: LossValues,
        output: AlignerForward,
    ):
        assert self.optimizer is not None

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

                if output.coupling_dec_out is not None:
                    dec = self.model.nd_aligner.spec_decoder

                    num_stages = len(output.coupling_dec_out.mel_losses)
                    decay = float(getattr(dec, "loss_decay_factor", 1.0))
                    normalize = bool(getattr(dec, "normalize_loss_weights", False))

                    weights_t = torch.tensor(
                        [decay**i for i in range(num_stages)],
                        device=output.coupling_dec_out.loss.device,
                        dtype=output.coupling_dec_out.loss.dtype,
                    )

                    if normalize:
                        weights_t = weights_t / weights_t.sum().clamp_min(1e-8)

                    coupling_metrics = {
                        "Weighted_Loss": float(output.coupling_dec_out.loss.detach().cpu()),
                    }

                    for i, loss_i in enumerate(output.coupling_dec_out.mel_losses):
                        coupling_metrics[f"Mel_Loss/stage_{i}"] = float(loss_i.detach().cpu())
                        coupling_metrics[f"Mel_Weight/stage_{i}"] = float(
                            weights_t[i].detach().cpu()
                        )

                    self.logger.log_metrics(
                        coupling_metrics,
                        step,
                        prefix="Train/CouplingDecoder",
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

        if (self.train_time_eval_logger is not None) and (
            step % self.data_config.extra_exp.train_time_eval_per_step == 0
        ):
            self._run_train_time_timit_eval(
                epoch=epoch,
                step=step,
                is_test=False,
            )

    @override
    def validation_step(
        self,
        batch: TrainBatch,
        epoch: int,
        step: int,
    ) -> tuple[float, LossValues, AlignerForward]:
        x, x_len = batch.text, batch.text_lengths

        y, y_len = batch.spec, batch.spec_lengths  # mel or linspec.
        y_recon, y_recon_len = batch.recon_spec, batch.recon_spec_lengths  # always mel.

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
                y_recon=y_recon,
                y_recon_lengths=y_recon_len,
                compute_hard_path=True,
                compute_diagonal_loss=compute_diagonal_loss,
            ),
        )

        out = forward_out.aligner_output
        unweighted_losses = self._build_loss_values(out)

        return forward_out.loss.item(), unweighted_losses, out

    @override
    def on_valid_epoch_end(
        self,
        epoch: int,
        step: int,
        avg_val_loss: float,
        avg_metrics: LossValues,
        last_batch: TrainBatch | None,
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

            if last_output is not None and last_output.coupling_dec_out is not None:
                dec = self.model.nd_aligner.spec_decoder

                num_stages = len(last_output.coupling_dec_out.mel_losses)
                decay = float(getattr(dec, "loss_decay_factor", 1.0))
                normalize = bool(getattr(dec, "normalize_loss_weights", False))

                weights_t = torch.tensor(
                    [decay**i for i in range(num_stages)],
                    device=last_output.coupling_dec_out.loss.device,
                    dtype=last_output.coupling_dec_out.loss.dtype,
                )

                if normalize:
                    weights_t = weights_t / weights_t.sum().clamp_min(1e-8)

                coupling_metrics = {
                    "Weighted_Loss": float(last_output.coupling_dec_out.loss.detach().cpu()),
                }

                for i, loss_i in enumerate(last_output.coupling_dec_out.mel_losses):
                    coupling_metrics[f"Mel_Loss/stage_{i}"] = float(loss_i.detach().cpu())
                    coupling_metrics[f"Mel_Weight/stage_{i}"] = float(weights_t[i].detach().cpu())

                self.logger.log_metrics(
                    coupling_metrics,
                    step,
                    prefix="Valid/CouplingDecoder",
                )

            if last_batch is not None and last_output is not None:
                self._log_visuals(
                    last_batch,
                    last_output,
                    step,
                    prefix="Valid",
                )

    def _log_visuals(
        self,
        batch: TrainBatch,
        out: AlignerForward,
        step: int,
        prefix: str,
    ):
        assert self.logger is not None

        s_len = int(batch.spec_lengths[0].item())
        recon_len = int(batch.recon_spec_lengths[0].item())
        t_len = int(batch.text_lengths[0].item())
        tokens = self._decode_token_labels(batch.text[0, :t_len])

        self.logger.log_figure(
            f"{prefix}/GT_Mel",
            plot_spectrogram(batch.recon_spec[0, :, :recon_len]),
            step,
        )
        self.logger.log_figure(
            f"{prefix}/Mel_recon",
            plot_spectrogram(out.recon[0, :s_len].transpose(0, 1)),
            step,
        )

        if out.coupling_dec_out is not None:
            for i, mel_i in enumerate(out.coupling_dec_out.mel_outputs):
                self.logger.log_figure(
                    f"{prefix}/Mel_st_{i}",
                    plot_spectrogram(mel_i[0, :recon_len].detach().transpose(0, 1)),
                    step,
                )

        if self.data_config.audio.feature_type != "mel":
            self.logger.log_figure(
                f"{prefix}/Alignment_Input_Feature",
                plot_spectrogram(batch.spec[0, :, :s_len]),
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

    @override
    def on_training_end(self, epoch: int, step: int):
        if self.train_time_eval_logger is not None:
            from pathlib import Path

            assert self.model.nd_aligner.input_maker is not None

            timit_benchmarker = TIMITBenchMarker(
                root_dir=self.data_config.extra_exp.timit_test_root_dir,
                ref_audio_sr=self.data_config.extra_exp.timit_sr,
                hyp_audio_sr=self.data_config.audio.sr,
                hyp_hop_length=self.data_config.audio.hop_length,
                input_maker=self.model.nd_aligner.input_maker,
                hyp_ignore_symbols=self.tokenizer.ignore_symbols,
                max_ref_words_per_hyp_word=5,
                seed=self.data_config.dataset.seed,
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
        )

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

    def _decode_token_labels(
        self,
        token_ids: torch.Tensor,
    ) -> list[str]:
        """
        Decode token ids into per-token IPA labels for y-axis plotting.

        Args:
            token_ids: (T,)

        Returns:
            labels: list[str], length T
        """
        token_ids = token_ids.detach().cpu().long().view(-1)

        return [
            cast(str, self.tokenizer.decode(token_ids[i : i + 1])) for i in range(token_ids.numel())
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
        assert self.optimizer is not None
        self.model: NDAlignerTrainingModule

        assert self.train_time_eval_logger is not None

        was_training = self.model.training

        self.model.eval()
        assert self.model.nd_aligner.input_maker is not None

        if (
            self.timit_ckpt_manager is not None and is_test
        ):  # find-best-checkpoint and load best-aligner
            best_ckpt_path = self.timit_ckpt_manager.get_best_checkpoint_path()
            self.model.load_checkpoint(best_ckpt_path, device=self.device)

        row = self.train_time_eval_logger.__call__(
            epoch=epoch,
            step=step,
            aligner=self.model.nd_aligner,
            is_test=is_test,
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


def main():  #
    # torch.backends.cudnn.benchmark = False

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

    set_seed(
        seed=data_config.train.seed,
        deterministic=True,
    )

    trainer = NDAlignerTrainer(
        data_config=data_config,
        model_config=model_config,
        device=device,
    )
    trainer.run()


if __name__ == "__main__":
    main()
