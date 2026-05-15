import argparse
from typing import Any, cast, override

import torch
from torch.utils.data import DataLoader
from tts.tokenizer.text_tokenizer import TextTokenizer
from tts.config.stage1.data_config import DataConfig, LossConfigs, TrainConfigs
from tts.config.stage1.model_config import MonotonicTTSConfigs
from tts.config.utils.io import load_config
from tts.data.data_types import LossValues, LossWeights, TTSBatch
from tts.data.tts_datafactory import TTSDataFactory
from tts.logger.utils.plot_alignment import plot_alignment
from tts.logger.utils.plot_spectrogram import plot_spectrogram
from tts.models.init_monotonic_tts import init_monotonic_tts
from tts.models.monotonic_tts import MonotonicTTSSynthesizer, SynthesizerForwardOutput
from tts.utils.anneal import get_linear_anneal_weight
from functools import cached_property
from .base_trainer import BaseTrainer

TEXT_TOKENIZER = TextTokenizer()


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


class Stage1Trainer(BaseTrainer[DataConfig, MonotonicTTSConfigs]):
    @override
    def setup_model(
        self,
    ) -> tuple[
        torch.nn.Module,
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LRScheduler | None,
    ]:
        print("🏗️ Initializing model...")
        model = init_monotonic_tts(config=self.model_config).to(self.device)
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
    def setup_dataloader(self) -> tuple[DataLoader[Any], DataLoader[Any]]:
        print("📦 Initializing datasets...")
        data_factory = TTSDataFactory(self.data_config)
        return data_factory.train_loader, data_factory.valid_loader

    def _get_loss_weights(self, step: int) -> LossWeights:
        """Calculates current loss weights based on annealing schedule."""
        cfg: LossConfigs = self.data_config.loss

        return LossWeights(
            dur=1.0,
            mel_recon=get_linear_anneal_weight(
                step,
                cfg.mel_recon_start_step,
                cfg.mel_recon_end_step,
                cfg.mel_recon_initial_weight,
                cfg.mel_recon_final_weight,
            ),
            align_forward=get_linear_anneal_weight(
                step,
                cfg.align_forward_start_step,
                cfg.align_forward_end_step,
                cfg.align_forward_initial_weight,
                cfg.align_forward_final_weight,
            ),
            align_diag=get_linear_anneal_weight(
                step,
                cfg.align_diag_start_step,
                cfg.align_diag_end_step,
                cfg.align_diag_initial_weight,
                cfg.align_diag_final_weight,
            ),
            align_viterbi_kl=get_linear_anneal_weight(
                step,
                cfg.align_viterbi_kl_start_step,
                cfg.align_viterbi_kl_end_step,
                cfg.align_viterbi_kl_initial_weight,
                cfg.align_viterbi_kl_final_weight,
            ),
            align_viterbi_ot=get_linear_anneal_weight(
                step,
                cfg.align_viterbi_ot_start_step,
                cfg.align_viterbi_ot_end_step,
                cfg.align_viterbi_ot_initial_weight,
                cfg.align_viterbi_ot_final_weight,
            ),
        )

    @override
    def train_step(
        self,
        batch: TTSBatch,
        step: int,
    ) -> tuple[torch.Tensor, LossValues, SynthesizerForwardOutput]:
        self.train_cfg: TrainConfigs
        self.model: MonotonicTTSSynthesizer

        # --- Aligner freezing schedule ---
        if self.train_cfg.freeze_aligner:
            assert self.model.crf_aligner is not None

            should_freeze = step < self.train_cfg.freeze_aligner_until

            modules_to_freeze: dict[str, torch.nn.Module | None] = {
                # "text_encoder": self.model.text_encoder,
                # "spec_encoder": self.model.spec_encoder,
            }

            params_to_freeze: dict[str, torch.nn.Parameter | None] = {}

            # Check one representative parameter to avoid printing every step.
            first_param: torch.nn.Parameter | None = None

            for module in modules_to_freeze.values():
                if module is None:
                    continue
                first_param = next(module.parameters(), None)
                if first_param is not None:
                    break

            if first_param is None:
                for param in params_to_freeze.values():
                    if param is not None:
                        first_param = param
                        break

            if first_param is not None and first_param.requires_grad == should_freeze:
                state_str = "FREEZING" if should_freeze else "UNFREEZING"

                module_names = [
                    name for name, module in modules_to_freeze.items() if module is not None
                ]
                param_names = [
                    name for name, param in params_to_freeze.items() if param is not None
                ]

                names = ", ".join(module_names + param_names)
                print(f"❄️ {state_str} {names} at step {step}")

                for module in modules_to_freeze.values():
                    if module is None:
                        continue
                    for param in module.parameters():
                        param.requires_grad = not should_freeze

                for param in params_to_freeze.values():
                    if param is None:
                        continue
                    param.requires_grad = not should_freeze

        x, x_lengths = batch.text, batch.text_lengths
        y, y_lengths = batch.spec, batch.spec_lengths
        cond = batch.cond

        loss_weights = self._get_loss_weights(step)

        compute_viterbi_loss = (loss_weights.align_viterbi_kl > 0.0) or (
            loss_weights.align_viterbi_ot > 0.0
        )
        compute_diagonal_loss = loss_weights.align_diag > 0.0

        out = cast(
            SynthesizerForwardOutput,
            self.model(
                x=x,
                x_lengths=x_lengths,
                y=y,
                y_lengths=y_lengths,
                cond=cond,
                compute_viterbi_loss=compute_viterbi_loss,
                compute_diagonal_loss=compute_diagonal_loss,
            ),
        )

        weighted_total_loss = (
            +(out.dur_loss * loss_weights.dur)
            + (out.mel_recon_loss * loss_weights.mel_recon)
            + (out.align_forward_loss * loss_weights.align_forward)
            + (out.align_diag_loss * loss_weights.align_diag)
            + (out.align_viterbi_kl_loss * loss_weights.align_viterbi_kl)
            + (out.align_viterbi_ot_loss * loss_weights.align_viterbi_ot)
        )

        losses = LossValues(
            dur=out.dur_loss.item(),
            mel_recon=out.mel_recon_loss.item(),
            align_forward=out.align_forward_loss.item(),
            align_diag=out.align_diag_loss.item(),
            align_viterbi_kl=out.align_viterbi_kl_loss.item(),
            align_viterbi_ot=out.align_viterbi_ot_loss.item(),
        )
        return weighted_total_loss, losses, out

    @override
    def on_train_step_end(
        self,
        batch: TTSBatch,
        step: int,
        is_step_boundary: bool,
        weighted_loss: float,
        metrics: LossValues,
        output: SynthesizerForwardOutput,
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
        step: int,
    ) -> tuple[float, LossValues, SynthesizerForwardOutput]:
        x, x_len = batch.text, batch.text_lengths
        y, y_len = batch.spec, batch.spec_lengths
        cond = batch.cond

        loss_weights = self._get_loss_weights(step)

        compute_viterbi_loss = (loss_weights.align_viterbi_kl > 0.0) or (
            loss_weights.align_viterbi_ot > 0.0
        )
        compute_diagonal_loss = loss_weights.align_diag > 0.0

        out = cast(
            SynthesizerForwardOutput,
            self.model(
                x,
                x_len,
                y,
                y_len,
                cond,
                compute_viterbi_loss=compute_viterbi_loss,
                compute_diagonal_loss=compute_diagonal_loss,
            ),
        )

        weighted_val_loss = (
            +(out.dur_loss * loss_weights.dur)
            + (out.mel_recon_loss * loss_weights.mel_recon)
            + (out.align_forward_loss * loss_weights.align_forward)
            + (out.align_diag_loss * loss_weights.align_diag)
            + (out.align_viterbi_kl_loss * loss_weights.align_viterbi_kl)
            + (out.align_viterbi_ot_loss * loss_weights.align_viterbi_ot)
        ).item()

        losses = LossValues(
            dur=out.dur_loss.item(),
            mel_recon=out.mel_recon_loss.item(),
            align_forward=out.align_forward_loss.item(),
            align_diag=out.align_diag_loss.item(),
            align_viterbi_kl=out.align_viterbi_kl_loss.item(),
            align_viterbi_ot=out.align_viterbi_ot_loss.item(),
        )
        return weighted_val_loss, losses, out

    @override
    def on_validation_epoch_end(
        self,
        step: int,
        avg_val_loss: float,
        avg_metrics: LossValues,
        last_batch: TTSBatch | None,
        last_output: SynthesizerForwardOutput | None,
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

    def _log_visuals(
        self,
        batch: TTSBatch,
        out: SynthesizerForwardOutput,
        step: int,
        prefix: str,
    ):
        assert self.logger is not None

        s_len = int(batch.spec_lengths[0].item())
        t_len = int(batch.text_lengths[0].item())
        tokens = self._decode_token_labels(batch.text[0, :t_len])

        # ------------------------------------------------------------------
        # Build first-sample valid monotone mask for visualization.
        # This is better than a full rectangular mask because alpha/beta/gamma
        # contain unreachable monotone states.
        # ------------------------------------------------------------------

        self.logger.log_figure(
            f"{prefix}/GT_Mel",
            plot_spectrogram(batch.spec[0, :, :s_len]),
            step,
        )
        self.logger.log_figure(
            f"{prefix}/Mel_recon",
            plot_spectrogram(out.mel_recon[0, :s_len].transpose(0, 1)),
            step,
        )

        # ------------------------------------------------------------------
        # Alignment maps
        # ------------------------------------------------------------------
        soft_attn_2d = out.attn[0, :s_len, :t_len].detach()

        self.logger.log_figure(
            f"{prefix}/Soft_Alignment_Gamma",
            plot_alignment(soft_attn_2d, tokens=tokens),
            step,
        )

        if out.hard_gamma is not None:
            hard_attn_2d = out.hard_gamma[0, :s_len, :t_len].detach()
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
            device=out.attn.device,
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
            f"{prefix}/Alpha_Prefix_State_Distribution",
            plot_alignment(alpha_prob, tokens=tokens),
            step,
        )
        self.logger.log_figure(
            f"{prefix}/Beta_Suffix_State_Distribution",
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

            wav_hat = self.model.mel2wav(out.mel_recon[:1])
            self.logger.log_audio(
                f"{prefix}/Predicted_Audio",
                wav_hat[0, :, :audio_len],
                step,
                self.data_config.audio.sr,
            )
            self.logger.log_text(f"{prefix}/Script", script, step)

    def _log_inference(self, batch: TTSBatch, step: int):
        assert self.logger is not None

        x, x_len = batch.text[:1], batch.text_lengths[:1]
        t_len = int(x_len[0].item())
        tokens = self._decode_token_labels(x[0, :t_len])
        cond = batch.cond[:1]
        script = batch.scripts[0]

        inf_out = self.model.inference(x, x_len, cond)
        mel_len_inf = int(inf_out.dur[0].sum().item())
        s_len_inf = mel_len_inf

        self.logger.log_figure(
            "Valid/Inferred_Mel",
            plot_spectrogram(inf_out.mel_hat[0, :s_len_inf].transpose(0, 1)),
            step,
        )
        self.logger.log_figure(
            "Valid/Inference_Duration_Alignment",
            plot_alignment(inf_out.attn[0, :s_len_inf, :t_len], tokens=tokens),
            step,
        )

        if inf_out.wav_hat is not None:
            hop_length = self.data_config.audio.hop_length
            audio_len_inf = mel_len_inf * hop_length
            self.logger.log_audio(
                "Valid/Inferred_Audio",
                inf_out.wav_hat[0, :, :audio_len_inf],
                step,
                self.data_config.audio.sr,
            )
            self.logger.log_text("Valid/Inferred_Script", script, step)

    def _format_loss_values(self, losses: LossValues) -> str:
        return (
            f"mel_recon={losses.mel_recon:.4f} | "
            + f"dur={losses.dur:.4f} | "
            + f"align_forward={losses.align_forward:.4f} | "
            + f"align_diag={losses.align_diag:.4f} | "
            + f"align_viterbi_kl={losses.align_viterbi_kl:.4f} | "
            + f"align_viterbi_ot={losses.align_viterbi_ot:.4f}"
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


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 Monotonic TTS")
    parser.add_argument("-c", "--data_config", type=str, help="Path to data config JSON")
    parser.add_argument("-m", "--model_config", type=str, help="Path to model config JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = load_config(args.data_config, DataConfig) if args.data_config else DataConfig()
    model_config = (
        load_config(args.model_config, MonotonicTTSConfigs)
        if args.model_config
        else MonotonicTTSConfigs()
    )

    trainer = Stage1Trainer(data_config, model_config, device)
    trainer.run()


if __name__ == "__main__":
    main()
