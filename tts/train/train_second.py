import argparse
from dataclasses import replace
from typing import Any, NamedTuple, cast, override

import torch
from torch.utils.data import DataLoader

from tts.config.stage1.data_config import DataConfig as Stage1DataConfig
from tts.config.stage1.model_config import MonotonicTTSConfigs
from tts.config.stage2.data_config import Stage2DataConfig
from tts.config.stage2.model_config import DiffusionTTSConfigs
from tts.config.utils.io import load_config
from tts.data.data_types import TTSBatch
from tts.data.tts_datafactory import TTSDataFactory
from tts.logger.utils.plot_spectrogram import plot_spectrogram
from tts.logger.utils.plot_alignment import plot_alignment
from tts.models.diffusion_tts import DiffusionForwardOutput, KarrasTTSSynthesizer
from tts.models.init_diffusion_tts import init_diffusion_tts

from .base_trainer import BaseTrainer


class DiffusionLossValues(NamedTuple):
    total: float
    avg_sigma: float


class Stage2Trainer(BaseTrainer[Stage2DataConfig, DiffusionTTSConfigs]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model: KarrasTTSSynthesizer | None = None

    @override
    def setup_model(
        self,
    ) -> tuple[
        torch.nn.Module,
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LRScheduler | None,
    ]:
        print("🏗️ Initializing Stage 2 Model (Integrated)...")
        model = init_diffusion_tts(config=self.model_config, device=str(self.device))

        # >> FREEZE Stage 1 Backbone
        print("❄️ Freezing Stage 1 Backbone parameters...")
        for param in model.syn_backbone.parameters():
            param.requires_grad = False

        # Ensure Stage 2 (Estimator/Denoiser) is trainable
        for param in model.estimator.parameters():
            param.requires_grad = True

        model.train()

        model.print_parameter_summary()

        # Load Stage 1 Pre-trained Weights (Mandatory for Stage 2 training)
        if self.train_cfg.stage1_ckpt_path:
            print(f"📥 Loading Stage 1 weights from: {self.train_cfg.stage1_ckpt_path}")
            model.syn_backbone.load_checkpoint(self.train_cfg.stage1_ckpt_path, device=self.device)
        else:
            print("⚠️ Warning: Stage 2 training started without Stage 1 checkpoint.")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.train_cfg.lr,
            betas=cast(tuple[float, float], self.train_cfg.betas),
            eps=self.train_cfg.eps,
            weight_decay=self.train_cfg.weight_decay,
        )

        # Simple step decay for Diffusion refinement
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=1.0)

        return model, optimizer, scheduler

    @override
    def setup_dataloader(self) -> tuple[DataLoader[Any], DataLoader[Any]]:
        print("📦 Initializing datasets for Stage 2...")
        # Bridge config for DataFactory (it expects Stage 1 layout for loaders)
        from tts.config.stage1.data_config import DataConfig as S1DataConfig

        bridge_config = S1DataConfig(audio=self.data_config.audio, dataset=self.data_config.dataset)
        data_factory = TTSDataFactory(bridge_config)
        return data_factory.train_loader, data_factory.valid_loader

    @override
    def train_step(
        self,
        batch: TTSBatch,
        step: int,
    ) -> tuple[torch.Tensor, DiffusionLossValues, DiffusionForwardOutput]:
        assert self.model is not None
        # Integrated model handles backbone in eval mode automatically during forward,
        # but we ensure it here just in case.
        self.model.syn_backbone.eval()

        # Forward pass through integrated model (Backbone is used for conditioned features)
        out = cast(
            DiffusionForwardOutput,
            self.model(
                x=batch.text,
                x_lengths=batch.text_lengths,
                y=batch.spec,
                y_lengths=batch.spec_lengths,
                cond=batch.cond,
                text_cond_drop_prob=self.train_cfg.text_cond_drop_prob,
                spk_cond_drop_prob=self.train_cfg.spk_cond_drop_prob,
                text_cond_mask_ratio=self.train_cfg.text_cond_mask_ratio,
            ),
        )

        # >> EDM Preconditioning Loss: lambda(sigma) * ||D(x+n) - x||^2
        # mel_loss_unweighted is (B, n_mels, T_mel) squared error, already masked.
        loss_weight = self.model.estimator.get_loss_weight(out.sigma).view(-1, 1, 1)

        # Calculate weighted MSE loss normalized by valid frames
        # Use out.mask.sum() * n_mels for proper normalization across valid regions
        n_mels = batch.spec.size(1)
        weighted_loss_sum = (loss_weight * out.mel_loss_unweighted).sum()
        weighted_loss = weighted_loss_sum / (out.mask.sum() * n_mels + 1e-8)

        metrics = DiffusionLossValues(
            total=weighted_loss.item(),
            avg_sigma=out.sigma.mean().item(),
        )

        return weighted_loss, metrics, out

    @override
    def on_train_step_end(
        self,
        batch: TTSBatch,
        step: int,
        is_step_boundary: bool,
        weighted_loss: float,
        metrics: DiffusionLossValues,
        output: DiffusionForwardOutput,
    ):

        if is_step_boundary and (step % self.train_cfg.log_interval == 0 or step == 1):
            curr_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Step {step} | Loss: {weighted_loss:.4f} | Sigma: {metrics.avg_sigma:.3f} | LR: {curr_lr:.2e}"
            )

            if self.logger:
                self.logger.log_metrics(
                    {"Total_Loss": weighted_loss, "Avg_Sigma": metrics.avg_sigma},
                    step,
                    prefix="Train",
                )
                self.logger.log_learning_rate(curr_lr, step)

        if (
            is_step_boundary
            and self.logger
            and (step % self.train_cfg.img_log_interval == 0 or step == 1)
        ):
            s_len = int(batch.spec_lengths[0].item())
            script = batch.scripts[0]
            self.logger.log_figure(
                "Train/Denoised_Mel", plot_spectrogram(output.mel_hat[0, :, :s_len]), step
            )
            self.logger.log_figure("Train/GT_Mel", plot_spectrogram(batch.spec[0, :, :s_len]), step)

            assert self.model is not None
            # Log Ground-truth Audio (Reconstructed from GT Mel)
            if self.model.syn_backbone.vocoder is not None:
                hop_length = self.data_config.audio.hop_length
                audio_len = s_len * hop_length
                wav_gt = self.model.syn_backbone.mel2wav(batch.spec[:1])
                self.logger.log_audio(
                    "Train/GT_Audio",
                    wav_gt[0, :, :audio_len],
                    step,
                    self.data_config.audio.sr,
                )

            # 2. Full Inference Test (Stage 1 Path Prediction -> Stage 2 Diffusion 정제 -> Vocoder)
            inf_out = self.model.inference(
                x=batch.text[:1],
                x_lengths=batch.text_lengths[:1],
                cond=batch.cond[:1],
                n_steps=12,
                guidance_scale=(3.5, 1.5),
                cfg_mode="sequential",
            )
            # Note: inference returns (B, T_mel, n_mels)
            inf_mel_fig = plot_spectrogram(inf_out.mel_hat[0].transpose(0, 1))
            self.logger.log_figure("Train/Full_Inference_Mel", inf_mel_fig, step)

            if inf_out.wav_hat is not None:
                mel_len_inf = int(inf_out.dur[0].sum().item())
                hop_length = self.data_config.audio.hop_length
                audio_len_inf = mel_len_inf * hop_length
                self.logger.log_audio(
                    "Train/Inferred_Audio",
                    inf_out.wav_hat[0, :, :audio_len_inf],
                    step,
                    self.data_config.audio.sr,
                )
                self.logger.log_text("Train/Inferred_Script", script, step)

            # ------------------------------------------------------------------
            # Stage-1 logging (alignmant, text-features)
            # ------------------------------------------------------------------
            if output.soft_attn is not None:
                s_len = int(batch.spec_lengths[0].item())
                t_len = int(batch.text_lengths[0].item())

                soft_attn_2d = output.soft_attn[0, :s_len, :t_len].detach().cpu()
                self.logger.log_figure(
                    "Train/Soft_Alignment_Gamma",
                    plot_alignment(soft_attn_2d),
                    step,
                )

                if output.hard_attn is not None:
                    hard_attn_2d = output.hard_attn[0, :s_len, :t_len].detach().cpu()
                    self.logger.log_figure(
                        "Train/Hard_Viterbi_Alignment",
                        plot_alignment(hard_attn_2d),
                        step,
                    )

            if output.aligned_texts is not None:
                s_len_full = int(batch.spec_lengths[0].item())

                # output.aligned_texts: (B, C_text, T_mel)
                aligned_feat_2d = output.aligned_texts[0, :, :s_len_full].detach().cpu()

                self.logger.log_figure(
                    "Train/Stage1_Aligned_Feats",
                    plot_spectrogram(aligned_feat_2d),
                    step,
                )

    @override
    def validation_step(
        self,
        batch: TTSBatch,
        step: int,
    ) -> tuple[float, DiffusionLossValues, DiffusionForwardOutput]:
        assert self.model is not None
        out = cast(
            DiffusionForwardOutput,
            self.model(
                x=batch.text,
                x_lengths=batch.text_lengths,
                y=batch.spec,
                y_lengths=batch.spec_lengths,
                cond=batch.cond,
                spk_cond_drop_prob=self.train_cfg.spk_cond_drop_prob,
                text_cond_drop_prob=self.train_cfg.text_cond_drop_prob,
                text_cond_mask_ratio=self.train_cfg.text_cond_mask_ratio,
            ),
        )

        loss_weight = self.model.estimator.get_loss_weight(out.sigma).view(-1, 1, 1)
        n_mels = batch.spec.size(1)
        weighted_loss_sum = (loss_weight * out.mel_loss_unweighted).sum()
        val_loss = weighted_loss_sum / (out.mask.sum() * n_mels + 1e-8)

        metrics = DiffusionLossValues(
            total=val_loss.item(),
            avg_sigma=out.sigma.mean().item(),
        )

        return val_loss.item(), metrics, out

    @override
    def on_validation_epoch_end(
        self,
        step: int,
        avg_val_loss: float,
        avg_metrics: DiffusionLossValues,
        last_batch: TTSBatch | None,
        last_output: DiffusionForwardOutput | None,
    ):
        assert self.model is not None
        if self.logger:
            self.logger.log_metrics(
                {"Total_Loss": avg_val_loss, "Avg_Sigma": avg_metrics.avg_sigma},
                step,
                prefix="Valid",
            )

            if last_batch is not None and last_output is not None:
                s_len = int(last_batch.spec_lengths[0].item())
                script = last_batch.scripts[0]
                # 1. GT vs Denoised (Teacher-forced context)
                self.logger.log_figure(
                    "Valid/GT_Mel", plot_spectrogram(last_batch.spec[0, :, :s_len]), step
                )
                self.logger.log_figure(
                    "Valid/Denoised_Mel", plot_spectrogram(last_output.mel_hat[0, :, :s_len]), step
                )

                # Log Ground-truth Audio (Reconstructed from GT Mel)
                if self.model.syn_backbone.vocoder is not None:
                    hop_length = self.data_config.audio.hop_length
                    audio_len = s_len * hop_length
                    wav_gt = self.model.syn_backbone.mel2wav(last_batch.spec[:1])
                    self.logger.log_audio(
                        "Valid/GT_Audio",
                        wav_gt[0, :, :audio_len],
                        step,
                        self.data_config.audio.sr,
                    )

                # 2. Full Inference Test (Stage 1 Path Prediction -> Stage 2 Diffusion 정제 -> Vocoder)
                inf_out = self.model.inference(
                    x=last_batch.text[:1],
                    x_lengths=last_batch.text_lengths[:1],
                    cond=last_batch.cond[:1],
                    n_steps=35,
                    guidance_scale=(5.0, 3.0),
                    cfg_mode="sequential",
                )
                # Note: inference returns (B, T_mel, n_mels)
                inf_mel_fig = plot_spectrogram(inf_out.mel_hat[0].transpose(0, 1))
                self.logger.log_figure("Valid/Full_Inference_Mel", inf_mel_fig, step)

                if inf_out.wav_hat is not None:
                    mel_len_inf = int(inf_out.dur[0].sum().item())
                    hop_length = self.data_config.audio.hop_length
                    audio_len_inf = mel_len_inf * hop_length
                    self.logger.log_audio(
                        "Valid/Inferred_Audio",
                        inf_out.wav_hat[0, :, :audio_len_inf],
                        step,
                        self.data_config.audio.sr,
                    )
                    self.logger.log_text("Valid/Inferred_Script", script, step)


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 Diffusion TTS (EDM)")
    parser.add_argument("-c", "--data_config", type=str, help="Path to Stage 2 data config JSON")
    parser.add_argument("-m", "--model_config", type=str, help="Path to Stage 2 model config JSON")

    parser.add_argument(
        "--s1_data_config",
        type=str,
        default=None,
        help=(
            "Optional path to Stage 1 data config JSON. "
            "If provided, it overrides data_config.audio and data_config.dataset."
        ),
    )
    parser.add_argument(
        "--s1_model_config",
        type=str,
        default=None,
        help=(
            "Optional path to Stage 1 model config JSON. "
            "If provided, it overrides model_config.s1_config."
        ),
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = (
        load_config(args.data_config, Stage2DataConfig) if args.data_config else Stage2DataConfig()
    )

    model_config = (
        load_config(args.model_config, DiffusionTTSConfigs)
        if args.model_config
        else DiffusionTTSConfigs()
    )

    # Override Stage 2 audio/dataset config from Stage 1 data config.
    # Stage 2 keeps its own train config.
    if args.s1_data_config is not None:
        s1_data_config = load_config(args.s1_data_config, Stage1DataConfig)
        data_config = replace(
            data_config,
            audio=s1_data_config.audio,
            dataset=s1_data_config.dataset,
        )

    # Override Stage 1 model config inside Stage 2 model config.
    if args.s1_model_config is not None:
        s1_model_config = load_config(args.s1_model_config, MonotonicTTSConfigs)
        model_config = replace(
            model_config,
            s1_config=s1_model_config,
        )

    trainer = Stage2Trainer(data_config, model_config, device)
    trainer.run()


if __name__ == "__main__":
    main()
