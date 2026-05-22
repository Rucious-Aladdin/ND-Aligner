import os
from dataclasses import dataclass, field

from ..stage1.data_config import AudioConfig, DatasetConfigs


@dataclass(frozen=True)
class DiffusionTrainConfigs:
    # --- Logging & Checkpointing ---
    log_dir: str = "./runs"
    # run_name: str = "karras_vctk+progress+smoothing"
    run_name: str = "karras_libritts+progress+smoothing"

    continue_path: str = ""
    continue_dir: str = ""
    reset_optimizer: bool = False
    reset_scheduler: bool = False

    # --- Stage 1 Model Loading ---
    # Path to the frozen stage 1 checkpoint (REQUIRED for Stage 2 training)
    stage1_ckpt_path: str = "./checkpoints/unet_unary_checkpoints/ckpt_step_235000_libritts.pth"
    # stage1_ckpt_path: str = "./checkpoints/unet_unary_checkpoints/ckpt_step_524000_vctk.pth"

    # --- Intervals ---
    val_sanity_check: bool = True
    val_interval: int = 1
    log_interval: int = 10
    img_log_interval: int = 500
    save_interval: int = 1000

    # --- Training Loop ---
    max_epochs: int = 1000
    max_steps: int = 1000000

    # --- Hardware & Dataloader ---
    seed: int = 1234
    batch_size: int = 32
    val_batch_size: int = 8
    grad_accumulation_steps: int = 1
    num_workers: int = 8
    fp16_run: bool = False
    drop_last: bool = True

    # --- Optimizer & Scheduler ---
    lr: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 1e-6
    grad_clip_thresh: float = 1.0

    # --- Experiment Tracking ---
    use_tensorboard: bool = True

    # --- Classifier-Free Guidance ---
    text_cond_mask_ratio: float = 0.10
    text_cond_drop_prob: float = 0.10
    spk_cond_drop_prob: float = 0.15

    # --- Checkpoint Management ---
    keep_best_count: int = 3
    keep_last_count: int = 3
    monitor_loss: str = "total"


@dataclass(frozen=True)
class Stage2DataConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    dataset: DatasetConfigs = field(default_factory=DatasetConfigs)
    train: DiffusionTrainConfigs = field(default_factory=DiffusionTrainConfigs)
