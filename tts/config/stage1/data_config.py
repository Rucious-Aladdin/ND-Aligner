# see https://github.com/jik876/hifi-gan/blob/master/config_v1.json
# use 22050 hifi-gan for compatibility
import os
import os.path
from dataclasses import dataclass, field

SAMPLE_RATE = 22050
DATA_PARENT_DIR = "/shared/data_zfs/blue2959"


@dataclass(frozen=True)
class AudioConfig:
    sr: int = SAMPLE_RATE

    # >> mel-spectrogram configs
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    num_freq: int = 1025


@dataclass(frozen=True)
class DatasetConfigs:
    # List of datasets to load: "ljspeech", "vctk", "libritts"
    dataset_list: list[str] = field(default_factory=lambda: ["libritts"])
    # dataset_list: list[str] = field(default_factory=lambda: ["vctk"])

    # Root directories for each dataset type (preprocessed)
    ljspeech_root: str = os.path.join(DATA_PARENT_DIR, "LJSpeech-1.1-preprocessed")
    vctk_root: str = os.path.join(DATA_PARENT_DIR, "VCTK-preprocessed")
    libritts_root: str = os.path.join(DATA_PARENT_DIR, "LibriTTS-preprocessed")

    data_cache_dir: str = os.path.join(DATA_PARENT_DIR, "cache-vctk")

    seed: int = 42
    val_ratio: float = 0.01

    min_duration_sec: float = 1.5
    max_duration_sec: float = 25.0


@dataclass(frozen=True)
class TrainConfigs:
    # --- Logging & Checkpointing ---
    log_dir: str = "./runs"
    run_name: str = "monotonic_tts_local_support_emission_vctk"
    continue_path: str = ""
    continue_dir: str = ""
    reset_optimizer: bool = False
    reset_scheduler: bool = False

    # --- Intervals (Steps or Epochs) ---
    val_sanity_check: bool = True
    val_interval: int = 1
    log_interval: int = 10
    img_log_interval: int = 500
    save_interval: int = 1000

    # --- Training Loop Limits ---
    max_epochs: int = 1000
    max_steps: int = 1000000

    # --- Hardware & Dataloader ---
    seed: int = 1234
    batch_size: int = 8
    val_batch_size: int = 8
    grad_accumulation_steps: int = 1
    num_workers: int = 8
    fp16_run: bool = False
    drop_last: bool = True

    # --- Optimizer & Scheduler ---
    lr: float = 1e-4
    lr_decay_rate: float = 0.99999768
    betas: list[float] = field(default_factory=lambda: [0.8, 0.99])
    eps: float = 1e-9
    weight_decay: float = 1e-6
    grad_clip_thresh: float = 5.0

    # --- Experiment Tracking ---
    use_tensorboard: bool = True

    # --- Checkpoint Management ---
    keep_best_count: int = 3
    keep_last_count: int = 3
    monitor_loss: str = "mel_recon"  # "mel" "aux" "dur" "align_nll" ...

    # --- Aligner Training Schedule ---
    freeze_aligner: bool = False
    freeze_aligner_until: int = 5000


@dataclass(frozen=True)
class LossConfigs:
    # Spec Decoder Loss Scheduling
    mel_recon_initial_weight: float = 25.0
    mel_recon_final_weight: float = 25.0
    mel_recon_start_step: int = 0
    mel_recon_end_step: int = 100000

    # Alignment NLL Loss Scheduling
    align_forward_initial_weight: float = 1.0
    align_forward_final_weight: float = 1.0
    align_forward_start_step: int = 2500
    align_forward_end_step: int = 5000

    # Alignment Diagonal Loss Scheduling
    align_diag_initial_weight: float = 5.0
    align_diag_final_weight: float = 0.0
    align_diag_start_step: int = 5000
    align_diag_end_step: int = 0

    # Alignment Viterbi KL Loss Scheduling
    align_viterbi_kl_initial_weight: float = 0.0
    align_viterbi_kl_final_weight: float = 0.0
    align_viterbi_kl_start_step: int = 15000
    align_viterbi_kl_end_step: int = 100000

    # Alignment Viterbi OT Loss Scheduling
    align_viterbi_ot_initial_weight: float = 0.0
    align_viterbi_ot_final_weight: float = 0.0
    align_viterbi_ot_start_step: int = 8000
    align_viterbi_ot_end_step: int = 30000


@dataclass(frozen=True)
class DataConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    dataset: DatasetConfigs = field(default_factory=DatasetConfigs)
    train: TrainConfigs = field(default_factory=TrainConfigs)
    loss: LossConfigs = field(default_factory=LossConfigs)
