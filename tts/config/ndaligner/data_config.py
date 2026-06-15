# see https://github.com/jik876/hifi-gan/blob/master/config_v1.json
# use 22050 hifi-gan for compatibility
import os
import os.path
from dataclasses import dataclass, field

SAMPLE_RATE = 22050
DATA_PARENT_DIR = "/shared/data_zfs/blue2959"

N_MELS = 80

DATASETS = ["vctk"]
SEED = 42


def cache_dir_name() -> str:
    dataset_tag = "+".join(DATASETS)
    feature_tag = f"mel{N_MELS}"

    return f"cache-{dataset_tag}-{feature_tag}"


@dataclass(frozen=True)
class AudioConfig:
    sr: int = SAMPLE_RATE

    # >> mel-spectrogram configs
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = N_MELS
    f_min: float = 0.0
    f_max: float = 8000.0
    num_freq: int = 1025


@dataclass(frozen=True)
class DatasetConfigs:
    # List of datasets to load: "ljspeech", "vctk", "libritts"
    dataset_list: list[str] = field(default_factory=lambda: list(DATASETS))

    # Root directories for each dataset type (preprocessed)
    ljspeech_root: str = os.path.join(DATA_PARENT_DIR, "LJSpeech-1.1-preprocessed")
    ljspeech_num_test_samples: int = 30

    vctk_root: str = os.path.join(DATA_PARENT_DIR, "VCTK-preprocessed")
    vctk_test_speakers: list[str] = field(
        default_factory=lambda: ["p225", "p226", "p227", "p228", "p229", "p232"]
    )

    libritts_root: str = os.path.join(DATA_PARENT_DIR, "LibriTTS-preprocessed")

    data_cache_dir: str = field(
        default_factory=lambda: os.path.join(DATA_PARENT_DIR, cache_dir_name())
    )

    seed: int = SEED
    val_ratio: float = 0.01
    num_buckets: int = 10

    min_duration_sec: float = 1.5
    max_duration_sec: float = 25.0


@dataclass(frozen=True)
class ExperimentConfigs:
    train_time_eval_logging: bool = True

    base_dir: str = "/shared/data_zfs/blue2959/ND_Aligner/experiments"
    exp_name: str = "vctk_base+delta_mel+recon_weight_1.0"
    exp_variant: str = "vctk_base+delta_mel+lower_recon_weight"

    timit_root_dir: str = "/shared/data_zfs/blue2959/TIMIT/TRAIN"
    timit_test_root_dir: str = "/shared/data_zfs/blue2959/TIMIT/TEST"
    timit_sr: int = 16_000
    timit_max_num_test_samples: int = 200


@dataclass(frozen=True)
class TrainConfigs:
    # --- Logging & Checkpointing ---
    log_dir: str = "./runs"
    run_name: str = "nd_aligner_vctk"

    continue_path: str = ""
    continue_dir: str = ""
    reset_optimizer: bool = False
    reset_scheduler: bool = False

    # --- Intervals (Steps or Epochs) ---
    val_sanity_check: bool = True
    val_sanity_check_full_epoch: bool = True
    val_interval: int = 1
    val_interval_step: int = 500
    val_interval_step_skip_hook: bool = False

    log_interval: int = 10
    img_log_interval: int = 1000
    save_interval: int = 1000

    # --- Training Loop Limits ---
    max_epochs: int = 20
    max_steps: int = 300000  # deprecated

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
    monitor_loss: str = "recon"  # "recon" ...

    # Viterbi Maximum-path only training
    viterbi_only_training: bool = False
    detach_decoder: bool = True


@dataclass(frozen=True)
class LossConfigs:
    # Spec Decoder Loss Scheduling
    recon_init_weight: float = 5.0
    recon_final_weight: float = 5.0
    recon_start_step: int = 0
    recon_end_step: int = 100000

    # Alignment NLL Loss Scheduling
    crf_init_weight: float = 5.0
    crf_final_weight: float = 5.0
    crf_start_step: int = 0
    crf_end_step: int = 10000

    # Alignment Diagonal Loss Scheduling
    diag_init_weight: float = 5.0
    diag_final_weight: float = 0.0
    diag_start_step: int = 1000
    diag_end_step: int = 1010

    # Alignment Viterbi KL Loss Scheduling
    viterbi_kl_init_weight: float = 0.0
    viterbi_kl_final_weight: float = 0.0
    viterbi_kl_start_step: int = 50000
    viterbi_kl_end_step: int = 100000

    # Alignment Viterbi OT Loss Scheduling
    viterbi_ot_init_weight: float = 0.0
    viterbi_ot_final_weight: float = 0.0
    viterbi_ot_start_step: int = 150000
    viterbi_ot_end_step: int = 200000


@dataclass(frozen=True)
class DataConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    dataset: DatasetConfigs = field(default_factory=DatasetConfigs)
    train: TrainConfigs = field(default_factory=TrainConfigs)
    extra_exp: ExperimentConfigs = field(default_factory=ExperimentConfigs)
    loss: LossConfigs = field(default_factory=LossConfigs)
