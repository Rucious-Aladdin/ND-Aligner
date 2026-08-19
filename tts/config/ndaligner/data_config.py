# see https://github.com/jik876/hifi-gan/blob/master/config_v1.json
# use 22050 hifi-gan for compatibility
import os
import os.path
from dataclasses import dataclass, field

from ..preprocess.preprocess_config import SPK_ENCODER_TYPE, PreprocessConfigs

INPUT_FEATURE_TYPE = "mel"
# INPUT_FEATURE_TYPE: str = "linspec"

DATA_PARENT_DIR = "/shared/data_zfs/blue2959"
TOKENIZER_TYPE: str = "espeak"  # "arpa"
ARPA_TOKENIZER_LEXION_PATH = "./tts/baseline/FastSpeech2/lexicon/vctk-lexicon.txt"

N_MELS = 80
N_FFT = 1024

# ==================================

SAMPLE_RATE = 16000

# hop = 10ms / win = 25ms
WIN_LENGTH = 400
HOP_LENGTH = 160

# # hop = 5ms / win = 25ms
# WIN_LENGTH = 400
# HOP_LENGTH = 80

# ==================================

# # SAMPLE_RATE=22050

# # hop = 11.67ms / win = 46.8ms
# WIN_LENGTH = 1024
# HOP_LENGTH = 256

# # hop = 11.67ms / win = 23.4ms
# WIN_LENGTH = 1024
# HOP_LENGTH = 256

# ==================================


# DATASETS = ["vctk", "libritts"]
DATASETS = ["vctk"]

SEED = 42


def cache_dir_name() -> str:
    dataset_tag = "+".join(DATASETS)

    if INPUT_FEATURE_TYPE == "mel":
        feature_tag = f"mel{N_MELS}_sr{SAMPLE_RATE}_hop{HOP_LENGTH}_win{WIN_LENGTH}"
    elif INPUT_FEATURE_TYPE == "linspec":
        n_freq = N_FFT // 2 + 1
        feature_tag = f"linspec{n_freq}"
    else:
        raise ValueError()

    tokenizer_tag = f"{TOKENIZER_TYPE}"
    spk_encoder_tag = f"{SPK_ENCODER_TYPE}"
    return (
        f"cache-{dataset_tag}-feature_{feature_tag}-tokenizer_{tokenizer_tag}-spk_{spk_encoder_tag}"
    )


def dataset_postfix(x) -> str:
    if SAMPLE_RATE == 16_000:
        return f"{x}-16k"
    elif SAMPLE_RATE == 22_050:
        return x
    else:
        raise ValueError()


@dataclass(frozen=True)
class AudioConfigs:
    sr: int = SAMPLE_RATE
    feature_type: str = INPUT_FEATURE_TYPE

    # >> mel-spectrogram configs
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    win_length: int = WIN_LENGTH
    n_mels: int = N_MELS
    f_min: float = 0.0
    f_max: float = 8000.0
    num_freq: int = field(default_factory=lambda: N_FFT // 2 + 1)


@dataclass(frozen=True)
class DatasetConfigs:
    # List of datasets to load: "ljspeech", "vctk", "libritts"
    dataset_list: list[str] = field(default_factory=lambda: list(DATASETS))

    # Root directories for each dataset type (preprocessed)

    # LJ-Speech
    ljspeech_root: str = os.path.join(
        DATA_PARENT_DIR,
        dataset_postfix(
            "LJSpeech-1.1-preprocessed",
        ),
    )
    ljspeech_num_test_samples: int = 30

    # VCTK
    vctk_root: str = os.path.join(
        DATA_PARENT_DIR,
        dataset_postfix(
            "VCTK-preprocessed-trimmed",
        ),
    )
    vctk_test_speakers: list[str] = field(default_factory=lambda: [])

    # LibriTTS
    libritts_root: str = os.path.join(
        DATA_PARENT_DIR,
        dataset_postfix(
            "LibriTTS-preprocessed-trimmed",
        ),
    )
    libritts_subsets: list[str] = field(
        default_factory=lambda: ["train-clean-100", "train-clean-360"]
    )  # train-clean-360

    data_cache_dir: str = field(
        default_factory=lambda: os.path.join(DATA_PARENT_DIR, cache_dir_name())
    )

    seed: int = SEED
    val_ratio: float = 0.01
    num_buckets: int = 10

    min_duration_sec: float = 1.5
    max_duration_sec: float = 15.0

    tokenizer_type: str = TOKENIZER_TYPE
    fastspeech2_lexicon_path: str = ARPA_TOKENIZER_LEXION_PATH


@dataclass(frozen=True)
class ExperimentConfigs:
    train_time_eval_logging: bool = True
    train_time_eval_per_step: int = 1000

    base_dir: str = "/shared/data_zfs/blue2959/ND_Aligner/experiments/v2.0/main"
    exp_name: str = "vctk+libritts+full+sr16k+hop10ms+win25ms"
    exp_variant: str = "vctk+libritts+full+sr16k+hop10ms+win25ms"

    timit_root_dir: str = "/shared/data_zfs/blue2959/TIMIT/TRAIN"
    timit_test_root_dir: str = "/shared/data_zfs/blue2959/TIMIT/TEST"
    timit_sr: int = 16_000
    timit_max_num_test_samples: int = 250


@dataclass(frozen=True)
class TrainConfigs:
    # --- Logging & Checkpointing ---
    log_dir: str = "./runs"
    run_name: str = "nd_aligner_main_vctk+libritts+full+sr16k+hop10ms+win25ms"

    continue_path: str = ""
    continue_dir: str = ""
    reset_optimizer: bool = False
    reset_scheduler: bool = False

    # --- Intervals (Steps or Epochs) ---
    val_sanity_check: bool = True
    val_sanity_check_full_epoch: bool = True
    val_interval: int = 1  # epoch-based validataion
    val_interval_step: int = -1
    val_interval_step_skip_hook: bool = False

    log_interval: int = 10
    img_log_interval: int = 2500
    save_interval: int = 2500

    # --- Training Loop Limits ---
    max_epochs: int = 500
    max_steps: int = 300000  # deprecated

    # --- Hardware & Dataloader ---
    seed: int = 1234
    batch_size: int = 4
    val_batch_size: int = 8
    grad_accumulation_steps: int = 2
    num_workers: int = 8
    fp16_run: bool = False
    drop_last: bool = True

    # --- Optimizer & Scheduler ---
    lr: float = 1e-4
    lr_decay_rate: float = 1.0
    betas: list[float] = field(default_factory=lambda: [0.8, 0.99])
    eps: float = 1e-9
    weight_decay: float = 1e-2
    grad_clip_thresh: float = 2.0

    # --- Experiment Tracking ---
    use_tensorboard: bool = True

    # --- Checkpoint Management ---
    keep_best_count: int = 3
    keep_last_count: int = 3
    monitor_loss: str = "recon"  # "recon" ...


@dataclass(frozen=True)
class LossConfigs:
    # Spec Decoder Loss Scheduling
    recon_init_weight: float = 15.0
    recon_final_weight: float = 15.0
    recon_start_step: int = 0
    recon_end_step: int = 100000

    # Alignment NLL Loss Scheduling
    crf_init_weight: float = 5.0
    crf_final_weight: float = 5.0
    crf_start_step: int = 0
    crf_end_step: int = 10000

    # Alignment Diagonal Loss Scheduling
    diag_init_weight: float = 10.0
    diag_final_weight: float = 0.0
    diag_start_step: int = 0
    diag_end_step: int = 30000


@dataclass(frozen=True)
class DataConfig:
    preprocess: PreprocessConfigs = field(default_factory=PreprocessConfigs)
    audio: AudioConfigs = field(default_factory=AudioConfigs)
    dataset: DatasetConfigs = field(default_factory=DatasetConfigs)
    train: TrainConfigs = field(default_factory=TrainConfigs)
    extra_exp: ExperimentConfigs = field(default_factory=ExperimentConfigs)
    loss: LossConfigs = field(default_factory=LossConfigs)
