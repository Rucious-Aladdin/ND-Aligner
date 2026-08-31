import os
import os.path
from dataclasses import dataclass, field

from nd_aligner.utils.global_constant import DATA_ROOT_DIR, SAMPLE_RATE

# Global root for datasets (base directory for preprocessed data)

SPK_ENCODER_TYPE = "resemblyzer"


def spk_dim() -> int:
    if SPK_ENCODER_TYPE == "resemblyzer":
        return 256
    else:
        raise ValueError()


@dataclass(frozen=True)
class PreprocessConfigs:
    data_root_dir: str = os.path.join(DATA_ROOT_DIR, "LibriSpeech")
    preprocessed_dir: str = os.path.join(DATA_ROOT_DIR, "LibriSpeech-preprocessed-trimmed-16k")
    spk_encoder_type: str = SPK_ENCODER_TYPE
    spk_embedding_dim: int = field(default_factory=spk_dim)

    resample_sr: int = SAMPLE_RATE

    # ------------------------------------------------------------------
    # Silence / VAD
    # ------------------------------------------------------------------
    silence_trim: bool = True
    denoise_before_vad: bool = False
    lpf_before_vad: bool = False
    lpf_cutoff_freq: float = 4000.0
    silence_margin_sec: float = 0.0

    # ------------------------------------------------------------------
    # Amplitude normalization
    # ------------------------------------------------------------------
    peak_normalize: bool = True
    peak_target: float = 0.6
