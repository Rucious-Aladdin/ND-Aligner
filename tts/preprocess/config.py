# use 22050 hifi-gan for compatibility
import os
import os.path
from dataclasses import dataclass, field

from tts.global_constant import DATA_PARENT_DIR, SAMPLE_RATE

# Global root for datasets (base directory for preprocessed data)


@dataclass(frozen=True)
class PreprocessConfig:
    data_root_dir: str = os.path.join(DATA_PARENT_DIR, "VCTK")
    preprocessed_dir: str = os.path.join(DATA_PARENT_DIR, "VCTK-preprocessed")
    spk_embedding_dim: int = 192  # ECAPA-TDNN output dimension

    resample_sr: int = SAMPLE_RATE

    silence_trim: bool = True
    denoise_before_vad: bool = True
    lpf_before_vad: bool = True
    lpf_cutoff_freq: float = 4000.0

    silence_margin_sec: float = 0.0

    rms_normalize: bool = True
    rms_target: float = 0.1

    text_preprocess_type: str = "ipa"  # "raw"
