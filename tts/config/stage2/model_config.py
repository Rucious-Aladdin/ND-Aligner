from dataclasses import dataclass, field
from functools import partial

from tts.models.utils.fix_len_compatibility import fix_len_compatibility

from ..stage1.model_config import MonotonicTTSConfigs


@dataclass(frozen=True)
class UnetConfigs:
    dim: int = 64
    dim_mults: list[int] = field(default_factory=lambda: [1, 2, 4])
    groups: int = 8


@dataclass(frozen=True)
class ScoreEstimatorConfigs:
    sigma_data: float = 2.0990
    mu_data: float = -4.9307
    p_mean: float = -1.2
    p_std: float = 1.2


@dataclass(frozen=True)
class DiffusionTTSConfigs:
    # Backbone Stage 1 configuration
    s1_config: MonotonicTTSConfigs = field(default_factory=MonotonicTTSConfigs)

    # Stage 2 (Diffusion) specific configuration
    unet: UnetConfigs = field(default_factory=UnetConfigs)
    estimator: ScoreEstimatorConfigs = field(default_factory=ScoreEstimatorConfigs)

    # Unet Downsampling aware (seq_len % (2^num_downsample) == 0)
    num_unet_downsample: int = 2
    unet_out_size: int = field(default_factory=partial(fix_len_compatibility, 2 * 22050 // 256, 2))
