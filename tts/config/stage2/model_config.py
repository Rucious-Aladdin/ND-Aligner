from dataclasses import dataclass, field
from functools import partial

from tts.models.diffusion import cond_adapter
from tts.models.utils.fix_len_compatibility import fix_len_compatibility

from ..stage1.model_config import MonotonicTTSConfigs, TEXT_DIM


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
class ConditionAdapterConfigs:
    in_dim: int = TEXT_DIM
    progress_hidden_dim: int = 128
    smoothing_hidden_dim: int = 128
    smoothing_kernel_size: int = 3
    smoothing_num_layers: int = 8
    apply_smoothing: bool = True
    apply_local_text_progress: bool = True
    apply_global_text_progress: bool = True
    apply_spec_progress: bool = True


@dataclass(frozen=True)
class DiffusionTTSConfigs:
    # Backbone Stage 1 configuration
    s1_config: MonotonicTTSConfigs = field(default_factory=MonotonicTTSConfigs)

    # Stage 2 (Diffusion) specific configuration
    unet: UnetConfigs = field(default_factory=UnetConfigs)
    estimator: ScoreEstimatorConfigs = field(default_factory=ScoreEstimatorConfigs)
    cond_adapter: ConditionAdapterConfigs = field(default_factory=ConditionAdapterConfigs)

    # Unet Downsampling aware (seq_len % (2^num_downsample) == 0)
    num_unet_downsample: int = 2
    unet_out_size: int = field(default_factory=partial(fix_len_compatibility, 2 * 22050 // 256, 2))

    apply_local_text_progress: bool = True
    apply_global_text_progress: bool = True
    apply_spec_progress: bool = True
    progress_hidden_dim: int = 128
