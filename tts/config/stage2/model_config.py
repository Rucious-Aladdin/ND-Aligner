from dataclasses import dataclass, field

from ..ndaligner.model_config import N_MELS, SPK_COND_DIM, TEXT_DIM, NDAlignerConfigs


@dataclass(frozen=True)
class DiffusionModelConfigs:
    n_mels: int = N_MELS
    sigma_data: float = 2.0990
    mu_data: float = -4.9307
    p_mean: float = -1.2
    p_std: float = 1.2


@dataclass(frozen=True)
class ConditionAdapterConfigs:
    in_dim: int = TEXT_DIM
    spk_dim: int = SPK_COND_DIM
    progress_hidden_dim: int = 128
    smoothing_hidden_dim: int = 128
    smoothing_kernel_size: int = 3
    smoothing_num_layers: int = 6
    apply_local_text_progress: bool = True
    apply_global_text_progress: bool = True
    apply_spec_progress: bool = True


@dataclass(frozen=True)
class ConformerDenoiserConfigs:
    n_mels: int = N_MELS
    conformer_cond_dim: int = TEXT_DIM
    conformer_hidden_dim: int = 256
    conformer_num_layers: int = 6
    conformer_num_attention_heads: int = 4
    conformer_feed_forward_expansion_factor: int = 4
    conformer_conv_expansion_factor: int = 2
    conformer_input_dropout_p: float = 0.1
    conformer_feed_forward_dropout_p: float = 0.1
    conformer_attention_dropout_p: float = 0.1
    conformer_conv_dropout_p: float = 0.1
    conformer_conv_kernel_size: int = 7
    conformer_half_step_residual: bool = True
    conformer_attn_window_size: int = 3


@dataclass(frozen=True)
class DiffusionTTSConfigs:
    # Backbone Stage 1 configuration
    s1_config: NDAlignerConfigs = field(default_factory=NDAlignerConfigs)

    # Stage 2 (Diffusion) specific configuration
    model: DiffusionModelConfigs = field(default_factory=DiffusionModelConfigs)
    cond_adapter: ConditionAdapterConfigs = field(default_factory=ConditionAdapterConfigs)
    denoiser: ConformerDenoiserConfigs = field(default_factory=ConformerDenoiserConfigs)
