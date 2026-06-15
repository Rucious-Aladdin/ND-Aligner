from dataclasses import dataclass, field

N_MELS = 80
SPEC_DIM = 192

TEXT_DIM = 192
SPK_COND_DIM = 192  # ECAPA-TDNN


@dataclass(frozen=True)
class TextEncoderConfigs:
    # common dimensions
    n_vocab: int = 256
    dim_out: int = TEXT_DIM
    dim_hidden: int = 256
    kernel_sizes: list[int] = field(default_factory=lambda: [1])


@dataclass(frozen=True)
class SpecEncoderConfigs:
    # dimensions
    in_dim: int = N_MELS
    hidden_dim: int = SPEC_DIM
    cond_dim: int = SPK_COND_DIM

    # architectures
    kernel_size: int = 3
    dropout_p: float = 0.1
    dilation_sizes: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])


@dataclass(frozen=True)
class SpecDecoderConfigs:
    conformer_out_dim: int = N_MELS
    conformer_input_dim: int = TEXT_DIM
    conformer_cond_dim: int = SPK_COND_DIM
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
class DurationPredictorConfigs:
    in_channels: int = TEXT_DIM
    gin_channels: int = SPK_COND_DIM
    filter_channels: int = 256

    kernel_size: int = 3
    p_dropout: float = 0.1
    n_flows: int = 6
    apply_positional_encoding: bool = True


@dataclass(frozen=True)
class CRFAlignerConfigs:
    dim_spec: int = SPEC_DIM
    dim_text: int = TEXT_DIM
    dim_cond: int = SPK_COND_DIM
    dim_unary_latent: int = 48
    cond_channels: int = 32

    # unary network configs
    unary_network_type: str = "conv"  # "unet" "conv"
    unary_base_dim: int = 16
    unary_groups: int = 8

    # conv only configs
    conv_num_layers: int = 4
    conv_kernel_size: list[int] = field(default_factory=lambda: [3, 3])

    # normalization and support configs
    unary_support_type: str = "local"  # "local" or "global"
    unary_radius: int = 10
    unary_temperature: float = 1.0
    unary_scale_init: float = -2.0
    unary_apply_progress_feature: bool = False


@dataclass(frozen=True)
class HifiGANVocoderConfigs:
    config_path: str = "./checkpoints/hifigan/config.json"
    ckpt_path: str = "./checkpoints/hifigan/generator_v1"


@dataclass(frozen=True)
class MonotonicTTSConfigs:
    txt_enc: TextEncoderConfigs = field(default_factory=TextEncoderConfigs)
    spec_enc: SpecEncoderConfigs = field(default_factory=SpecEncoderConfigs)
    spec_dec: SpecDecoderConfigs = field(default_factory=SpecDecoderConfigs)
    dur_predictor: DurationPredictorConfigs = field(default_factory=DurationPredictorConfigs)
    aligner: CRFAlignerConfigs = field(default_factory=CRFAlignerConfigs)
    vocoder: HifiGANVocoderConfigs = field(default_factory=HifiGANVocoderConfigs)
