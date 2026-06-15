from dataclasses import dataclass, field

from .data_config import N_MELS

SPEC_DIM = 256
TEXT_DIM = 192
SPK_COND_DIM = 192  # ECAPA-TDNN

USE_DELTA_MEL = True
USE_DELTA_DELTA_MEL = False


def spec_indim() -> int:
    return N_MELS * (1 + int(USE_DELTA_MEL) + int(USE_DELTA_DELTA_MEL))


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
    in_dim: int = field(default_factory=spec_indim)
    hidden_dim: int = SPEC_DIM
    cond_dim: int = SPK_COND_DIM

    # architectures
    kernel_size: int = 3
    dropout_p: float = 0.15
    dilation_sizes: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])


@dataclass(frozen=True)
class SpecDecoderConfigs:
    in_channels: int = TEXT_DIM
    out_channels: int = N_MELS
    hidden_channels: int = 256
    cond_dim: int = SPK_COND_DIM
    kernel_sizes: list[int] = field(default_factory=lambda: [1, 1, 1])
    dilation_base: int = 1
    dropout: float = 0.15


@dataclass(frozen=True)
class CRFAlignerConfigs:
    dim_spec: int = SPEC_DIM
    dim_text: int = TEXT_DIM
    dim_cond: int = SPK_COND_DIM
    dim_unary_latent: int = 48
    cond_channels: int = 32

    # unary network configs
    unary_network_type: str = "conv"  # "unet" "conv" "negative-l2"
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
class NDAlignerConfigs:
    txt_enc: TextEncoderConfigs = field(default_factory=TextEncoderConfigs)
    spec_enc: SpecEncoderConfigs = field(default_factory=SpecEncoderConfigs)
    spec_dec: SpecDecoderConfigs = field(default_factory=SpecDecoderConfigs)
    aligner: CRFAlignerConfigs = field(default_factory=CRFAlignerConfigs)

    use_delta_mel: bool = USE_DELTA_MEL
    use_delta_delta_mel: bool = USE_DELTA_DELTA_MEL
