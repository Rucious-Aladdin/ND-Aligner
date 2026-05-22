from dataclasses import dataclass, field
from tts.tokenizer.letters import SYMBOL_DICTS

N_MELS = 80
SPEC_DIM = 256

TEXT_DIM = 256
SPK_COND_DIM = 192  # ECAPA-TDNN

NUM_TEXT_TOKENS = len(SYMBOL_DICTS) + 1


@dataclass(frozen=True)
class TextEmbedderConfigs:
    # common dimensions
    n_vocab: int = 256
    dim_out: int = TEXT_DIM


@dataclass(frozen=True)
class SpecEncoderConfigs:
    # dimensions
    in_dim: int = N_MELS
    hidden_dim: int = SPEC_DIM
    cond_dim: int = SPK_COND_DIM

    # architectures
    kernel_size: int = 1
    dropout_p: float = 0.1
    dilation_sizes: list[int] = field(default_factory=lambda: [1, 1, 1, 1])


@dataclass(frozen=True)
class SpecDecoderConfigs:
    in_channels: int = TEXT_DIM
    out_channels: int = N_MELS
    hidden_channels: int = 256
    cond_dim: int = SPK_COND_DIM
    dilation_base: int = 1
    dropout: float = 0.1
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 3, 3, 3])


@dataclass(frozen=True)
class DurationPredictorConfigs:
    # dimensions
    in_channels: int = TEXT_DIM
    gin_channels: int = SPK_COND_DIM
    filter_channels: int = 256

    # architectures
    kernel_size: int = 3
    p_dropout: float = 0.1
    n_flows: int = 6


@dataclass(frozen=True)
class CRFAlignerConfigs:
    # dimensions
    dim_spec: int = SPEC_DIM
    dim_text: int = TEXT_DIM
    dim_cond: int = SPK_COND_DIM
    dim_unary_latent: int = 48
    cond_channels: int = 32
    unet_base_dim: int = 16
    unet_groups: int = 8

    unary_support_type: str = "local"  # "local" or "global"
    unary_radius: int = 10
    unary_temperature: float = 1.0
    unary_scale_init: float = -2.0


@dataclass(frozen=True)
class HifiGANVocoderConfigs:
    config_path: str = "./checkpoints/hifigan/config.json"
    ckpt_path: str = "./checkpoints/hifigan/generator_v1"


@dataclass(frozen=True)
class MonotonicTTSConfigs:
    txt_enc: TextEmbedderConfigs = field(default_factory=TextEmbedderConfigs)
    spec_enc: SpecEncoderConfigs = field(default_factory=SpecEncoderConfigs)
    spec_dec: SpecDecoderConfigs = field(default_factory=SpecDecoderConfigs)
    dur_predictor: DurationPredictorConfigs = field(default_factory=DurationPredictorConfigs)
    aligner: CRFAlignerConfigs = field(default_factory=CRFAlignerConfigs)
    vocoder: HifiGANVocoderConfigs = field(default_factory=HifiGANVocoderConfigs)
