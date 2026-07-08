from dataclasses import dataclass, field

from tts.config.preprocess.preprocess_config import PreprocessConfigs
from tts.tokenizer.load_tokenizer import load_tokenizer

from .data_config import (
    FASTSPEECH2_TOKENIZER_LEXION_PATH,
    INPUT_FEATURE_TYPE,
    N_FFT,
    N_MELS,
    TOKENIZER_TYPE,
    AudioConfigs,
)

SPEC_DIM = 192
TEXT_DIM = 192
SPK_COND_DIM = 192  # ECAPA-TDNN

USE_DELTA_FEAT = False
USE_DELTA_DELTA_FEAT = False
USE_OPTIONAL_SKIP_SEP = True  # z -> z+2 transition for <blank> or " " token

tokenizer = load_tokenizer(TOKENIZER_TYPE)


def spec_indim() -> int:
    if INPUT_FEATURE_TYPE == "mel":
        return N_MELS * (1 + int(USE_DELTA_FEAT) + int(USE_DELTA_DELTA_FEAT))
    elif INPUT_FEATURE_TYPE == "linspec":
        n_linspec = N_FFT // 2 + 1
        return n_linspec * (1 + int(USE_DELTA_FEAT) + int(USE_DELTA_DELTA_FEAT))
    else:
        raise ValueError()


def n_vocabs() -> int:
    return tokenizer.n_vocab * 2


@dataclass(frozen=True)
class TextEncoderConfigs:
    # common dimensions
    n_vocab: int = field(default_factory=n_vocabs)
    dim_out: int = TEXT_DIM
    dim_hidden: int = 256
    dim_cond: int = SPK_COND_DIM
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
    dilation_sizes: list[int] = field(default_factory=lambda: [1, 1, 2, 2, 3, 3])


@dataclass(frozen=True)
class SpecDecoderConfigs:
    in_channels: int = TEXT_DIM
    out_channels: int = N_MELS
    hidden_channels: int = 192
    cond_dim: int = SPK_COND_DIM
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 3, 3, 3])
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
    unary_network_type: str = "conv"  # "conv" "negative-l2"
    unary_support_type: str = "global"  # "global", "raw", "bernoulli"
    unary_temperature: float = 1.0
    unary_scale_init: float = -2.0

    # conv only configs
    conv_num_layers: int = 12
    conv_num_groups: int = 8
    conv_dim_hidden: int = 48
    conv_kernel_size: list[int] = field(default_factory=lambda: [3, 3])


@dataclass(frozen=True)
class NDAlignerConfigs:
    txt_enc: TextEncoderConfigs = field(default_factory=TextEncoderConfigs)
    spec_enc: SpecEncoderConfigs = field(default_factory=SpecEncoderConfigs)
    spec_dec: SpecDecoderConfigs = field(default_factory=SpecDecoderConfigs)
    aligner: CRFAlignerConfigs = field(default_factory=CRFAlignerConfigs)

    # for inference from wavforms
    preprocess: PreprocessConfigs = field(default_factory=PreprocessConfigs)
    audio: AudioConfigs = field(default_factory=AudioConfigs)

    use_delta_feat: bool = USE_DELTA_FEAT
    use_delta_delta_feat: bool = USE_DELTA_DELTA_FEAT
    use_optional_skip_sep: bool = USE_OPTIONAL_SKIP_SEP

    tokenizer_type: str = TOKENIZER_TYPE
    fastspeech2_tokenizer_lexion_path = FASTSPEECH2_TOKENIZER_LEXION_PATH
    separator_token_id: int = tokenizer.seperator_id
