from dataclasses import dataclass, field

from nd_aligner.config.preprocess.preprocess_config import PreprocessConfigs, spk_dim
from nd_aligner.tokenizer.load_tokenizer import load_tokenizer

from .data_config import (
    INPUT_FEATURE_TYPE,
    N_FFT,
    N_MELS,
    TOKENIZER_TYPE,
    AudioConfigs,
)

SPEC_DIM = 128
TEXT_DIM = 128

SHARED_HIDDEN_DIM = 192
DEC_HIDDEN_DIM = 192

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
    return tokenizer.n_vocab * 2  # margin of n_vocab..


@dataclass(frozen=True)
class TextEncoderConfigs:
    n_vocab: int = field(default_factory=n_vocabs)
    dim_out: int = TEXT_DIM
    dim_hidden: int = SHARED_HIDDEN_DIM
    dim_cond: int = field(default_factory=spk_dim)
    kernel_sizes: list[int] = field(default_factory=lambda: [1])


@dataclass(frozen=True)
class SpecEncoderConfigs:
    in_dim: int = field(default_factory=spec_indim)
    out_dim: int = SPEC_DIM
    hidden_dim: int = 192
    cond_dim: int = field(default_factory=spk_dim)

    kernel_size: int = 3
    dropout_p: float = 0.15
    dilation_sizes: list[int] = field(default_factory=lambda: [1, 1, 1])


@dataclass(frozen=True)
class SpecDecoderConfigs:
    decoder_type: str = "coupling"  # conv1d, "coupling"

    in_channels: int = TEXT_DIM
    out_channels: int = N_MELS
    hidden_channels: int = DEC_HIDDEN_DIM
    cond_dim: int = field(default_factory=spk_dim)
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 3, 3, 3, 3, 3, 3, 3])
    dilation_base: int = 1
    dropout: float = 0.15

    coupling_cond_proj_dim: int = 128
    coupling_num_refine_steps: int = 6
    coupling_step_emb_dim: int = 64
    coupling_loss_decay_factor: float = 1.0  # (w_st in TCD)
    coupling_kernel_size: int = 3
    coupling_normalize_loss_weights: bool = True
    coupling_stage_loss_mode: str = "geometric"


@dataclass(frozen=True)
class CRFAlignerConfigs:
    dim_spec: int = SPEC_DIM
    dim_text: int = TEXT_DIM
    dim_cond: int = field(default_factory=spk_dim)
    dim_unary_latent: int = 128
    cond_channels: int = 32

    # unary network configs (estimate node potentials of CRF!)
    unary_network_type: str = "conv"  # "conv" "l2"
    unary_support_type: str = "global"  # "global", "raw", "bernoulli"
    unary_temperature: float = 1.0
    unary_scale_init: float = -2.0

    # conv only configs
    conv_num_layers: int = 6
    conv_num_groups: int = 8
    conv_dim_hidden: int = 64
    conv_kernel_size: list[int] = field(default_factory=lambda: [3, 3])


@dataclass(frozen=True)
class InputMakerConfigs:
    zero_non_speech_region: bool = True
    trim_non_speech_region: bool = True
    reduce_noise: bool = False
    suppress_impulsive_peak: bool = False


@dataclass(frozen=True)
class NDAlignerConfigs:
    txt_enc: TextEncoderConfigs = field(default_factory=TextEncoderConfigs)
    spec_enc: SpecEncoderConfigs = field(default_factory=SpecEncoderConfigs)
    spec_dec: SpecDecoderConfigs = field(default_factory=SpecDecoderConfigs)
    aligner: CRFAlignerConfigs = field(default_factory=CRFAlignerConfigs)
    input_maker: InputMakerConfigs = field(default_factory=InputMakerConfigs)

    # for inference from wavforms
    preprocess: PreprocessConfigs = field(default_factory=PreprocessConfigs)
    audio: AudioConfigs = field(default_factory=AudioConfigs)

    use_delta_feat: bool = USE_DELTA_FEAT
    use_delta_delta_feat: bool = USE_DELTA_DELTA_FEAT
    use_optional_skip_sep: bool = USE_OPTIONAL_SKIP_SEP

    # AH_text -> A*H_text using viterbi alignment. it degrades performances. not recommend to use this.
    viterbi_ste_training: bool = False
    tokenizer_type: str = TOKENIZER_TYPE
    separator_token_id: int = tokenizer.seperator_id  # the word seperator. (optional silence state)
