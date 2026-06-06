import dataclasses

from ..config.stage1.model_config import MonotonicTTSConfigs

# from .modules.spec_decoder import SpecDecoder
from .modules.duration_predictor import StochasticDurationPredictor
from .modules.text_encoder import TextEncoder
from .modules.hifigan_vocoder import Generator
from .modules.crf_aligner import MonotonicCRFAligner

from .modules.spec_encoder import SpecEncoder
from .modules.conformer_decoder import ConformerSpecDecoder
from .modules.spk_encoder import SpeakerEncoder
from .monotonic_tts import MonotonicTTSSynthesizer


def init_monotonic_tts(
    config: MonotonicTTSConfigs | None = None,
    load_spec_encoder: bool = True,  # maybe False if inference only
    load_aligner: bool = True,  # maybe False if inference only
    load_vocoder: bool = True,
    load_speaker_encoder: bool = False,
    device: str = "cpu",
):

    if config is None:
        config = MonotonicTTSConfigs()

    if load_spec_encoder:
        spec_encoder = SpecEncoder(**dataclasses.asdict(config.spec_enc))
    else:
        spec_encoder = None

    if load_aligner:
        aligner = MonotonicCRFAligner(**dataclasses.asdict(config.aligner))
    else:
        aligner = None

    if load_vocoder:
        vocoder = Generator.from_config_path(**dataclasses.asdict(config.vocoder))
    else:
        vocoder = None

    if load_speaker_encoder:
        speaker_encoder = SpeakerEncoder(device=device)
    else:
        speaker_encoder = None

    text_encoder_align = TextEncoder(**dataclasses.asdict(config.txt_enc))
    text_encoder_gen = TextEncoder(**dataclasses.asdict(config.txt_enc))
    spec_decoder = ConformerSpecDecoder(**dataclasses.asdict(config.spec_dec))
    dur_predictor = StochasticDurationPredictor(**dataclasses.asdict(config.dur_predictor))

    model = MonotonicTTSSynthesizer(
        text_encoder_align=text_encoder_align,
        text_encoder_gen=text_encoder_gen,
        dur_predictor=dur_predictor,
        spec_encoder=spec_encoder,
        aligner=aligner,
        spec_decoder=spec_decoder,
        vocoder=vocoder,
        speaker_encoder=speaker_encoder,
    )

    return model
