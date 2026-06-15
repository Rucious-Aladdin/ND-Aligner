import dataclasses

from ..config.ndaligner.model_config import NDAlignerConfigs
from ..config.ndaligner.training_module_config import NDAlignerTrainingModuleConfigs
from .modules.crf_aligner import MonotonicCRFAligner
from .modules.hifigan_vocoder import Generator
from .modules.spec_decoder import SpecDecoder
from .modules.spec_encoder import SpecEncoder
from .modules.spk_encoder import ECAPASpeakerEncoder
from .modules.text_encoder import TextEncoder
from .ndaligner import NDAligner, NDAlignerTrainingModule


def init_nd_aligner(
    config: NDAlignerConfigs | None = None,
    device: str = "cpu",
):

    if config is None:
        config = NDAlignerConfigs()

    text_encoder = TextEncoder(**dataclasses.asdict(config.txt_enc))
    spec_encoder = SpecEncoder(**dataclasses.asdict(config.spec_enc))
    aligner = MonotonicCRFAligner(**dataclasses.asdict(config.aligner))
    spec_decoder = SpecDecoder(**dataclasses.asdict(config.spec_dec))

    model = NDAligner(
        text_encoder=text_encoder,
        spec_encoder=spec_encoder,
        crf_aligner=aligner,
        spec_decoder=spec_decoder,
        use_delta_mel=config.use_delta_mel,
        use_delta_delta_mel=config.use_delta_delta_mel,
    )

    return model.to(device)


def init_nd_aligner_training_module(
    config: NDAlignerTrainingModuleConfigs | None = None,
    load_vocoder: bool = False,
    load_speaker_encoder: bool = False,
    device: str = "cpu",
):

    if config is None:
        config = NDAlignerTrainingModuleConfigs()

    nd_aligner = init_nd_aligner(
        config=config.nd_aligner,
        device=device,
    )

    spk_enc = vocoder = None

    if load_speaker_encoder:
        spk_enc = ECAPASpeakerEncoder(device=device)

    if load_vocoder:
        vocoder = Generator.from_config_path(
            config_path=config.vocoder.config_path,
            ckpt_path=config.vocoder.ckpt_path,
        ).to(device)

    return NDAlignerTrainingModule(
        nd_aligner=nd_aligner,
        vocoder=vocoder,
        speaker_encoder=spk_enc,
    )
