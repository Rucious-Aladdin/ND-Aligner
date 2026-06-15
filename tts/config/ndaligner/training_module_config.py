from dataclasses import dataclass, field

from .model_config import NDAlignerConfigs


@dataclass(frozen=True)
class HiFiGANVocoderConfigs:
    config_path: str = "./checkpoints/hifigan/config.json"
    ckpt_path: str = "./checkpoints/hifigan/generator_v1"


@dataclass(frozen=True)
class NDAlignerTrainingModuleConfigs:
    nd_aligner: NDAlignerConfigs = field(default_factory=NDAlignerConfigs)
    vocoder: HiFiGANVocoderConfigs = field(default_factory=HiFiGANVocoderConfigs)
