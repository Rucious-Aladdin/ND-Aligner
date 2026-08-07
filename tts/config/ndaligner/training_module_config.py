from dataclasses import dataclass, field

from .model_config import NDAlignerConfigs


@dataclass(frozen=True)
class NDAlignerTrainingModuleConfigs:
    nd_aligner: NDAlignerConfigs = field(default_factory=NDAlignerConfigs)
