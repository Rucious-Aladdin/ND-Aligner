import dataclasses

from ..config.stage2.model_config import DiffusionTTSConfigs
from .diffusion.karras_diffusion import KarrasDiffusionModel
from .diffusion.submodules.cond_adapter import ConditionAdapter
from .diffusion.conformer_denoiser import ConformerDenoiser, ConformerDenoiserNetwork
from .diffusion_tts import KarrasTTSSynthesizer
from .init_monotonic_tts import init_monotonic_tts


def init_diffusion_model(config: DiffusionTTSConfigs) -> KarrasDiffusionModel:
    cond_adapter = ConditionAdapter(**dataclasses.asdict(config.cond_adapter))
    network = ConformerDenoiserNetwork(**dataclasses.asdict(config.denoiser))
    denoiser = ConformerDenoiser(network=network, cond_adapter=cond_adapter)

    diffusion_model = KarrasDiffusionModel(
        denoiser=denoiser,
        sigma_data=config.model.sigma_data,
        mu_data=config.model.mu_data,
        p_mean=config.model.p_mean,
        p_std=config.model.p_std,
    )
    return diffusion_model


def init_diffusion_tts(
    config: DiffusionTTSConfigs | None = None,
    device: str = "cpu",
) -> KarrasTTSSynthesizer:
    """
    Initializes the integrated KarrasTTSSynthesizer (Stage 1 + Stage 2).
    """
    if config is None:
        config = DiffusionTTSConfigs()

    # 1. Initialize Stage 1 Backbone
    # Note: For Stage 2 training, we usually need the full backbone.
    syn_backbone = init_monotonic_tts(
        config=config.s1_config,
        load_vocoder=True,
        load_speaker_encoder=True,
        device=device,
    )

    # 2. Extract dimensions from Stage 1 for Stage 2 consistency
    diffusion_model = init_diffusion_model(config)

    # 5. Initialize Integrated Model
    model = KarrasTTSSynthesizer(
        syn=syn_backbone,
        diffusion_model=diffusion_model,
    )
    return model.to(device)
