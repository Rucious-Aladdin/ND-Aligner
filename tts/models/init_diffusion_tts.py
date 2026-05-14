from ..config.stage2.model_config import DiffusionTTSConfigs
from .diffusion.denoiser_net import MelDenoiserNetwork
from .diffusion.score_estimator import KarrasScoreEstimator
from .diffusion_tts import KarrasTTSSynthesizer
from .init_monotonic_tts import init_monotonic_tts


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
    n_mels = config.s1_config.spec_dec.out_channels
    # conditioning dimension comes from Stage 1 text encoder's output
    cond_dim = config.s1_config.txt_enc.dim_out
    # speaker embedding dimension
    spk_emb_dim = config.s1_config.aligner.dim_cond

    # 3. Initialize Stage 2 Denoiser (U-Net wrapper)
    denoiser = MelDenoiserNetwork(
        n_mels=n_mels,
        pho_cond_dim=cond_dim,
        dim=config.unet.dim,
        dim_mults=tuple(config.unet.dim_mults),
        groups=config.unet.groups,
        spk_emb_dim=spk_emb_dim,
    )

    # 4. Initialize Score Estimator (EDM Preconditioning)
    estimator = KarrasScoreEstimator(
        denoiser=denoiser,
        sigma_data=config.estimator.sigma_data,
        mu_data=config.estimator.mu_data,
        p_mean=config.estimator.p_mean,
        p_std=config.estimator.p_std,
    )

    # 5. Initialize Integrated Model
    model = KarrasTTSSynthesizer(
        syn=syn_backbone,
        estimator=estimator,
        num_unet_downsample=config.num_unet_downsample,
        unet_out_size=config.unet_out_size,
    )

    return model.to(device)
