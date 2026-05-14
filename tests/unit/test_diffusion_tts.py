import pytest
import torch
from tts.config.stage1.model_config import MonotonicTTSConfigs
from tts.config.stage2.model_config import DiffusionTTSConfigs
from tts.models.init_monotonic_tts import init_monotonic_tts
from tts.models.diffusion.score_estimator import KarrasScoreEstimator
from tts.models.diffusion.denoiser_net import MelDenoiserNetwork
from tts.models.diffusion_tts import KarrasTTSSynthesizer

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def diffusion_tts(device):
    # Stage 2 Config
    diff_config = DiffusionTTSConfigs()
    
    # Stage 1 Model
    syn_model = init_monotonic_tts(
        config=diff_config.s1_config,
        load_spec_encoder=True,
        load_aligner=True,
    ).to(device)
    
    # Stage 2 Model (Denoiser & Estimator)
    denoiser = MelDenoiserNetwork(
        n_mels=diff_config.s1_config.dec.num_mels,
        cond_dim=diff_config.s1_config.aligner.dim_cond,
        dim=diff_config.unet.dim,
        dim_mults=diff_config.unet.dim_mults,
        spk_emb_dim=diff_config.s1_config.aligner.dim_cond,
        cfg_dropout_prob=diff_config.cfg_dropout_prob,
    )
    estimator = KarrasScoreEstimator(
        denoiser=denoiser,
        sigma_data=diff_config.estimator.sigma_data,
        mu_data=diff_config.estimator.mu_data,
        p_mean=diff_config.estimator.p_mean,
        p_std=diff_config.estimator.p_std,
    )
    
    model = KarrasTTSSynthesizer(
        syn=syn_model,
        estimator=estimator,
    ).to(device)
    
    return model

def test_diffusion_tts_forward_non_multiple_of_4(diffusion_tts, device):
    """
    Test if the model can handle mel lengths that are not multiples of 4.
    """
    diffusion_tts.train()
    
    diff_config = DiffusionTTSConfigs()
    B = 2
    # Non-multiple of 4 lengths
    y_lengths = torch.tensor([11, 13], dtype=torch.long, device=device)
    T_mel_max = y_lengths.max().item()
    
    T_text_max = 20
    x_lengths = torch.tensor([15, 18], dtype=torch.long, device=device)
    
    n_vocab = diff_config.s1_config.txt_enc.n_vocab
    n_mels = diff_config.s1_config.dec.num_mels
    spk_emb_dim = diff_config.s1_config.aligner.dim_cond
    
    x = torch.randint(0, n_vocab, (B, T_text_max), device=device)
    y = torch.randn(B, n_mels, T_mel_max, device=device)
    cond = torch.randn(B, spk_emb_dim, device=device)
    
    # Run forward
    out = diffusion_tts(
        x=x,
        x_lengths=x_lengths,
        y=y,
        y_lengths=y_lengths,
        cond=cond,
    )
    assert out.mel_hat.shape == y.shape

def test_diffusion_tts_inference_non_multiple_of_4(diffusion_tts, device):
    """
    Test inference with arbitrary text lengths which results in arbitrary mel lengths.
    """
    diffusion_tts.eval()
    
    diff_config = DiffusionTTSConfigs()
    B = 1
    T_text = 15
    x_lengths = torch.tensor([T_text], dtype=torch.long, device=device)
    
    n_vocab = diff_config.s1_config.txt_enc.n_vocab
    spk_emb_dim = diff_config.s1_config.aligner.dim_cond
    
    x = torch.randint(0, n_vocab, (B, T_text), device=device)
    cond = torch.randn(B, spk_emb_dim, device=device)
    
    # Run inference
    with torch.no_grad():
        out = diffusion_tts.inference(
            x=x,
            x_lengths=x_lengths,
            cond=cond,
            n_steps=5, # Small steps for fast test
        )
    
    assert out.mel_hat.ndim == 3
    assert out.wav_hat.ndim == 2
