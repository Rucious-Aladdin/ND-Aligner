import torch

from tts.benchmark.timit.benchmarker import TIMITBenchMarker
from tts.benchmark.timit.word_mapper import HYP_IGNORE_SYMBOLS
from tts.config.ndaligner.data_config import DataConfig
from tts.config.ndaligner.training_module_config import NDAlignerTrainingModuleConfigs
from tts.config.utils.io import load_config
from tts.models.init_ndaligner import init_nd_aligner_training_module
from tts.models.modules.hifigan_vocoder import Generator
from tts.models.modules.spk_encoder import ECAPASpeakerEncoder
from tts.tokenizer.text_tokenizer import TextTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

aligner_data_cfg_path = "./runs/nd_aligner_vctk_20260611-113756/data_config.json"
aligner_training_module_cfg_path = "./runs/nd_aligner_vctk_20260611-113756/model_config.json"
aligner_training_module_ckpt_path = "./runs/nd_aligner_vctk_20260611-113756/checkpoints_timit_bae/best_step_timit_bae_step_13500_epoch_5.pth"

vocoder_config_path = "./checkpoints/hifigan/config.json"
vocoder_ckpt_path = "./checkpoints/hifigan/generator_v1"


TIMIT_ROOT = "/shared/data_zfs/blue2959/TIMIT/TEST"


if __name__ == "__main__":
    data_config = load_config(aligner_data_cfg_path, DataConfig)
    model_config = load_config(
        aligner_training_module_cfg_path,
        NDAlignerTrainingModuleConfigs,
    )

    aligner_training_module = init_nd_aligner_training_module(
        config=model_config,
    ).to(device=device)
    aligner_training_module.load_checkpoint(
        ckpt_path=aligner_training_module_ckpt_path,
        device=device,
    )
    aligner = aligner_training_module.nd_aligner.eval()

    txt_tokenizer = TextTokenizer()
    spk_encoder = ECAPASpeakerEncoder(device=device)

    timit_benchmarker = TIMITBenchMarker(
        root_dir=TIMIT_ROOT,
        ref_audio_sr=16_000,
        hyp_audio_sr=22_050,
        hyp_hop_length=256,
        tokenizer=txt_tokenizer,
        audio_config=data_config.audio,
        hyp_ignore_symbols=HYP_IGNORE_SYMBOLS,
        max_ref_words_per_hyp_word=5,
        spk_cond_dim=192,
        rms_normalize=True,
        rms_target=0.15,
    )

    vocoder = Generator.from_config_path(
        config_path=vocoder_config_path,
        ckpt_path=vocoder_ckpt_path,
    )

    metrics = timit_benchmarker(
        aligner=aligner,
        speaker_encoder=spk_encoder,
        vocoder=vocoder,
        max_test_samples=100,
    )

    print(metrics.word_boundary_error)
    print(metrics.p_word_100ms)
    print(metrics.p_word_50ms)
    print(metrics.p_word_25ms)
    print(metrics.p_word_10ms)
    print(metrics.mcd_dtw)
    print(metrics.posterior_entropy)
