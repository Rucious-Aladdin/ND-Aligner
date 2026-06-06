from __future__ import annotations

import numpy as np
import torch
import librosa
import pysptk

_LOGDB_CONST = 10.0 / np.log(10.0) * np.sqrt(2.0)


def _trim_wav_with_mask(
    wav: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    """
    Args:
        wav:  (T,)
        mask: (T,), 1 for valid samples
    Returns:
        wav_np: (T_valid,)
    """
    if mask.dim() != 1:
        mask = mask.view(-1)

    if wav.dim() != 1:
        wav = wav.view(-1)

    valid_len = int(mask.to(dtype=torch.long).sum().item())
    if valid_len <= 0:
        raise ValueError("wav mask has no valid samples.")

    wav = wav[:valid_len]
    wav_np = wav.detach().cpu().float().numpy().astype(np.float64)

    return wav_np


def _wav_to_mgc(
    wav: np.ndarray,
    *,
    frame_length: int = 1024,
    hop_length: int = 256,
    order: int = 25,
    alpha: float = 0.45,  # 22050Hz 에서는 통상 0.45를 씁니다 (16kHz가 0.41)
) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float64)
    if wav.ndim != 1:
        wav = wav.reshape(-1)

    power_spectrogram = np.abs(librosa.stft(wav, n_fft=frame_length, hop_length=hop_length)) ** 2

    mgc = pysptk.sp2mc(power_spectrogram.T, order=order, alpha=alpha)

    return mgc


def _compute_mcd_dtw_single(
    ref_wav: np.ndarray,
    hyp_wav: np.ndarray,
    *,
    exclude_c0: bool = True,
    frame_length: int = 1024,
    hop_length: int = 256,
    order: int = 25,
    alpha: float = 0.45,
) -> float:
    ref_mgc = _wav_to_mgc(
        ref_wav, frame_length=frame_length, hop_length=hop_length, order=order, alpha=alpha
    )
    hyp_mgc = _wav_to_mgc(
        hyp_wav, frame_length=frame_length, hop_length=hop_length, order=order, alpha=alpha
    )

    if exclude_c0:
        ref_mgc = ref_mgc[:, 1:]
        hyp_mgc = hyp_mgc[:, 1:]

    _, wp = librosa.sequence.dtw(X=ref_mgc.T, Y=hyp_mgc.T, metric="euclidean")

    ref_idx = wp[:, 0]
    hyp_idx = wp[:, 1]

    ref_aligned = ref_mgc[ref_idx]
    hyp_aligned = hyp_mgc[hyp_idx]

    diff = ref_aligned - hyp_aligned
    local_dist = np.sqrt((diff * diff).sum(axis=-1))  # (T_path,)

    mcd = _LOGDB_CONST * float(local_dist.mean())
    return mcd


def compute_mcd_dtw(
    ref_wavs: torch.Tensor,
    ref_wav_mask: torch.Tensor,
    hyp_wavs: torch.Tensor,
    hyp_wav_mask: torch.Tensor,
    *,
    exclude_c0: bool = True,
) -> torch.Tensor:
    """
    Compute per-sample MCD-DTW between reference and hypothesis waveforms.

    Assumptions:
        - sampling rate is 22050 Hz
        - wav tensors are already mono waveform tensors
        - masks are sample-level masks

    Args:
        ref_wavs:     (B, T_ref)
        ref_wav_mask: (B, T_ref), 1 for valid samples
        hyp_wavs:     (B, T_hyp)
        hyp_wav_mask: (B, T_hyp), 1 for valid samples
        exclude_c0:
            If True, excludes the 0-th mel-cepstral coefficient.

    Returns:
        mcd_dtw: (B,)
    """
    if ref_wavs.dim() != 2:
        raise ValueError(f"ref_wavs must have shape (B, T_ref), got {tuple(ref_wavs.shape)}.")
    if hyp_wavs.dim() != 2:
        raise ValueError(f"hyp_wavs must have shape (B, T_hyp), got {tuple(hyp_wavs.shape)}.")

    if ref_wav_mask.dim() == 3:
        ref_wav_mask = ref_wav_mask.squeeze(1)
    if hyp_wav_mask.dim() == 3:
        hyp_wav_mask = hyp_wav_mask.squeeze(1)

    if ref_wav_mask.shape != ref_wavs.shape:
        raise ValueError(
            f"ref_wav_mask shape {tuple(ref_wav_mask.shape)} "
            + f"does not match ref_wavs shape {tuple(ref_wavs.shape)}."
        )

    if hyp_wav_mask.shape != hyp_wavs.shape:
        raise ValueError(
            f"hyp_wav_mask shape {tuple(hyp_wav_mask.shape)} "
            + f"does not match hyp_wavs shape {tuple(hyp_wavs.shape)}."
        )

    B = ref_wavs.size(0)
    scores: list[float] = []

    for b in range(B):
        ref_wav_np = _trim_wav_with_mask(ref_wavs[b], ref_wav_mask[b])
        hyp_wav_np = _trim_wav_with_mask(hyp_wavs[b], hyp_wav_mask[b])

        score = _compute_mcd_dtw_single(
            ref_wav=ref_wav_np,
            hyp_wav=hyp_wav_np,
            exclude_c0=exclude_c0,
        )
        scores.append(score)

    return torch.tensor(scores, dtype=torch.float32, device=ref_wavs.device)


if __name__ == "__main__":
    torch.manual_seed(1234)

    B = 2
    T_ref = 22050  # 1 sec at 22050 Hz
    T_hyp = 24000

    # Random waveform batch
    ref_wavs = torch.randn(B, T_ref) * 0.1
    hyp_wavs = torch.randn(B, T_hyp) * 0.1

    # Sample-level masks
    ref_lengths = torch.tensor([22050, 18000])
    hyp_lengths = torch.tensor([24000, 19000])

    ref_idx = torch.arange(T_ref).unsqueeze(0)
    hyp_idx = torch.arange(T_hyp).unsqueeze(0)

    ref_wav_mask = (ref_idx < ref_lengths.unsqueeze(1)).float()
    hyp_wav_mask = (hyp_idx < hyp_lengths.unsqueeze(1)).float()

    # 1. Random vs random: should be positive
    mcd_rand = compute_mcd_dtw(
        ref_wavs=ref_wavs,
        ref_wav_mask=ref_wav_mask,
        hyp_wavs=hyp_wavs,
        hyp_wav_mask=hyp_wav_mask,
        exclude_c0=True,
    )

    print("Random vs random MCD-DTW:", mcd_rand)
    print("Mean:", mcd_rand.mean().item())

    # 2. Same vs same: should be zero or extremely close to zero
    mcd_same = compute_mcd_dtw(
        ref_wavs=ref_wavs,
        ref_wav_mask=ref_wav_mask,
        hyp_wavs=ref_wavs.clone(),
        hyp_wav_mask=ref_wav_mask.clone(),
        exclude_c0=True,
    )

    print("Same vs same MCD-DTW:", mcd_same)
    print("Mean:", mcd_same.mean().item())

    print("test passed.")
