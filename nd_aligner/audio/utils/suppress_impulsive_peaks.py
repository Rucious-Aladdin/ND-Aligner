import torch


def suppress_impulsive_peaks(
    wav: torch.Tensor,
    percentile: float = 99.0,
    multiplier: float = 1.0,
) -> torch.Tensor:
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100.")

    threshold = (
        torch.quantile(
            wav.detach().abs().float().reshape(-1),
            percentile / 100.0,
        )
        * multiplier
    )

    if threshold.item() <= 0:
        return wav

    threshold = threshold.to(
        device=wav.device,
        dtype=wav.dtype,
    )

    return torch.clamp(
        wav,
        min=-threshold,
        max=threshold,
    )
