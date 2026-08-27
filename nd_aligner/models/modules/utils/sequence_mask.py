import torch


def sequence_mask(length: torch.Tensor, max_length: int | None = None):
    if max_length is None:
        max_length = length.max()  # pyright: ignore

    assert max_length is not None
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)
