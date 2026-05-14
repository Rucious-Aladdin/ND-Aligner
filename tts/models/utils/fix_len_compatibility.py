from functools import lru_cache


@lru_cache(maxsize=5)
def fix_len_compatibility(length: int, num_downsamplings_in_unet: int = 2) -> int:
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1
