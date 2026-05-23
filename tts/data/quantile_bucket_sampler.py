import random
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler
from typing import override


class QuantileDurationBatchSampler(Sampler[list[int]]):
    """
    Quantile-based duration bucket batch sampler.

    Behavior:
        1. Split samples into approximately equal-size duration buckets.
        2. Shuffle samples inside each bucket without replacement.
        3. Repeatedly shuffle active bucket order.
        4. From each active bucket, yield one batch of batch_size samples.
        5. If a bucket has fewer than batch_size remaining samples, drop it.
           This is drop_last=True behavior.
    """

    def __init__(
        self,
        batch_size: int,
        bucketing_keys: list[float],
        num_buckets: int = 10,
        *,
        seed: int = 1234,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if num_buckets <= 0:
            raise ValueError(f"num_buckets must be positive, got {num_buckets}.")
        if not bucketing_keys:
            raise ValueError("bucketing_keys must be non-empty.")

        self.bucketing_keys = bucketing_keys
        self.batch_size = int(batch_size)
        self.num_buckets = int(num_buckets)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self.buckets = self._make_quantile_buckets()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _make_quantile_buckets(self) -> list[list[int]]:
        indices = list(range(len(self.bucketing_keys)))

        # Sort by duration. Quantile buckets are made by equal-count slicing.
        indices.sort(key=lambda i: self.bucketing_keys[i])

        n = len(indices)
        num_buckets = min(self.num_buckets, n)

        buckets: list[list[int]] = []
        for b in range(num_buckets):
            start = (b * n) // num_buckets
            end = ((b + 1) * n) // num_buckets
            bucket = indices[start:end]
            if bucket:
                buckets.append(bucket)

        return buckets

    @override
    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)

        # Copy buckets because we destructively consume them this epoch.
        buckets = [bucket.copy() for bucket in self.buckets]

        if self.shuffle:
            for bucket in buckets:
                rng.shuffle(bucket)

        # Pointers for non-replacement sampling inside each bucket.
        ptrs = [0 for _ in buckets]
        active = [i for i, bucket in enumerate(buckets) if len(bucket) >= self.batch_size]

        while active:
            # Group-level random order:
            # e.g. [1, 2, 5, 3, 4], then next round [4, 5, 3, 2, 1], ...
            if self.shuffle:
                rng.shuffle(active)

            next_active: list[int] = []

            for bucket_id in active:
                bucket = buckets[bucket_id]
                ptr = ptrs[bucket_id]
                remaining = len(bucket) - ptr

                if remaining < self.batch_size:
                    # drop_last behavior for this bucket.
                    continue

                batch = bucket[ptr : ptr + self.batch_size]
                ptrs[bucket_id] += self.batch_size

                yield batch

                if len(bucket) - ptrs[bucket_id] >= self.batch_size:
                    next_active.append(bucket_id)

            active = next_active

    def __len__(self) -> int:
        # Since drop_last=True by design, each bucket contributes floor(len/batch_size).
        return sum(len(bucket) // self.batch_size for bucket in self.buckets)
