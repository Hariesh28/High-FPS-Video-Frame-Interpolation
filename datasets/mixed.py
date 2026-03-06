"""
MixedDataset — combine multiple VFI datasets with per-dataset sampling weights.

Standard torch ConcatDataset picks samples from each sub-dataset in proportion
to their lengths, which biases toward large datasets (Vimeo-90K: ~51k triplets,
Adobe: ~300k triplets, NTIRE: varies).  MixedDataset.build() creates a
ConcatDataset together with a WeightedRandomSampler so the actual per-epoch
sampling ratio matches the user-specified float weights exactly.

Usage:
    mixed, sampler = MixedDataset.build(
        datasets=[ntire, vimeo90k, adobe240],
        weights=[0.20, 0.50, 0.30],   # must sum to 1.0 (or will be normalised)
        num_samples=100_000,           # epoch length in samples
        generator=torch.Generator().manual_seed(42),  # optional determinism
    )
    loader = DataLoader(mixed, batch_size=16, sampler=sampler, drop_last=True)
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler


class MixedDataset:
    """Factory for (ConcatDataset, WeightedRandomSampler) pairs.

    Not itself a Dataset — call MixedDataset.build() to obtain a usable pair.
    """

    @staticmethod
    def build(
        datasets: List[Dataset],
        weights: List[float],
        num_samples: int = 100_000,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        """Build a mixed dataset with correct per-source sampling proportions.

        Args:
            datasets:    List of Dataset objects (all must return (L, M, R)).
            weights:     Per-dataset desired sampling probability (will be
                         normalised to sum=1).  Length must match datasets.
            num_samples: Total number of samples drawn per epoch.
            replacement: Whether to sample with replacement (default True).
                         Set False only for small datasets where you want
                         guaranteed unique samples — may cause StopIteration.
            generator:   Optional RNG for reproducible sampling.

        Returns:
            (concat_dataset, sampler):
                concat_dataset — standard ConcatDataset of all datasets.
                sampler        — WeightedRandomSampler configured for the
                                 desired mixing ratios and num_samples.

        Raises:
            ValueError: If lengths of datasets and weights don't match, or
                        any weight is negative.
        """
        if len(datasets) != len(weights):
            raise ValueError(
                f"len(datasets)={len(datasets)} must equal len(weights)={len(weights)}"
            )
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative.")
        if sum(weights) == 0:
            raise ValueError("Weights must not all be zero.")

        # Normalise weights to sum=1
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

        # Build one sample-weight per element in the concatenated dataset
        sample_weights: List[float] = []
        for ds, w in zip(datasets, norm_weights):
            n = len(ds)  # type: ignore[arg-type]
            if n == 0:
                continue
            # Each sample in this dataset gets weight = desired_fraction / dataset_size
            # so that each dataset collectively contributes in proportion to w.
            per_sample = w / n
            sample_weights.extend([per_sample] * n)

        concat_ds = ConcatDataset(datasets)  # type: ignore[arg-type]

        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )

        # Attach metadata for logging
        concat_ds.dataset_names = [  # type: ignore[attr-defined]
            type(ds).__name__ for ds in datasets
        ]
        concat_ds.dataset_weights = norm_weights  # type: ignore[attr-defined]

        return concat_ds, sampler

    @staticmethod
    def deterministic_mixed_epoch(
        datasets: List[Dataset],
        weights: List[float],
        num_samples: int = 100_000,
        seed: int = 42,
    ) -> torch.Tensor:
        """Precompute a deterministic epoch-length index array.

        Unlike WeightedRandomSampler, this uses numpy to draw `num_samples`
        indices from the concatenated dataset according to `weights` in a
        fully reproducible way.  Useful for:
          - Logging the exact samples seen in an experiment.
          - Debugging dataset imbalances.
          - Reproducing results without relying on torch RNG state.

        Args:
            datasets:    List of Dataset objects.
            weights:     Per-dataset desired sampling probabilities (normalised internally).
            num_samples: Number of indices in the epoch.
            seed:        Random seed for reproducibility.

        Returns:
            indices: torch.LongTensor of shape [num_samples] with indices into
                     the ConcatDataset ordering.
        """
        import numpy as np

        if len(datasets) != len(weights):
            raise ValueError(
                f"len(datasets)={len(datasets)} must equal len(weights)={len(weights)}"
            )
        total_w = sum(weights)
        norm_w = [w / total_w for w in weights]

        rng = np.random.default_rng(seed)

        # For each dataset, produce a pool of indices in the global concat space.
        ds_sizes = [len(ds) for ds in datasets]  # type: ignore[arg-type]
        ds_offsets = [0] + list(np.cumsum(ds_sizes[:-1]))

        all_indices: List[int] = []
        for ds, offset, w in zip(datasets, ds_offsets, norm_w):
            n_draw = max(1, round(num_samples * w))
            n = len(ds)  # type: ignore[arg-type]
            local = rng.integers(0, n, size=n_draw)
            global_idx = local + offset
            all_indices.extend(global_idx.tolist())

        # Shuffle the combined index list, then truncate/extend to exactly num_samples
        all_indices_arr = np.array(all_indices, dtype=np.int64)
        rng.shuffle(all_indices_arr)

        if len(all_indices_arr) > num_samples:
            all_indices_arr = all_indices_arr[:num_samples]
        elif len(all_indices_arr) < num_samples:
            extra = rng.choice(all_indices_arr, size=num_samples - len(all_indices_arr))
            all_indices_arr = np.concatenate([all_indices_arr, extra])

        return torch.from_numpy(all_indices_arr).long()
