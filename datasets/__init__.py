"""
GMTI-Net Datasets Package

Exports all three VFI dataset classes plus MixedDataset — a utility that
combines multiple datasets with per-dataset sampling weights using
torch.utils.data.WeightedRandomSampler so mixing ratios are exact
regardless of individual dataset sizes.

Example:
    from datasets import NTIREDataset, Vimeo90KDataset, Adobe240Dataset, MixedDataset
    from torch.utils.data import DataLoader

    ntire   = NTIREDataset("data/ntire/train",   mode="train", crop_size=256)
    vimeo   = Vimeo90KDataset("data/vimeo",      split="train", crop_size=256)
    adobe   = Adobe240Dataset("data/adobe240",   split="train", crop_size=256)

    # 50% Vimeo, 30% Adobe, 20% NTIRE
    mixed, sampler = MixedDataset.build(
        datasets=[ntire, vimeo, adobe],
        weights=[0.20, 0.50, 0.30],
        num_samples=100_000,
    )
    loader = DataLoader(mixed, batch_size=16, sampler=sampler, num_workers=8)
"""

from .ntire_dataset import NTIREDataset
from .vimeo90k import Vimeo90KDataset
from .adobe240 import Adobe240Dataset
from .mixed import MixedDataset

__all__ = [
    "NTIREDataset",
    "Vimeo90KDataset",
    "Adobe240Dataset",
    "MixedDataset",
]
