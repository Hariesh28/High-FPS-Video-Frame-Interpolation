"""
tests/test_mixed_dataset.py

Verifies MixedDataset + WeightedRandomSampler semantics:
  - Total concat length equals sum of individual dataset lengths.
  - Sampler draws num_samples samples per epoch.
  - Deterministic sampling with a fixed generator seed produces identical batches.
  - Sampling proportions are approximately correct over many draws.
"""

import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest
from PIL import Image

from datasets.ntire_dataset import NTIREDataset
from datasets.mixed import MixedDataset


# ── helpers ───────────────────────────────────────────────────────────────────


def make_dummy_ntire(root: Path, n_videos=2, n_frames=5, size=(32, 32)) -> NTIREDataset:
    """Create a minimal NTIRE-style folder and return an NTIREDataset."""
    for v in range(1, n_videos + 1):
        vid = root / f"vid_{v}"
        vid.mkdir(parents=True, exist_ok=True)
        for i in range(n_frames):
            img = Image.new("RGB", size, color=(i * 10, 0, 0))
            img.save(str(vid / f"frame_{i:06d}.png"))
    return NTIREDataset(str(root), mode="train", crop_size=16, augment=False)


# ── tests ─────────────────────────────────────────────────────────────────────


def test_concat_length(tmp_path):
    """ConcatDataset length equals sum of sub-dataset lengths."""
    ds_a = make_dummy_ntire(tmp_path / "a", n_videos=2, n_frames=5)
    ds_b = make_dummy_ntire(tmp_path / "b", n_videos=1, n_frames=4)

    concat, sampler = MixedDataset.build(
        datasets=[ds_a, ds_b],
        weights=[0.5, 0.5],
        num_samples=20,
    )
    assert len(concat) == len(ds_a) + len(ds_b)


def test_sampler_num_samples(tmp_path):
    """WeightedRandomSampler draws exactly num_samples per epoch."""
    ds_a = make_dummy_ntire(tmp_path / "a")
    ds_b = make_dummy_ntire(tmp_path / "b")

    num_samples = 50
    _, sampler = MixedDataset.build(
        datasets=[ds_a, ds_b],
        weights=[0.7, 0.3],
        num_samples=num_samples,
    )
    indices = list(sampler)
    assert len(indices) == num_samples


def test_deterministic_sampling(tmp_path):
    """Same generator seed produces identical sample sequences."""
    ds_a = make_dummy_ntire(tmp_path / "a")
    ds_b = make_dummy_ntire(tmp_path / "b")

    def build(seed):
        gen = torch.Generator().manual_seed(seed)
        _, sampler = MixedDataset.build(
            datasets=[ds_a, ds_b],
            weights=[0.5, 0.5],
            num_samples=30,
            generator=gen,
        )
        return list(sampler)

    assert build(99) == build(99)
    # Different seeds produce different sequences (with overwhelming probability)
    assert build(99) != build(100)


def test_weight_normalisation_sums_to_one(tmp_path):
    """Weights are normalised to sum=1 internally."""
    ds_a = make_dummy_ntire(tmp_path / "a", n_videos=2, n_frames=5)
    ds_b = make_dummy_ntire(tmp_path / "b", n_videos=2, n_frames=5)

    concat, sampler = MixedDataset.build(
        datasets=[ds_a, ds_b],
        weights=[3.0, 1.0],  # un-normalised: 75% / 25%
        num_samples=1000,
    )
    indices = list(sampler)
    # Indices in [0, len(ds_a)) belong to ds_a; ≥ len(ds_a) belong to ds_b.
    boundary = len(ds_a)
    frac_a = sum(1 for i in indices if i < boundary) / len(indices)
    # Expected ≈ 0.75; allow ±10% tolerance for randomness with 1000 samples.
    assert (
        0.65 <= frac_a <= 0.85
    ), f"Sampling fraction for ds_a = {frac_a:.3f}, expected ~0.75"


def test_item_shapes(tmp_path):
    """Each item from the ConcatDataset has shape [3, crop, crop]."""
    crop = 16
    ds_a = make_dummy_ntire(tmp_path / "a", n_videos=1, n_frames=4, size=(64, 64))
    ds_b = make_dummy_ntire(tmp_path / "b", n_videos=1, n_frames=4, size=(64, 64))
    # Override crop_size to a known value
    # Pass crop sizes and enable dataset augment directly so transform is properly built internally
    ds_a = make_dummy_ntire(tmp_path / "a", n_videos=1, n_frames=4, size=(64, 64))
    ds_b = make_dummy_ntire(tmp_path / "b", n_videos=1, n_frames=4, size=(64, 64))
    for ds in (ds_a, ds_b):
        ds.crop_size = crop
        ds.augment = True

    concat, _ = MixedDataset.build(
        datasets=[ds_a, ds_b], weights=[0.5, 0.5], num_samples=10
    )

    L, M, R = concat[0]
    assert L.shape == (3, crop, crop), f"L shape {L.shape} != (3,{crop},{crop})"
    assert M.shape == (3, crop, crop)
    assert R.shape == (3, crop, crop)


def test_deterministic_epoch(tmp_path):
    """deterministic_mixed_epoch produces identical arrays for same seed, different for different seeds."""
    ds_a = make_dummy_ntire(tmp_path / "a", n_videos=2, n_frames=5)
    ds_b = make_dummy_ntire(tmp_path / "b", n_videos=2, n_frames=5)
    num_samples = 40

    idx1 = MixedDataset.deterministic_mixed_epoch(
        datasets=[ds_a, ds_b], weights=[0.6, 0.4], num_samples=num_samples, seed=7
    )
    idx2 = MixedDataset.deterministic_mixed_epoch(
        datasets=[ds_a, ds_b], weights=[0.6, 0.4], num_samples=num_samples, seed=7
    )
    idx3 = MixedDataset.deterministic_mixed_epoch(
        datasets=[ds_a, ds_b], weights=[0.6, 0.4], num_samples=num_samples, seed=8
    )

    # Same seed → identical
    assert torch.equal(idx1, idx2), "Same seed should produce identical index arrays"
    # Different seed → different
    assert not torch.equal(
        idx1, idx3
    ), "Different seeds should produce different arrays"
    # Length matches requested
    assert len(idx1) == num_samples, f"Expected {num_samples} indices, got {len(idx1)}"

    # All indices in [0, total_len)
    total = len(ds_a) + len(ds_b)
    assert (idx1 >= 0).all() and (
        idx1 < total
    ).all(), f"Indices out of bounds [0, {total}): min={idx1.min()}, max={idx1.max()}"
