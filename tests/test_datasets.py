"""
tests/test_datasets.py

Smoke tests for Vimeo90KDataset and Adobe240Dataset.
Creates minimal mock directory structures and verifies:
  - __len__ returns the expected number of triplets.
  - __getitem__ returns (L, M, R) each of shape [3, H, W].
  - Values are in [0, 1].
  - Augmentation path does not crash.
"""

import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from datasets.vimeo90k import Vimeo90KDataset
from datasets.adobe240 import Adobe240Dataset


# ── Factories for mock datasets ───────────────────────────────────────────────


def _make_vimeo90k(root: Path, n_seqs: int = 4, size: tuple = (64, 64)) -> Path:
    """Create minimal Vimeo-90K triplet structure and return root path."""
    seq_root = root / "sequences"
    triplet_list = []

    for i in range(1, n_seqs + 1):
        cat = f"{i:05d}"
        seq = "0001"
        sdir = seq_root / cat / seq
        sdir.mkdir(parents=True, exist_ok=True)
        for fname in ("im1.png", "im2.png", "im3.png"):
            img = Image.new("RGB", size, color=(random.randint(0, 255), 0, 0))
            img.save(str(sdir / fname))
        triplet_list.append(f"{cat}/{seq}")

    # Write train list
    (root / "tri_trainlist.txt").write_text("\n".join(triplet_list) + "\n")
    (root / "tri_testlist.txt").write_text("\n".join(triplet_list) + "\n")
    return root


def _make_adobe240(
    root: Path, n_videos: int = 2, n_frames: int = 5, size: tuple = (64, 64)
) -> Path:
    """Create minimal Adobe-240FPS structure and return root path."""
    for v in range(1, n_videos + 1):
        vid = root / f"video_{v:03d}"
        vid.mkdir(parents=True, exist_ok=True)
        for i in range(n_frames):
            img = Image.new("RGB", size, color=(i * 20, 0, 0))
            img.save(str(vid / f"frame_{i:06d}.png"))
    return root


# ── Vimeo90K tests ────────────────────────────────────────────────────────────


class TestVimeo90KDataset:
    def test_len(self, tmp_path):
        root = _make_vimeo90k(tmp_path / "vimeo", n_seqs=4)
        ds = Vimeo90KDataset(str(root), split="train", crop_size=32, augment=False)
        assert len(ds) == 4

    def test_item_shape(self, tmp_path):
        crop = 32
        root = _make_vimeo90k(tmp_path / "vimeo", size=(64, 64))
        ds = Vimeo90KDataset(str(root), split="train", crop_size=crop, augment=False)
        L, M, R = ds[0]
        assert L.shape == (3, crop, crop), f"Expected (3,{crop},{crop}), got {L.shape}"
        assert M.shape == (3, crop, crop)
        assert R.shape == (3, crop, crop)

    def test_values_in_range(self, tmp_path):
        root = _make_vimeo90k(tmp_path / "vimeo", size=(64, 64))
        ds = Vimeo90KDataset(str(root), split="train", crop_size=32, augment=False)
        L, M, R = ds[0]
        assert L.min() >= 0.0 and L.max() <= 1.0
        assert M.min() >= 0.0 and M.max() <= 1.0
        assert R.min() >= 0.0 and R.max() <= 1.0

    def test_augmentation_no_crash(self, tmp_path):
        root = _make_vimeo90k(tmp_path / "vimeo", n_seqs=2, size=(64, 64))
        ds = Vimeo90KDataset(str(root), split="train", crop_size=32, augment=True)
        for i in range(min(4, len(ds))):
            L, M, R = ds[i]
            assert not (L != L).any(), "NaN in L after augmentation"

    def test_missing_listfile_raises(self, tmp_path):
        root = tmp_path / "empty_vimeo"
        root.mkdir()
        with pytest.raises(FileNotFoundError):
            Vimeo90KDataset(str(root), split="train")


# ── Adobe240 tests ────────────────────────────────────────────────────────────


class TestAdobe240Dataset:
    def test_len(self, tmp_path):
        root = _make_adobe240(tmp_path / "adobe", n_videos=2, n_frames=5)
        ds = Adobe240Dataset(str(root), split="train", crop_size=32, augment=False)
        # Each video with 5 frames yields 3 triplets → 2 × 3 = 6
        assert len(ds) == 6

    def test_item_shape(self, tmp_path):
        crop = 32
        root = _make_adobe240(tmp_path / "adobe", size=(64, 64))
        ds = Adobe240Dataset(str(root), split="train", crop_size=crop, augment=False)
        L, M, R = ds[0]
        assert L.shape == (3, crop, crop)
        assert M.shape == (3, crop, crop)
        assert R.shape == (3, crop, crop)

    def test_values_in_range(self, tmp_path):
        root = _make_adobe240(tmp_path / "adobe", size=(64, 64))
        ds = Adobe240Dataset(str(root), split="train", crop_size=32, augment=False)
        L, M, R = ds[0]
        assert L.min() >= 0.0 and L.max() <= 1.0

    def test_augmentation_no_crash(self, tmp_path):
        root = _make_adobe240(tmp_path / "adobe", n_videos=2, n_frames=6, size=(64, 64))
        ds = Adobe240Dataset(str(root), split="train", crop_size=32, augment=True)
        for i in range(min(4, len(ds))):
            L, M, R = ds[i]
            assert not (L != L).any()

    def test_empty_root_produces_empty_dataset(self, tmp_path):
        root = tmp_path / "empty_adobe"
        root.mkdir()
        ds = Adobe240Dataset(str(root), split="train", crop_size=32, augment=False)
        assert len(ds) == 0
