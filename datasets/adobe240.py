"""
Adobe-240FPS Triplet Dataset for VFI.

Expected directory structure:
    <root>/
        video_001/
            frame_000000.png
            frame_000001.png
            ...
        video_002/
            ...

Every consecutive triple of frames (i, i+1, i+2) forms one (L, M, R) triplet
where M at index i+1 is the ground-truth intermediate frame (t=0.5 between
frames i and i+2, assuming the source video is already at 240fps so adjacent
frames are equidistant in time).

Usage:
    ds = Adobe240Dataset(root="data/adobe240", split="train", crop_size=256)
    L, M, R = ds[0]   # each [3, crop_size, crop_size] float32 in [0,1]
"""

import os
import glob
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class Adobe240Dataset(Dataset):
    """Adobe-240FPS Triplet dataset (same (L, M, R) interface as NTIREDataset).

    All frames in each video directory are sorted lexicographically, then
    consecutive triples are extracted: (frame[i], frame[i+1], frame[i+2]).

    Args:
        root:       Path to dataset root containing per-video sub-directories.
        split:      'train' or 'val' / 'test' — controls augmentation default.
        crop_size:  Random crop size. 0 = return full frames.
        augment:    Whether to apply augmentations (auto-False for val/test).
        splits_file: Optional path to a text file listing sub-directory names
                     for this split (one per line). If None, all sub-dirs used.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        crop_size: int = 256,
        augment: bool = True,
        splits_file: str = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size
        self.augment = augment and (split == "train")

        # Discover video directories
        if splits_file and Path(splits_file).exists():
            with open(splits_file) as f:
                vid_names = [l.strip() for l in f if l.strip()]
            vid_dirs = [self.root / n for n in vid_names if (self.root / n).is_dir()]
        else:
            vid_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())

        self.triplets = []
        for vid_dir in vid_dirs:
            frames = sorted(glob.glob(str(vid_dir / "*.png")))
            # Also support .jpg
            if not frames:
                frames = sorted(glob.glob(str(vid_dir / "*.jpg")))
            for i in range(len(frames) - 2):
                self.triplets.append((frames[i], frames[i + 1], frames[i + 2]))

        print(
            f"[Adobe240Dataset] {split}: {len(self.triplets)} triplets "
            f"from {len(vid_dirs)} videos in {root}"
        )

    def __len__(self) -> int:
        return len(self.triplets)

    # ── Internal helpers (identical API to NTIREDataset) ──────────────────────

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = torch.from_numpy(np.array(img)).float() / 255.0
        return img.permute(2, 0, 1)  # [3, H, W]

    def _random_crop(self, imgs, crop_size: int):
        _, H, W = imgs[0].shape
        if H < crop_size or W < crop_size:
            scale = max(crop_size / H, crop_size / W) + 0.01
            new_h, new_w = int(H * scale), int(W * scale)
            imgs = [TF.resize(img, [new_h, new_w], antialias=True) for img in imgs]
            _, H, W = imgs[0].shape
        top = random.randint(0, H - crop_size)
        left = random.randint(0, W - crop_size)
        return [img[:, top : top + crop_size, left : left + crop_size] for img in imgs]

    def _augment(self, imgs):
        if random.random() < 0.5:
            imgs = [TF.hflip(img) for img in imgs]
        if random.random() < 0.5:
            imgs = [TF.vflip(img) for img in imgs]
        if random.random() < 0.25:
            angle = random.choice([90, 180, 270])
            imgs = [TF.rotate(img, angle) for img in imgs]
        if random.random() < 0.5:  # temporal reversal
            imgs = [imgs[2], imgs[1], imgs[0]]
        if random.random() < 0.5:
            bf = 1.0 + random.uniform(-0.1, 0.1)
            imgs = [TF.adjust_brightness(img, bf) for img in imgs]
        if random.random() < 0.5:
            cf = 1.0 + random.uniform(-0.1, 0.1)
            imgs = [TF.adjust_contrast(img, cf) for img in imgs]
        if random.random() < 0.5:
            gf = 1.0 + random.uniform(-0.05, 0.05)
            imgs = [TF.adjust_gamma(img, gf) for img in imgs]
        return imgs

    # ── __getitem__ ────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        """
        Returns:
            L: [3, H, W] left frame
            M: [3, H, W] middle frame (ground truth t=0.5)
            R: [3, H, W] right frame
        """
        path_L, path_M, path_R = self.triplets[idx]
        imgs = [self._load_image(p) for p in (path_L, path_M, path_R)]

        if self.augment:
            imgs = self._random_crop(imgs, self.crop_size)
            imgs = self._augment(imgs)
        elif self.crop_size > 0:
            imgs = self._random_crop(imgs, self.crop_size)

        imgs = [img.clamp(0.0, 1.0) for img in imgs]
        return imgs[0], imgs[1], imgs[2]
