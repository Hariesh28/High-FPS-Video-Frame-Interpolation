"""
Vimeo-90K Triplet Dataset for VFI.

Expected directory structure (standard Vimeo-90K septuplet release):
    <root>/
        tri_trainlist.txt        (one relative sequence path per line)
        tri_testlist.txt
        sequences/
            00001/
                0001/
                    im1.png   ← left frame
                    im2.png   ← middle frame (ground truth at t=0.5)
                    im3.png   ← right frame
                0002/
                    ...

Usage:
    ds = Vimeo90KDataset(root="data/vimeo_triplet", split="train", crop_size=256)
    L, M, R = ds[0]   # each [3, crop_size, crop_size] float32 in [0,1]
"""

import os
import random
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class Vimeo90KDataset(Dataset):
    """Vimeo-90K Triplet dataset (same (L, M, R) interface as NTIREDataset).

    Args:
        root:       Path to vimeo_triplet directory.
        split:      'train' or 'test'.
        crop_size:  Random crop size for training. 0 = no crop (full frame).
        augment:    Apply data augmentation (only meaningful in train split).
    """

    LISTFILES = {
        "train": "tri_trainlist.txt",
        "test": "tri_testlist.txt",
        "val": "tri_testlist.txt",  # alias
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        crop_size: int = 256,
        augment: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size
        self.augment = augment and (split == "train")

        list_file = self.LISTFILES.get(split)
        if list_file is None:
            raise ValueError(
                f"Unknown split '{split}'. Choose from {list(self.LISTFILES)}"
            )

        list_path = self.root / list_file
        if not list_path.exists():
            raise FileNotFoundError(
                f"Split list not found: {list_path}\n"
                "Download Vimeo-90K triplet dataset and point root= to vimeo_triplet/."
            )

        seq_root = self.root / "sequences"
        self.triplets = []
        with open(list_path) as f:
            for line in f:
                rel = line.strip()
                if not rel:
                    continue
                seq_dir = seq_root / rel
                im1 = seq_dir / "im1.png"
                im2 = seq_dir / "im2.png"
                im3 = seq_dir / "im3.png"
                if im1.exists() and im2.exists() and im3.exists():
                    self.triplets.append((str(im1), str(im2), str(im3)))

        print(
            f"[Vimeo90KDataset] {split}: {len(self.triplets)} triplets "
            f"from {list_path.name}"
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
