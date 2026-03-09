"""
GMTI-Net NTIRE Dataset

Loads video frame triplets (L, M, R) from directory structure:
    root/
      vid_1/
        frame_000001.png
        frame_000020.png
        ...
      vid_2/
        ...

Triplets extracted from sorted frames: all (i, mid, j) where j-i >= 2 and j-i is even.

Augmentations (train):
    Random crop (256/384/512), h-flip, v-flip, rotation, temporal reversal,
    brightness/contrast/gamma jitter.
"""

import os
import random
import glob
import logging
from typing import List, Tuple, Optional, Union

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


class NTIREDataset(Dataset):
    """
    NTIRE Video Frame Interpolation dataset.

    Attributes:
        root (str): Path to dataset root.
        mode (str): 'train' or 'val'.
        crop_size (int): Size for random cropping.
        augment (bool): Whether to apply data augmentation.
    """

    def __init__(
        self, root: str, mode: str = "train", crop_size: int = 256, augment: bool = True
    ):
        super().__init__()
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")

        self.root = root
        self.mode = mode
        self.crop_size = crop_size
        self.augment = augment and (mode == "train")

        # Discover all video directories
        self.triplets = []
        vid_dirs = sorted(glob.glob(os.path.join(root, "vid_*")))

        for vid_dir in vid_dirs:
            frames = sorted(glob.glob(os.path.join(vid_dir, "*.png")))
            if len(frames) < 3:
                continue

            # Extract valid triplets via simple sliding window (i, i+1, i+2)
            for i in range(len(frames) - 2):
                self.triplets.append((frames[i], frames[i + 1], frames[i + 2]))

        logger.info(
            f"[NTIREDataset] {mode}: Found {len(self.triplets)} triplets "
            f"from {len(vid_dirs)} videos in {root}"
        )

    def __len__(self):
        return len(self.triplets)

    def _load_image(self, path: str) -> torch.Tensor:
        """
        Load image as float32 tensor in [0, 1].

        Args:
            path (str): Path to image file.

        Returns:
            torch.Tensor: Image tensor of shape [3, H, W].
        """
        img = Image.open(path).convert("RGB")
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img = img.permute(2, 0, 1)  # [3, H, W]
        return img

    def _random_crop(self, imgs, crop_size):
        """Apply same random crop to all images."""
        _, H, W = imgs[0].shape
        if H < crop_size or W < crop_size:
            # Resize if image is smaller than crop
            scale = max(crop_size / H, crop_size / W) + 0.01
            new_h, new_w = int(H * scale), int(W * scale)
            imgs = [TF.resize(img, [new_h, new_w], antialias=True) for img in imgs]
            _, H, W = imgs[0].shape

        top = random.randint(0, H - crop_size)
        left = random.randint(0, W - crop_size)
        imgs = [img[:, top : top + crop_size, left : left + crop_size] for img in imgs]
        return imgs

    def _augment(self, imgs):
        """Apply augmentations to all images identically."""
        # Horizontal flip
        if random.random() < 0.5:
            imgs = [TF.hflip(img) for img in imgs]

        # Vertical flip
        if random.random() < 0.5:
            imgs = [TF.vflip(img) for img in imgs]

        # Rotation (90° increments)
        if random.random() < 0.25:
            angle = random.choice([90, 180, 270])
            imgs = [TF.rotate(img, angle) for img in imgs]

        # Temporal reversal
        if random.random() < 0.5:
            imgs = [imgs[2], imgs[1], imgs[0]]

        # Photometric augmentation (same for all frames)
        if random.random() < 0.5:
            # Brightness
            brightness_factor = 1.0 + random.uniform(-0.1, 0.1)
            imgs = [TF.adjust_brightness(img, brightness_factor) for img in imgs]

        if random.random() < 0.5:
            # Contrast
            contrast_factor = 1.0 + random.uniform(-0.1, 0.1)
            imgs = [TF.adjust_contrast(img, contrast_factor) for img in imgs]

        if random.random() < 0.5:
            # Gamma
            gamma = 1.0 + random.uniform(-0.05, 0.05)
            imgs = [TF.adjust_gamma(img, gamma) for img in imgs]

        return imgs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get frame triplet.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (L, M, R) images.
        """
        path_L, path_M, path_R = self.triplets[idx]

        L = self._load_image(path_L)
        M = self._load_image(path_M)
        R = self._load_image(path_R)

        imgs = [L, M, R]

        # Random crop
        if self.augment:
            imgs = self._random_crop(imgs, self.crop_size)

        # Augmentations
        if self.augment:
            imgs = self._augment(imgs)

        # Clamp to [0, 1]
        imgs = [img.clamp(0.0, 1.0) for img in imgs]

        return imgs[0], imgs[1], imgs[2]
