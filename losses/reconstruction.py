"""
GMTI-Net Reconstruction Losses

Charbonnier Loss:
    L = sqrt((pred - gt)^2 + eps^2),  eps = 1e-3

Laplacian Pyramid Loss:
    4-level pyramid, L1 loss at each level
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (differentiable L1).

    L = sqrt((pred - gt)^2 + eps^2)
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps_sq = eps**2

    def forward(self, pred, gt):
        """
        Args:
            pred: [B, C, H, W]
            gt:   [B, C, H, W]
        Returns:
            loss: scalar
        """
        diff_sq = (pred - gt) ** 2
        loss = torch.sqrt(diff_sq + self.eps_sq)
        return loss.mean()


class LaplacianPyramidLoss(nn.Module):
    """Laplacian pyramid loss.

    Build Gaussian + Laplacian pyramid for pred and gt,
    compute L1 loss at each level.
    """

    def __init__(self, num_levels=4):
        super().__init__()
        self.num_levels = num_levels

    def _gaussian_pyramid(self, img, num_levels):
        """Build Gaussian pyramid."""
        pyramid = [img]
        current = img
        for _ in range(num_levels - 1):
            current = F.avg_pool2d(current, 2, 2)
            pyramid.append(current)
        return pyramid

    def _laplacian_pyramid(self, img, num_levels):
        """Build Laplacian pyramid from Gaussian pyramid."""
        gaussian = self._gaussian_pyramid(img, num_levels)
        laplacian = []
        for i in range(num_levels - 1):
            upsampled = F.interpolate(
                gaussian[i + 1],
                size=gaussian[i].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            laplacian.append(gaussian[i] - upsampled)
        laplacian.append(gaussian[-1])  # Coarsest level
        return laplacian

    def forward(self, pred, gt):
        """
        Args:
            pred: [B, C, H, W]
            gt:   [B, C, H, W]
        Returns:
            loss: scalar
        """
        lap_pred = self._laplacian_pyramid(pred, self.num_levels)
        lap_gt = self._laplacian_pyramid(gt, self.num_levels)

        loss = 0.0
        for lp, lg in zip(lap_pred, lap_gt):
            loss = loss + F.l1_loss(lp, lg)
        return loss
