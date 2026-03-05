"""
GMTI-Net Warping Module

Backward warping using grid_sample.
Dual warping: feature-level (H/4) + image-level (full res).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_to_grid(flow):
    """Convert flow [B,2,H,W] to normalized grid [B,H,W,2] for grid_sample."""
    B, _, H, W = flow.shape
    xs = torch.linspace(0, W - 1, W, device=flow.device)
    ys = torch.linspace(0, H - 1, H, device=flow.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)  # [H,W,2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,H,W,2]
    vgrid = grid + flow.permute(0, 2, 3, 1)  # pixel coords

    # Normalize to [-1, 1]
    vgrid_x = 2.0 * vgrid[..., 0] / (W - 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / (H - 1) - 1.0
    vgrid_norm = torch.stack((vgrid_x, vgrid_y), dim=-1)
    return vgrid_norm


def backward_warp(img, flow):
    """Backward warp using rigorous flow_to_grid setup and grid_sample."""
    grid = flow_to_grid(flow)
    warped = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    # Mask out coordinates falling beyond the frame limits
    valid = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)
    valid = valid.unsqueeze(1).float()
    return warped * valid


class DualWarping(nn.Module):
    """Dual warping: feature-level + image-level warping.

    Feature warping at H/4 scale for semantic alignment.
    Image warping at full resolution for pixel accuracy.
    Results are fused.
    """

    def __init__(self, feat_channels=96):
        super().__init__()

        # Feature-to-image fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(feat_channels + 3, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
        )

    def forward(self, img, features, flow, confidence=None):
        """
        Args:
            img: [B, 3, H, W] input image
            features: [B, C, H/4, W/4] stage3 features
            flow: [B, 2, H, W] flow at full resolution
            confidence: [B, 1, H, W] optional (not used in plain backward warp)
        Returns:
            warped_feat: [B, C, H/4, W/4] warped and fused features
            warped_img: [B, 3, H, W] warped image
        """
        B, _, H, W = img.shape
        _, C, Hf, Wf = features.shape

        # Image-level warping (full resolution)
        warped_img = backward_warp(img, flow)

        # Feature-level warping (scaled flow)
        flow_scaled = F.interpolate(
            flow, size=(Hf, Wf), mode="bilinear", align_corners=False
        )
        flow_scaled = flow_scaled * (Hf / H)  # Scale flow magnitude
        warped_feat = backward_warp(features, flow_scaled)

        # Fuse image and feature warps
        warped_img_down = F.interpolate(
            warped_img, size=(Hf, Wf), mode="bilinear", align_corners=False
        )
        fused = self.fusion(torch.cat([warped_feat, warped_img_down], dim=1))
        warped_feat = warped_feat + fused  # Residual fusion

        return warped_feat, warped_img
