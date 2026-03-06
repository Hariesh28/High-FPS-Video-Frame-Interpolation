"""
GMTI-Net Warping Module

Backward warping using grid_sample.
Dual warping: feature-level (H/4) + image-level (full res).

FP32 safety: grid_sample and all coordinate arithmetic run in float32
explicitly (via autocast guard + .float() casts) since fp16 rounding
of sub-pixel coordinates measurably degrades PSNR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
    """Convert flow [B,2,H,W] → normalised sampling grid [B,H,W,2].

    Runs in float32 regardless of the input dtype to avoid fp16 coordinate
    rounding errors that degrade PSNR.
    """
    B, _, H, W = flow.shape
    # Always compute grid in fp32
    flow_f32 = flow.float()

    xs = torch.linspace(0, W - 1, W, device=flow.device, dtype=torch.float32)
    ys = torch.linspace(0, H - 1, H, device=flow.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
    vgrid = grid + flow_f32.permute(0, 2, 3, 1)  # pixel coords, fp32

    # Normalise to [-1, 1]
    vgrid_x = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
    return torch.stack((vgrid_x, vgrid_y), dim=-1)  # [B, H, W, 2] fp32


def backward_warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warp img using flow.

    grid_sample is performed in float32 then cast back to the input dtype.
    This prevents fp16 bilinear interpolation rounding that silently lowers PSNR.

    Args:
        img:  [B, C, H, W]  — any dtype.
        flow: [B, 2, H, W]  — any dtype.
    Returns:
        warped: [B, C, H, W] in the same dtype as img.
    """
    orig_dtype = img.dtype

    # Force fp32 for grid computation and sampling
    with torch.amp.autocast("cuda", enabled=False):
        img_f32 = img.float()
        grid = flow_to_grid(flow)  # always fp32 from flow_to_grid

        warped = F.grid_sample(
            img_f32, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        # Validity mask: pixels whose sampling location is inside [-1,1]²
        valid = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)
        valid = valid.unsqueeze(1).float()
        warped = warped * valid

    return warped.to(orig_dtype)


class DualWarping(nn.Module):
    """Dual warping: feature-level + image-level warping.

    Feature warping at H/4 scale for semantic alignment.
    Image warping at full resolution for pixel accuracy.
    Results are fused via a residual conv.
    """

    def __init__(self, feat_channels: int = 96):
        super().__init__()

        # Feature-to-image fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(feat_channels + 3, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
        )

    def forward(
        self,
        img: torch.Tensor,
        features: torch.Tensor,
        flow: torch.Tensor,
        confidence=None,  # kept for API compatibility; unused in backward warp
    ):
        """
        Args:
            img:        [B, 3, H, W] input image.
            features:   [B, C, H/4, W/4] stage-3 features.
            flow:       [B, 2, H, W] flow at full resolution.
            confidence: [B, 1, H, W] (optional, not used).
        Returns:
            warped_feat: [B, C, H/4, W/4] warped and fused features.
            warped_img:  [B, 3, H, W] warped image.
        """
        _, _, Hf, Wf = features.shape
        B, _, H, W = img.shape

        # Image-level warping (full resolution) — fp32 guard inside backward_warp
        warped_img = backward_warp(img, flow)

        # Feature-level warping — scale flow to feature resolution
        flow_scaled = F.interpolate(
            flow, size=(Hf, Wf), mode="bilinear", align_corners=False
        )
        flow_scaled = flow_scaled * (Hf / H)  # scale flow magnitude
        warped_feat = backward_warp(features, flow_scaled)

        # Fuse
        warped_img_down = F.interpolate(
            warped_img, size=(Hf, Wf), mode="bilinear", align_corners=False
        )
        fused = self.fusion(torch.cat([warped_feat, warped_img_down], dim=1))
        warped_feat = warped_feat + fused  # residual fusion

        return warped_feat, warped_img
