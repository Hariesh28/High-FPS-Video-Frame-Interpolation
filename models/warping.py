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
from typing import Tuple, Optional


def flow_to_grid(
    flow: torch.Tensor, accel: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Convert flow (and acceleration) to a normalized sampling grid.
    If accel is provided, uses quadratic motion: x_mid = x0 + 0.5*v + 0.25*a

    Args:
        flow (torch.Tensor): Optical flow [B, 2, H, W].
        accel (Optional[torch.Tensor]): Acceleration [B, 2, H, W].

    Returns:
        torch.Tensor: Normalized sampling grid [B, H, W, 2] in float32.
    """
    B, _, H, W = flow.shape
    flow_f32 = flow.float()

    xs = torch.linspace(0, W - 1, W, device=flow.device, dtype=torch.float32)
    ys = torch.linspace(0, H - 1, H, device=flow.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

    # V-mid = 0.5 * flow + 0.25 * accel (if accel provided)
    # Actually, flow is typically estimated for t=1.
    # Displacement at t=0.5 is 0.5*v + 0.25*a.
    v_mid = 0.5 * flow_f32
    if accel is not None:
        v_mid = v_mid + 0.25 * accel.float()

    vgrid = grid + v_mid.permute(0, 2, 3, 1)

    # Normalise to [-1, 1]
    vgrid_x = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
    return torch.stack((vgrid_x, vgrid_y), dim=-1)


def backward_warp(
    img: torch.Tensor, flow: torch.Tensor, accel: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply backward warping with optional quadratic motion.
    """
    orig_dtype = img.dtype

    with torch.amp.autocast("cuda", enabled=False):
        img_f32 = img.float()
        grid = flow_to_grid(flow, accel)

        warped = F.grid_sample(
            img_f32, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        valid = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)
        valid = valid.unsqueeze(1).float()
        warped = warped * valid

    return warped.to(orig_dtype)


def multi_hypothesis_warp(
    img: torch.Tensor, flows_k: torch.Tensor, conf_k: torch.Tensor
) -> torch.Tensor:
    """
    Fuses K warped versions of the image using predicted confidences.
    flows_k: [B, K, 2, H, W]
    conf_k: [B, K, 1, H, W] (already softmaxed)
    """
    B, K, _, H, W = flows_k.shape
    warped_sum = 0
    for k in range(K):
        w_k = backward_warp(img, flows_k[:, k])
        warped_sum = warped_sum + w_k * conf_k[:, k]
    return warped_sum


def local_kernel_warp(
    img: torch.Tensor, flow: torch.Tensor, kernel_params: torch.Tensor
) -> torch.Tensor:
    """
    Separable Anisotropic Kernel Warping.
    kernel_params: [B, 11, H, W]
        - [0:5]: Row kernel (5 taps)
        - [5:10]: Col kernel (5 taps)
        - [10]: Anisotropy stretch
    """
    B, C, H, W = img.shape
    with torch.amp.autocast("cuda", enabled=False):
        img_f32 = img.float()
        # 1. Softmax to get valid kernels
        k_row = torch.softmax(kernel_params[:, :5], dim=1)
        k_col = torch.softmax(kernel_params[:, 5:10], dim=1)
        aniso = torch.sigmoid(kernel_params[:, 10:11]) + 0.5  # [B, 1, H, W]

        # 2. Base Warping
        grid = flow_to_grid(flow)  # [B, H, W, 2]

        offsets = torch.linspace(-2, 2, 5, device=img.device)
        # Apply anisotropy to offsets: result [B, H, W, 5]
        aniso_3d = aniso.squeeze(1).unsqueeze(-1)  # [B, H, W, 1]
        off_col = offsets.view(1, 1, 1, 5) * aniso_3d
        off_row = offsets.view(1, 1, 1, 5) / aniso_3d

        # 1D sampling (Row): grid_row [B, H, W, 5, 2]
        grid_row = grid.unsqueeze(3) + torch.stack(
            [off_row, torch.zeros_like(off_row)], dim=-1
        ) * (2.0 / W)
        grid_row = grid_row.view(B, H, W * 5, 2)
        samples_row = F.grid_sample(
            img_f32,
            grid_row,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        samples_row = samples_row.view(B, 3, H, W, 5)
        img_row = torch.sum(
            samples_row * k_row.unsqueeze(1).permute(0, 1, 3, 4, 2), dim=-1
        )

        # 1D sampling (Col) on row-filtered result
        grid_base = flow_to_grid(torch.zeros_like(flow))
        grid_col = grid_base.unsqueeze(3) + torch.stack(
            [torch.zeros_like(off_col), off_col], dim=-1
        ) * (2.0 / H)
        grid_col = grid_col.view(B, H, W * 5, 2)
        samples_col = F.grid_sample(
            img_row,
            grid_col,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        samples_col = samples_col.view(B, 3, H, W, 5)
        warped = torch.sum(
            samples_col * k_col.unsqueeze(1).permute(0, 1, 3, 4, 2), dim=-1
        )

    return warped.to(img.dtype)


class DualWarping(nn.Module):
    """Dual warping with Pro upgrades."""

    def __init__(self, feat_channels: int = 96):
        super().__init__()
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
        kernel_params: Optional[torch.Tensor] = None,
        accel: Optional[torch.Tensor] = None,
        flows_k: Optional[torch.Tensor] = None,
        conf_k: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            accel: Patch-based acceleration for quadratic motion.
            flows_k: Multi-hypothesis flows [B, K, 2, H, W].
            conf_k: Multi-hypothesis weights [B, K, 1, H, W].
        """
        _, _, Hf, Wf = features.shape
        B, _, H, W = img.shape

        # 1. Image-level warping
        if flows_k is not None and conf_k is not None:
            # Multi-hypothesis fusion
            warped_img = multi_hypothesis_warp(img, flows_k, conf_k)
        elif kernel_params is not None:
            # Upsample kernel_params if needed
            if kernel_params.shape[2:] != img.shape[2:]:
                kernel_params = F.interpolate(
                    kernel_params,
                    size=img.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            warped_img = local_kernel_warp(img, flow, kernel_params)
        else:
            # Quadratic or Linear warp
            warped_img = backward_warp(img, flow, accel)

        # 2. Feature-level warping (usually coarse, keep it simple/linear)
        flow_scaled = F.interpolate(
            flow, size=(Hf, Wf), mode="bilinear", align_corners=False
        )
        flow_scaled = flow_scaled * (Hf / H)
        warped_feat = backward_warp(features, flow_scaled)

        # Fusion
        warped_img_down = F.interpolate(
            warped_img, size=(Hf, Wf), mode="bilinear", align_corners=False
        )
        fused = self.fusion(torch.cat([warped_feat, warped_img_down], dim=1))
        warped_feat = warped_feat + fused

        return warped_feat, warped_img
