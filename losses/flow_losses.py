"""
GMTI-Net Flow and Combined Losses

Warping Loss:     ||warp(L, F_LM) - M|| + ||warp(R, F_RM) - M||
Bidirectional:    ||F_LR + warp(F_RL, F_LR)||
Flow Smoothness:  |∇F|

Combined Loss:
    L_total = 1.0*charb + 0.2*lap + 0.1*warp + 0.05*bidir + 0.01*smooth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reconstruction import CharbonnierLoss, LaplacianPyramidLoss


def backward_warp(img, flow):
    """Backward warp using grid_sample (local copy to avoid circular imports)."""
    B, C, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=flow.device, dtype=flow.dtype),
        torch.arange(W, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
    x = 2.0 * (grid_x + flow[:, 0]) / (W - 1) - 1.0
    y = 2.0 * (grid_y + flow[:, 1]) / (H - 1) - 1.0
    grid = torch.stack([x, y], dim=-1)
    return F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


class WarpingLoss(nn.Module):
    """Warping consistency loss.

    Measures how well the estimated intermediate flows can
    warp the input frames to match the ground truth middle frame.
    """

    def __init__(self):
        super().__init__()

    def forward(self, L, R, flow_lm, flow_rm, gt):
        """
        Args:
            L: [B, 3, H, W] left frame
            R: [B, 3, H, W] right frame
            flow_lm: [B, 2, H, W] flow L→M
            flow_rm: [B, 2, H, W] flow R→M
            gt: [B, 3, H, W] ground truth middle frame
        Returns:
            loss: scalar
        """
        warped_L = backward_warp(L, flow_lm)
        warped_R = backward_warp(R, flow_rm)
        loss = F.l1_loss(warped_L, gt) + F.l1_loss(warped_R, gt)
        return loss


class BidirectionalFlowLoss(nn.Module):
    """Bidirectional flow consistency loss.

    ||F_LR(x) + F_RL(x + F_LR(x))|| should be ~0 for consistent flow.
    """

    def __init__(self):
        super().__init__()

    def forward(self, flow_lr, flow_rl):
        """
        Args:
            flow_lr: [B, 2, H, W] forward flow
            flow_rl: [B, 2, H, W] backward flow
        Returns:
            loss: scalar
        """
        # Warp F_RL using F_LR to get F_RL at positions displaced by F_LR
        flow_rl_warped = backward_warp(flow_rl, flow_lr)
        # Consistency: F_LR + warped_F_RL ≈ 0
        consistency = flow_lr + flow_rl_warped
        loss = torch.norm(consistency, p=2, dim=1).mean()
        return loss


class FlowSmoothnessLoss(nn.Module):
    """Edge-aware flow smoothness loss.

    |∇F| — penalizes spatial gradients of flow.
    """

    def __init__(self):
        super().__init__()

    def forward(self, flow):
        """
        Args:
            flow: [B, 2, H, W]
        Returns:
            loss: scalar
        """
        # Spatial gradients
        dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        loss = dx.abs().mean() + dy.abs().mean()
        return loss


class CombinedLoss(nn.Module):
    """Combined loss with all components and multi-scale supervision.

    L_total = w_charb * L_charb
            + w_lap * L_lap
            + w_warp * L_warp
            + w_bidir * L_bidir
            + w_smooth * L_smooth

    Multi-scale supervision at scales [1/16, 1/8, 1/4, 1]
    with weights [0.1, 0.2, 0.5, 1.0].
    """

    def __init__(
        self,
        w_charb=1.0,
        w_lap=0.2,
        w_warp=0.1,
        w_bidir=0.05,
        w_smooth=0.01,
        charb_eps=1e-3,
        multiscale_scales=None,
        multiscale_weights=None,
    ):
        super().__init__()
        self.w_charb = w_charb
        self.w_lap = w_lap
        self.w_warp = w_warp
        self.w_bidir = w_bidir
        self.w_smooth = w_smooth

        self.charb = CharbonnierLoss(eps=charb_eps)
        self.lap = LaplacianPyramidLoss(num_levels=4)
        self.warp_loss = WarpingLoss()
        self.bidir_loss = BidirectionalFlowLoss()
        self.smooth_loss = FlowSmoothnessLoss()

        self.ms_scales = multiscale_scales or [0.0625, 0.125, 0.25, 1.0]
        self.ms_weights = multiscale_weights or [0.1, 0.2, 0.5, 1.0]

    def forward(self, pred, gt, L, R, aux):
        """
        Args:
            pred: [B, 3, H, W] predicted frame
            gt: [B, 3, H, W] ground truth
            L: [B, 3, H, W] left input frame
            R: [B, 3, H, W] right input frame
            aux: dict from model forward with flow_lr, flow_rl, flow_lm, flow_rm
        Returns:
            total_loss: scalar
            loss_dict: dict of individual loss components
        """
        flow_lr = aux["flow_lr"]
        flow_rl = aux["flow_rl"]
        flow_lm = aux["flow_lm"]
        flow_rm = aux["flow_rm"]

        # ---- Multi-scale Charbonnier + Laplacian ----
        loss_charb = 0.0
        loss_lap = 0.0
        for scale, weight in zip(self.ms_scales, self.ms_weights):
            if scale < 1.0:
                h = int(pred.shape[2] * scale)
                w = int(pred.shape[3] * scale)
                pred_s = F.interpolate(
                    pred, size=(h, w), mode="bilinear", align_corners=False
                )
                gt_s = F.interpolate(
                    gt, size=(h, w), mode="bilinear", align_corners=False
                )
            else:
                pred_s = pred
                gt_s = gt

            loss_charb = loss_charb + weight * self.charb(pred_s, gt_s)
            loss_lap = loss_lap + weight * self.lap(pred_s, gt_s)

        # ---- Warping loss ----
        loss_warp = self.warp_loss(L, R, flow_lm, flow_rm, gt)

        # ---- Bidirectional flow consistency ----
        loss_bidir = self.bidir_loss(flow_lr, flow_rl)

        # ---- Flow smoothness ----
        loss_smooth = (
            self.smooth_loss(flow_lr)
            + self.smooth_loss(flow_rl)
            + self.smooth_loss(flow_lm)
            + self.smooth_loss(flow_rm)
        ) / 4.0

        # ---- Total ----
        total = (
            self.w_charb * loss_charb
            + self.w_lap * loss_lap
            + self.w_warp * loss_warp
            + self.w_bidir * loss_bidir
            + self.w_smooth * loss_smooth
        )

        loss_dict = {
            "charb": loss_charb.item() if torch.is_tensor(loss_charb) else loss_charb,
            "lap": loss_lap.item() if torch.is_tensor(loss_lap) else loss_lap,
            "warp": loss_warp.item(),
            "bidir": loss_bidir.item(),
            "smooth": loss_smooth.item(),
            "total": total.item(),
        }

        return total, loss_dict
