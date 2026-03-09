"""
GMTI-Net Flow and Combined Losses for High-PSNR Frame Interpolation.

This module implements a suite of loss functions designed to maximize PSNR
while maintaining structural and motion consistency:
    - Charbonnier & Laplacian: Core spatial reconstruction (Stage 1).
    - Frequency-Focal Loss: Band-weighted DCT supervision (Stage 3).
    - Heteroscedastic Loss: Uncertainty-aware regression (Stage 3).
    - Multi-Hypothesis Loss: Diverse motion candidate supervision (Stage 3).
    - Bidirectional & Smoothness: Flow regularizers.
    - VGG Perceptual: Structural regularizer (Stage 2).
    - MSE Loss: Direct PSNR optimizer.

All losses are accumulated in FP32 for numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from .reconstruction import CharbonnierLoss, LaplacianPyramidLoss
from utils.freq import block_dct, get_hf_mask
import torchvision.models as models


def backward_warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Apply backward warping using grid_sample.

    Args:
        img (torch.Tensor): Input frame of shape [B, C, H, W].
        flow (torch.Tensor): Optical flow of shape [B, 2, H, W].

    Returns:
        torch.Tensor: Warped frame of shape [B, C, H, W].
    """
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
    """
    Measures the alignment consistency between warped input frames and the target.
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


class GradientLoss(nn.Module):
    """Edge-sharpening gradient loss.

    L_grad = ||∇pred - ∇gt||_1
    """

    def forward(self, pred, gt):
        # spatial gradients
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        gt_dy = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
        gt_dx = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
        return F.l1_loss(pred_dy, gt_dy) + F.l1_loss(pred_dx, gt_dx)


class FrequencyFocalLoss(nn.Module):
    """
    Band-weighted DCT loss. Penalizes HF discrepancies more heavily.
    Includes normalization by block DC energy for stability.
    """

    def __init__(self, block_size: int = 8):
        super().__init__()
        self.block_size = block_size

    def forward(self, pred, gt, hf_boost=1.0):
        pred32, gt32 = pred.float(), gt.float()
        dct_pred = block_dct(pred32, self.block_size)
        dct_gt = block_dct(gt32, self.block_size)

        # DC components for normalization [B, C, H/b, W/b, 1, 1]
        dc_gt = dct_gt[..., 0, 0].unsqueeze(-1).unsqueeze(-1).abs().clamp_min(1e-3)

        # Create band-weights: higher weight for high frequencies
        b = self.block_size
        mask = torch.zeros((b, b), device=pred.device)
        for i in range(b):
            for j in range(b):
                mask[i, j] = 1.0 + hf_boost * (i + j) / (2 * b - 2)

        # Weighted normalized MSE
        diff = ((dct_pred - dct_gt) / dc_gt) ** 2
        loss = (diff * mask).mean()
        return loss


class HeteroscedasticLoss(nn.Module):
    """
    L = exp(-2s) * (err^2) + 2s
    where s = log_sigma.
    """

    def __init__(self, lambda_reg=1e-3):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, pred, gt, log_sigma):
        err_sq = (pred.float() - gt.float()) ** 2
        s = torch.clamp(log_sigma, -6.0, 6.0)

        loss = torch.exp(-2.0 * s) * err_sq + 2.0 * s

        # Small L2 penalty on s to prevent runaway variance
        reg = self.lambda_reg * (s**2)
        return loss.mean() + reg.mean()


class MultiHypothesisLoss(nn.Module):
    """
    Supervises K flow hypotheses with additional regularizers:
    1. Reconstruction Loss (L1)
    2. Entropy Regularizer (peaky distributions)
    3. Diversity Loss (spread hypotheses)
    """

    def __init__(self, K=3, w_entropy=1e-3, w_div=0.01):
        super().__init__()
        self.K = K
        self.w_entropy = w_entropy
        self.w_div = w_div

    def forward(self, L, R, flows_k_lr, flows_k_rl, conf_k_lr, conf_k_rl):
        B, K, _, H, W = flows_k_lr.shape
        loss_recon = 0

        # 1. Active supervision (unidirectional)
        for k in range(K):
            wR = backward_warp(R, flows_k_lr[:, k])
            loss_recon += F.l1_loss(wR, L)
            wL = backward_warp(L, flows_k_rl[:, k])
            loss_recon += F.l1_loss(wL, R)
        loss_recon /= 2 * K

        # 2. Entropy Regularizer: Σ p * log p
        p_lr = conf_k_lr.clamp(1e-9, 1.0)
        p_rl = conf_k_rl.clamp(1e-9, 1.0)
        loss_entropy = (p_lr * torch.log(p_lr)).sum(1).mean() + (
            p_rl * torch.log(p_rl)
        ).sum(1).mean()

        # 3. Diversity Loss: discourage identical flows
        div = 0
        for i in range(K):
            for j in range(i + 1, K):
                dist_lr = torch.mean(torch.abs(flows_k_lr[:, i] - flows_k_lr[:, j]))
                dist_rl = torch.mean(torch.abs(flows_k_rl[:, i] - flows_k_rl[:, j]))
                div += dist_lr + dist_rl
        loss_div = -div / (K * (K - 1) / 2 + 1e-6)

        return loss_recon + self.w_entropy * loss_entropy + self.w_div * loss_div


class VGGPerceptualLoss(nn.Module):
    """
    Normalized structural stability loss.
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pred, gt):
        # Explicit FP32 for perceptual loss
        with torch.amp.autocast("cuda", enabled=False):
            p = (pred.float() - self.mean) / self.std
            g = (gt.float() - self.mean) / self.std

            feat_p = self.vgg(p)
            feat_g = self.vgg(g)

            # Normalize by channel-wise mean magnitude for stability
            norm_g = feat_g.abs().mean(dim=[2, 3], keepdim=True).clamp_min(1e-3)
            # Use L1 loss as recommended for stability and PSNR focus
            return F.l1_loss(feat_p / norm_g, feat_g / norm_g)


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
        w_charb: float = 0.05,
        w_lap: float = 0.25,
        w_freq: float = 0.25,
        w_warp: float = 0.1,
        w_bidir: float = 0.05,
        w_smooth: float = 0.01,
        w_mse: float = 1.0,
        w_grad: float = 0.05,
        w_hetero: float = 0.1,
        w_multi: float = 0.05,
        w_perceptual: float = 0.005,
        w_accel: float = 1e-4,
        charb_eps: float = 1e-3,
        multiscale_scales=None,
        multiscale_weights=None,
    ):
        super().__init__()
        self.w_charb = w_charb
        self.w_lap = w_lap
        self.w_freq = w_freq
        self.w_warp = w_warp
        self.w_bidir = w_bidir
        self.w_smooth = w_smooth
        self.w_mse = w_mse
        self.w_grad = w_grad
        self.w_hetero = w_hetero
        self.w_multi = w_multi
        self.w_perceptual = w_perceptual
        self.w_accel = w_accel

        self.charb = CharbonnierLoss(eps=charb_eps)
        self.lap = LaplacianPyramidLoss(num_levels=4)
        self.freq_loss = FrequencyFocalLoss(block_size=8)
        self.warp_loss = WarpingLoss()
        self.bidir_loss = BidirectionalFlowLoss()
        self.smooth_loss = FlowSmoothnessLoss()
        self.grad_loss = GradientLoss()
        self.hetero_loss = HeteroscedasticLoss()
        self.multi_loss = MultiHypothesisLoss(K=3)
        self.vgg_loss = VGGPerceptualLoss()

        self.ms_scales = multiscale_scales or [0.0625, 0.125, 0.25, 1.0]
        self.ms_weights = multiscale_weights or [0.05, 0.2, 0.7, 1.0]

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        L: torch.Tensor,
        R: torch.Tensor,
        aux: Dict[str, torch.Tensor],
        progress: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with multi-scale supervision.

        Args:
            pred (torch.Tensor): Predicted frame [B, 3, H, W].
            gt (torch.Tensor): Ground truth frame [B, 3, H, W].
            L (torch.Tensor): Left frame.
            R (torch.Tensor): Right frame.
            aux (Dict): Auxiliary data from model (flow, masks, etc).

        Returns:
            Tuple: (total_loss, dictionary_of_components).
        """
        flow_lr = aux["flow_lr"]
        flow_rl = aux["flow_rl"]
        flow_lm = aux["flow_lm"]
        flow_rm = aux["flow_rm"]

        # ---- Multi-scale Charbonnier + Laplacian + Frequency ----
        loss_charb = 0.0
        loss_lap = 0.0
        loss_freq = 0.0
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

            # Apply DCT loss primarily at higher scales (>= 1/4)
            if scale >= 0.25:
                loss_freq = loss_freq + weight * self.freq_loss(pred_s, gt_s)

        # ---- Warping loss ----
        loss_warp = self.warp_loss(L, R, flow_lm, flow_rm, gt)

        # ---- Multi-Hypothesis Flow supervision + Reg ----
        loss_multi = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        if "flows_k_lr" in aux:
            loss_multi = self.multi_loss(
                L,
                R,
                aux["flows_k_lr"],
                aux["flows_k_rl"],
                aux["conf_k_lr"],
                aux["conf_k_rl"],
            )

        # ---- Bidirectional flow consistency ----
        loss_bidir = self.bidir_loss(flow_lr, flow_rl)

        # ---- Flow smoothness ----
        loss_smooth = (
            self.smooth_loss(flow_lr)
            + self.smooth_loss(flow_rl)
            + self.smooth_loss(flow_lm)
            + self.smooth_loss(flow_rm)
        ) / 4.0

        # ---- Quadratic Motion Regularizer (L2 on Accel) ----
        loss_accel = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        if "accel_lr" in aux:
            loss_accel = (aux["accel_lr"] ** 2).mean() + (aux["accel_rl"] ** 2).mean()

        # ---- MSE & Heteroscedastic (Log-Sigma) ----
        pred32 = pred.float()
        gt32 = gt.float()
        loss_mse = F.mse_loss(pred32, gt32)
        loss_hetero = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
        if "sigma_lr" in aux:
            loss_hetero = self.hetero_loss(
                pred32, gt32, (aux["sigma_lr"] + aux["sigma_rl"]) / 2.0
            )

        # ---- Frequency (DCT) Annealing ----
        # Start small (0.05 boost) -> final (0.25 boost) in last 20% steps
        if progress < 0.8:
            hf_boost = 0.05
        else:
            # Linear ramp from 0.05 to 0.25
            hf_boost = 0.05 + (0.25 - 0.05) * (progress - 0.8) / 0.2
        loss_freq = self.freq_loss(pred32, gt32, hf_boost=hf_boost)

        # ---- Gradient (edges) & Perceptual ----
        loss_grad = self.grad_loss(pred, gt)
        loss_vgg = self.vgg_loss(pred, gt)

        # ---- Total ----
        total = (
            self.w_charb * loss_charb
            + self.w_lap * loss_lap
            + self.w_freq * loss_freq
            + self.w_warp * loss_warp
            + self.w_bidir * loss_bidir
            + self.w_smooth * loss_smooth
            + self.w_mse * loss_mse
            + self.w_grad * loss_grad
            + self.w_hetero * loss_hetero
            + self.w_multi * loss_multi
            + self.w_perceptual * loss_vgg
            + self.w_accel * loss_accel
        )

        loss_dict = {
            "charb": loss_charb.item() if torch.is_tensor(loss_charb) else loss_charb,
            "lap": loss_lap.item() if torch.is_tensor(loss_lap) else loss_lap,
            "freq": loss_freq.item() if torch.is_tensor(loss_freq) else loss_freq,
            "warp": loss_warp.item(),
            "bidir": loss_bidir.item(),
            "smooth": loss_smooth.item(),
            "mse": loss_mse.item(),
            "grad": loss_grad.item(),
            "hetero": loss_hetero.item(),
            "vgg": loss_vgg.item(),
            "accel": loss_accel.item(),
            "total": total.item(),
        }

        return total, loss_dict
