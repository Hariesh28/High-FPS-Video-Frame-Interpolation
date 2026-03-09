"""
GMTI-Net: Global Motion-guided Transformer Interpolation Network

Full model assembly. Orchestrates:
    Encoder → Flow Estimator → Dual Warping → Occlusion →
    Transformer Fusion → Frequency-Aware Decoder

Forward pass:
    fL, fR = encoder(L), encoder(R)
    flows = flow_estimator(fL, fR)
    warps = dual_warp(L, R, features, flows)
    occ = occlusion(warps)
    fused = blend + transformer(fused)
    pred = decoder(fused)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Tuple, Optional

from .encoder import HybridEncoder
from .flow_estimator import FlowEstimator
from .warping import DualWarping, backward_warp
from .occlusion import OcclusionNetwork
from .transformer import TransformerFusion
from .decoder import FrequencyAwareDecoder

logger = logging.getLogger(__name__)


class GMTINet(nn.Module):
    """
    GMTI-Net: Video Frame Interpolation

    Given two input frames L (t=0) and R (t=1),
    predicts the intermediate frame M (t=0.5).
    """

    def __init__(
        self,
        in_channels=3,
        encoder_channels=(32, 64, 96, 128, 160),
        swin_depth=4,
        swin_heads=8,
        swin_window_size=8,
        swin_mlp_ratio=4.0,
        flow_refinement_iters=3,
        use_deformable=True,
        transformer_blocks=6,
        transformer_heads=8,
        transformer_dim=128,
        transformer_mlp_ratio=4.0,
    ):
        super().__init__()

        # Shared encoder (weight-shared for L and R)
        self.encoder = HybridEncoder(
            in_channels=in_channels,
            stage_channels=encoder_channels,
            swin_depth=swin_depth,
            swin_heads=swin_heads,
            swin_window_size=swin_window_size,
            swin_mlp_ratio=swin_mlp_ratio,
        )

        # Flow estimation
        self.flow_estimator = FlowEstimator(
            use_deformable=use_deformable,
            refine_iters=flow_refinement_iters,
        )

        # Dual warping (feature + image level)
        self.warp_L = DualWarping(feat_channels=encoder_channels[2])  # 96
        self.warp_R = DualWarping(feat_channels=encoder_channels[2])

        # Occlusion network
        self.occlusion = OcclusionNetwork(feat_channels=encoder_channels[2])  # 96

        # Transformer fusion
        self.transformer = TransformerFusion(
            in_channels=encoder_channels[2],  # 96
            embed_dim=transformer_dim,
            num_blocks=transformer_blocks,
            num_heads=transformer_heads,
            mlp_ratio=transformer_mlp_ratio,
        )

        # Frequency-aware decoder
        self.decoder = FrequencyAwareDecoder(in_channels=transformer_dim)

        logger.info(
            f"GMTI-Net initialized with {encoder_channels} encoder channels, "
            f"swin_depth={swin_depth}, transformer_blocks={transformer_blocks}"
        )

    def forward(
        self, L: torch.Tensor, R: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Execute the full GMTI-Net forward pass.

        Args:
            L (torch.Tensor): Left input frame [B, 3, H, W] at t=0.
            R (torch.Tensor): Right input frame [B, 3, H, W] at t=1.

        Returns:
            Tuple: (predicted_frame [B, 3, H, W], auxiliary_data_dict).
        """
        # Ensure input dimensions are divisible by 16 (for encoder)
        B, C, H, W = L.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            L = F.pad(L, (0, pad_w, 0, pad_h), mode="reflect")
            R = F.pad(R, (0, pad_w, 0, pad_h), mode="reflect")

        # 1. Encode both frames (shared weights)
        features_L = self.encoder(L)  # [f1, f2, f3, f4, f5]
        features_R = self.encoder(R)

        # 2. Estimate optical flow (with RAFT-style iteration and kernels)
        flow_lr, flow_rl, flow_lm, flow_rm, conf_lr, conf_rl, kern_l, kern_r = (
            self.flow_estimator(features_L, features_R)
        )

        # 3. Dual warping (using learned local kernels for micro-motion fix)
        warped_feat_L, warped_img_L = self.warp_L(L, features_L[2], flow_lm, kern_l)
        warped_feat_R, warped_img_R = self.warp_R(R, features_R[2], flow_rm, kern_r)

        # 4. Occlusion estimation and blending
        # Compute geometric consistency mask for training guidance
        if self.training:
            warped_rl = backward_warp(flow_rl, flow_lr)
            bidir_err = (flow_lr + warped_rl).abs().mean(dim=1, keepdim=True)
            geom_mask = (bidir_err < 1.0).float()
            occ_mask = self.occlusion(warped_feat_L, warped_feat_R, geom_mask)
        else:
            occ_mask = self.occlusion(warped_feat_L, warped_feat_R)

        # 4b. Confidence-Aware Blending (V3 improvement)
        # Combine occlusion mask with flow confidence for robust fusion
        conf_mask = conf_lr / (conf_lr + conf_rl + 1e-6)

        # Downsample confidence to match occlusion mask (1/4 res)
        conf_mask_s4 = F.interpolate(
            conf_mask, size=occ_mask.shape[2:], mode="bilinear", align_corners=False
        )

        fused_mask = 0.5 * occ_mask + 0.5 * conf_mask_s4
        fused_feat = OcclusionNetwork.blend(warped_feat_L, warped_feat_R, fused_mask)

        # 5. Transformer fusion with motion guidance
        fused = self.transformer(fused_feat, flow_lm)  # [B, 128, H/4, W/4]

        # 6. Warp high-resolution features for skip connections (U-Net style)
        # Warp Stage-1 (H) and Stage-2 (H/2) features from both L and R, then average
        s1_L = features_L[0]  # [B, 32, H, W]
        s1_R = features_R[0]
        s2_L = features_L[1]  # [B, 64, H/2, W/2]
        s2_R = features_R[1]

        # Warp S1 to M
        w_s1_L = backward_warp(s1_L, flow_lm)
        w_s1_R = backward_warp(s1_R, flow_rm)
        s1_skip = (w_s1_L + w_s1_R) / 2.0

        # Warp S2 to M (scale flow_lm/rm for H/2)
        w_s2_L = backward_warp(
            s2_L,
            F.interpolate(
                flow_lm, scale_factor=0.5, mode="bilinear", align_corners=False
            )
            * 0.5,
        )
        w_s2_R = backward_warp(
            s2_R,
            F.interpolate(
                flow_rm, scale_factor=0.5, mode="bilinear", align_corners=False
            )
            * 0.5,
        )
        s2_skip = (w_s2_L + w_s2_R) / 2.0

        # 7. Decode residual using warped skip connections
        res = self.decoder(fused, (s2_skip, s1_skip))  # [B, 3, H_padded, W_padded]

        # 7. Residual addition (base is average of L and R)
        base = (L + R) / 2.0
        pred = base + res

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            pred = pred[:, :, :H, :W]

        # Auxiliary outputs for loss computation
        aux = {
            "flow_lr": flow_lr,
            "flow_rl": flow_rl,
            "flow_lm": flow_lm,
            "flow_rm": flow_rm,
            "conf_lr": conf_lr,
            "conf_rl": conf_rl,
            "occ_mask": occ_mask,
            "fused_mask": fused_mask,
            "warped_img_L": (
                warped_img_L[:, :, :H, :W] if pad_h > 0 or pad_w > 0 else warped_img_L
            ),
            "warped_img_R": (
                warped_img_R[:, :, :H, :W] if pad_h > 0 or pad_w > 0 else warped_img_R
            ),
        }

        return pred, aux

    def inference(self, L: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Run inference only, returning the interpolated frame.

        Args:
            L (torch.Tensor): Left input frame [B, 3, H, W].
            R (torch.Tensor): Right input frame [B, 3, H, W].

        Returns:
            torch.Tensor: Interpolated middle frame [B, 3, H, W].
        """
        pred, _ = self.forward(L, R)
        return pred
