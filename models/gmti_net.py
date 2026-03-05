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

from .encoder import HybridEncoder
from .flow_estimator import FlowEstimator
from .warping import DualWarping, backward_warp
from .occlusion import OcclusionNetwork
from .transformer import TransformerFusion
from .decoder import FrequencyAwareDecoder


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
        swin_heads=6,
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

    def forward(self, L, R):
        """
        Args:
            L: [B, 3, H, W] left frame (t=0)
            R: [B, 3, H, W] right frame (t=1)
        Returns:
            pred: [B, 3, H, W] predicted middle frame (t=0.5)
            aux: dict with intermediate outputs for loss computation
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

        # 2. Estimate optical flow
        flow_lr, flow_rl, flow_lm, flow_rm, conf_lr, conf_rl = self.flow_estimator(
            features_L, features_R
        )

        # 3. Dual warping
        warped_feat_L, warped_img_L = self.warp_L(L, features_L[2], flow_lm, conf_lr)
        warped_feat_R, warped_img_R = self.warp_R(R, features_R[2], flow_rm, conf_rl)

        # 4. Occlusion estimation and blending
        occ_mask = self.occlusion(warped_feat_L, warped_feat_R, flow_lr, flow_rl)
        fused_feat = OcclusionNetwork.blend(warped_feat_L, warped_feat_R, occ_mask)

        # 5. Transformer fusion with motion guidance
        fused = self.transformer(fused_feat, flow_lm)  # [B, 128, H/4, W/4]

        # 6. Decode
        pred = self.decoder(fused)  # [B, 3, H_padded, W_padded]

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
            "warped_img_L": (
                warped_img_L[:, :, :H, :W] if pad_h > 0 or pad_w > 0 else warped_img_L
            ),
            "warped_img_R": (
                warped_img_R[:, :, :H, :W] if pad_h > 0 or pad_w > 0 else warped_img_R
            ),
        }

        return pred, aux

    def inference(self, L, R):
        """Simple inference: returns only the predicted frame."""
        pred, _ = self.forward(L, R)
        return pred
