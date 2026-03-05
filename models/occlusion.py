"""
GMTI-Net Occlusion Network

Input: warped features from both frames + flow + confidence
Output: occlusion mask O(x) ∈ [0,1]
Blend: F = O * F_L + (1-O) * F_R

Architecture:
    Conv3×3 (in→64) → ReLU
    Conv3×3 (64→32) → ReLU
    Conv1×1 (32→1)  → Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .warping import backward_warp


class OcclusionNetwork(nn.Module):
    """Predict occlusion mask from warped features and flow geometry.

    The occlusion mask determines how much each frame contributes
    to the final interpolated result at each pixel. Combines learned
    CNN mask with structural geometric bidirectional constraints.
    """

    def __init__(self, feat_channels=96):
        super().__init__()
        in_channels = 2 * feat_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, warp_L, warp_R, flow_lr=None, flow_rl=None, tau=1.5):
        """
        Args:
            warp_L: [B, C, H, W] setup features
            warp_R: [B, C, H, W]
            flow_lr: [B, 2, H, W] forward flow
            flow_rl: [B, 2, H, W] backward flow
            tau: float, pixel consistency tolerance
        Returns:
            occ_mask: [B, 1, H, W] occlusion mask bounded in [0,1]
        """
        x = torch.cat([warp_L, warp_R], dim=1)
        learned_mask = self.net(x)

        if flow_lr is not None and flow_rl is not None:
            # Calculate geometric bidirectional occlusion mask
            # For un-occluded pixels, flow_lr(x) ≈ -flow_rl(x + flow_lr(x))
            warped_rl = backward_warp(flow_rl, flow_lr)
            bidir_err = (flow_lr + warped_rl).abs().mean(dim=1, keepdim=True)

            # Soft geometric mask via a sigmoid on the negative error distance.
            # This produces values in (0,1] and allows the learned mask to be
            # modulated during training even when errors are small.
            geom_mask = torch.sigmoid(tau - bidir_err)

            # Resize geometric mask to match learned_mask's spatial resolution
            # learned_mask is produced at feature resolution (e.g., H/4); geom_mask
            # is computed at full image resolution. Interpolate down to feature size.
            geom_mask_ds = F.interpolate(
                geom_mask,
                size=learned_mask.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

            if self.training:
                # Combine learned mask and structural geometric mask strictly
                occ_mask = learned_mask * geom_mask_ds
            else:
                occ_mask = learned_mask
        else:
            occ_mask = learned_mask

        return occ_mask

    @staticmethod
    def blend(warp_L, warp_R, occ_mask):
        """Blend warped features using occlusion mask.

        Args:
            warp_L: [B, C, H, W]
            warp_R: [B, C, H, W]
            occ_mask: [B, 1, H, W]
        Returns:
            blended: [B, C, H, W]
        """
        return occ_mask * warp_L + (1.0 - occ_mask) * warp_R
