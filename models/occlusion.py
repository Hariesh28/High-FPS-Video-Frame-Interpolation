"""
GMTI-Net Occlusion Network

Input: warped features from both frames (+ optional flow for training)
Output: occlusion mask O(x) ∈ [0,1]
Blend: F = O * F_L + (1-O) * F_R

Architecture:
    Conv3×3 (in→64) → ReLU
    Conv3×3 (64→32) → ReLU
    Conv1×1 (32→1)  → Sigmoid

Training vs Inference difference
---------------------------------
* **Training**: the caller provides a precomputed ``training_geom_mask`` (e.g.
  from bidirectional flow consistency).  The output is ``learned_mask * geom_mask``
  so the geometric constraint steers learning early and fades out as training
  progresses (or can be omitted once the CNN is well-trained).
* **Inference**: ``training_geom_mask`` is ignored regardless of self.training;
  only the learned CNN mask is returned.  This gives stable, artefact-free results
  on novel inputs where geometric heuristics may fail.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OcclusionNetwork(nn.Module):
    """Predict per-pixel occlusion mask from warped features.

    The occlusion mask determines how much each frame contributes to the final
    interpolated result.  During training an external geometric bidirectional
    mask can be injected to guide the CNN; at inference time only the CNN
    output is used.
    """

    def __init__(self, feat_channels: int = 96):
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

    def forward(
        self,
        warp_L: torch.Tensor,
        warp_R: torch.Tensor,
        training_geom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute occlusion mask.

        Args:
            warp_L:              [B, C, H, W] warped left-frame features.
            warp_R:              [B, C, H, W] warped right-frame features.
            training_geom_mask:  [B, 1, H, W] precomputed geometric consistency
                                 mask (e.g. ``(bidir_err < tau).float()``).
                                 Only applied during training; ignored at inference.

        Returns:
            occ_mask: [B, 1, H, W] occlusion mask in [0, 1].

        Notes:
            Caller is responsible for computing the geometric mask, e.g.::

                warped_rl, _ = backward_warp(flow_rl, flow_lr)
                bidir_err    = (flow_lr + warped_rl).abs().mean(1, keepdim=True)
                geom_mask    = (bidir_err < tau).float()
        """
        x = torch.cat([warp_L, warp_R], dim=1)
        learned_mask = self.net(x)  # [B, 1, H, W] ∈ (0, 1)

        # Inference always uses the learned mask alone.
        if not self.training or training_geom_mask is None:
            return learned_mask

        # Training: modulate by geometric consistency mask (both in [0, 1]).
        # Resize geom_mask to match learned_mask resolution if needed.
        if training_geom_mask.shape[2:] != learned_mask.shape[2:]:
            training_geom_mask = F.interpolate(
                training_geom_mask,
                size=learned_mask.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        return learned_mask * training_geom_mask

    @staticmethod
    def blend(
        warp_L: torch.Tensor,
        warp_R: torch.Tensor,
        occ_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Blend warped features using the occlusion mask.

        Args:
            warp_L:   [B, C, H, W]
            warp_R:   [B, C, H, W]
            occ_mask: [B, 1, H, W]

        Returns:
            blended: [B, C, H, W]
        """
        return occ_mask * warp_L + (1.0 - occ_mask) * warp_R
