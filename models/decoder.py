"""
GMTI-Net Decoder

Frequency-aware reconstruction with:
- Low-frequency branch (progressive upsampling)
- High-frequency branch (edge + detail prediction)
- Detail refinement head
- Laplacian residual merge for final output

Input:  [B, 128, H/4, W/4] from transformer fusion
Output: [B, 3, H, W] reconstructed frame
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResBlock(nn.Module):
    """
    Residual block with GroupNorm and GELU.

    Attributes:
        channels (int): Number of input/output channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act(out + residual)


class LowFrequencyBranch(nn.Module):
    """Low-frequency progressive upsampling branch with skip connections.

    Fuses encoder features at H/2 and H resolutions to recover textures.
    """

    def __init__(self, in_channels=128):
        super().__init__()
        # H/4 → H/2
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, 3, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            ResBlock(96),
            ResBlock(96),
        )

        # H/2 → H
        self.up2 = nn.Sequential(
            nn.Conv2d(96 + 64, 64, 3, padding=1, bias=False),  # 64 from skip
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResBlock(64),
            ResBlock(64),
        )

        # Final projection
        self.to_rgb = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1, bias=False),  # 32 from skip
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResBlock(32),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, skips: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute progressive upsampling with skip fusion.

        Args:
            x (torch.Tensor): Latent features [B, 128, H/4, W/4].
            skips (Tuple): Warped encoder features (S2 [B, 64, H/2, W/2], S1 [B, 32, H, W]).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (low_freq_rgb, spatial_features).
        """
        s2, s1 = skips

        # H/4 → H/2
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Fuse Stage-2 Skip
        x = torch.cat([x, s2], dim=1)

        # H/2 → H
        x = self.up2(x)
        feat = x  # Save for refinement head
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Fuse Stage-1 Skip
        x = torch.cat([x, s1], dim=1)

        # To RGB
        low_freq = self.to_rgb(x)
        feat = F.interpolate(feat, scale_factor=2, mode="bilinear", align_corners=False)

        return low_freq, feat


class HighFrequencyBranch(nn.Module):
    """High-frequency branch: predict edge map and detail residual."""

    def __init__(self, in_channels=128):
        super().__init__()
        # Edge prediction
        self.edge_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        # Detail residual prediction
        self.detail_net = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict high-frequency detail residuals.

        Args:
            x (torch.Tensor): Latent features [B, 128, H/4, W/4].

        Returns:
            torch.Tensor: Detail residual [B, 3, H, W].
        """
        edge = self.edge_net(x)  # [B, 1, H/4, W/4]
        detail = self.detail_net(torch.cat([x, edge], dim=1))  # [B, 3, H/4, W/4]

        # Upsample to full resolution
        detail = F.interpolate(
            detail, scale_factor=4, mode="bilinear", align_corners=False
        )
        return detail


class DetailRefinementHead(nn.Module):
    """Final detail refinement: Conv + 4×ResBlock + Conv.

    Predicts high-frequency residual at full resolution.
    """

    def __init__(self, in_channels=67):
        super().__init__()
        # Input: predicted RGB (3) + low-freq features (64) = 67
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, pred, features):
        """
        Args:
            pred: [B, 3, H, W] current prediction
            features: [B, 64, H, W] low-freq features
        Returns:
            residual: [B, 3, H, W] detail residual
        """
        x = torch.cat([pred, features], dim=1)
        return self.net(x)


class FrequencyAwareDecoder(nn.Module):
    """Complete frequency-aware decoder.

    Combines low-frequency reconstruction, high-frequency detail,
    and a refinement head with Laplacian residual merge.
    """

    def __init__(self, in_channels=128):
        super().__init__()
        self.low_freq = LowFrequencyBranch(in_channels)
        self.high_freq = HighFrequencyBranch(in_channels)
        self.detail_refine = DetailRefinementHead(in_channels=67)  # 3 + 64

    def forward(
        self, x: torch.Tensor, skips: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Decode latent features into a reconstructed frame using Laplacian merge and U-Net skips.

        Args:
            x (torch.Tensor): Transformer-fused features [B, 128, H/4, W/4].
            skips (Tuple): Warped encoder features [(B,64,H/2,W/2), (B,32,H,W)].

        Returns:
            torch.Tensor: Reconstructed output frame [B, 3, H, W].
        """
        # Low-frequency branch with skip fusion
        low_freq, lf_features = self.low_freq(x, skips)  # [B,3,H,W], [B,64,H,W]

        # High-frequency branch
        high_freq = self.high_freq(x)  # [B,3,H,W]

        # Combine: Laplacian residual merge
        pred = low_freq + high_freq

        # Detail refinement
        detail_residual = self.detail_refine(pred, lf_features)
        pred = pred + detail_residual

        # Clamp to valid range
        pred = pred.clamp(0.0, 1.0)

        return pred
