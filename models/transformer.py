"""
GMTI-Net Transformer Fusion

Motion-guided transformer with flow-derived attention offsets.
6 blocks, 8 heads, dim=128, MLP ratio=4.

Attention: softmax(QK^T / sqrt(d)) V
Motion-guided: offsets from flow influence key/value sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from typing import Optional, Tuple


class MotionGuidedWindowAttention(nn.Module):
    """
    Multi-head window attention with motion-guided positional sampling.

    Attributes:
        dim (int): Input feature dimension.
        window_size (int): Size of the attention window.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, dim, window_size=8, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Flow-to-offset projection
        self.offset_proj = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_heads),
        )

    def forward(
        self, x: torch.Tensor, flow_offset: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional motion-guided bias.

        Args:
            x (torch.Tensor): Windowed features [B*num_windows, N, C].
            flow_offset (Optional[torch.Tensor]): Flow-derived offsets [B*num_windows, N, 2].

        Returns:
            torch.Tensor: Attentive features of shape [B*num_windows, N, C].
        """
        B_N, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_N, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each [B_N, heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B_N, heads, N, N]

        # Add motion-guided bias
        if flow_offset is not None:
            motion_bias = self.offset_proj(flow_offset)  # [B_N, N, heads]
            motion_bias = motion_bias.permute(0, 2, 1).unsqueeze(
                -1
            )  # [B_N, heads, N, 1]
            attn = attn + motion_bias

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_N, N, C)
        out = self.proj(out)
        return out


class TransformerBlock(nn.Module):
    """Single transformer block: (Shifted) window attention + FFN with pre-norm."""

    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MotionGuidedWindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x, flow=None):
        """
        x: [B, H, W, C]
        flow: [B, H, W, 2] optional flow guidance
        """
        H, W = x.shape[1], x.shape[2]
        shortcut = x
        x = self.norm1(x)

        # 1) Shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            if flow is not None:
                shifted_flow = torch.roll(
                    flow, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
                )
            else:
                shifted_flow = None
        else:
            shifted_x = x
            shifted_flow = flow

        # 2) Partition
        x_windows = window_partition(shifted_x, self.window_size)
        if shifted_flow is not None:
            flow_windows = window_partition(shifted_flow, self.window_size)
        else:
            flow_windows = None

        # 3) Window Attention
        # Note: Cyclic shift without attention mask is used for brevity, acceptable for continuous images
        attn_windows = self.attn(x_windows, flow_windows)

        # 4) Reverse Partition
        shifted_x_out = window_reverse(attn_windows, self.window_size, H, W)

        # 5) Reverse Shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x_out

        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (B*num_windows, window_size*window_size, C)
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class TransformerFusion(nn.Module):
    """
    Motion-guided transformer fusion module with Alternating Shifted-Window Attention.

    Attributes:
        in_channels (int): Input feature channels.
        embed_dim (int): Embedding dimension.
        num_blocks (int): Number of transformer blocks.
    """

    def __init__(
        self,
        in_channels=96,
        embed_dim=128,
        num_blocks=6,
        num_heads=8,
        window_size=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.window_size = window_size

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, embed_dim, 1)

        # Transformer blocks with alternating shifts
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(num_blocks)
            ]
        )

        # Output projection
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(
        self, x: torch.Tensor, flow: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Execute transformer-based feature fusion.

        Args:
            x (torch.Tensor): Fused features [B, C, H, W].
            flow (Optional[torch.Tensor]): Full-resolution flow for guidance.

        Returns:
            torch.Tensor: Refined features of shape [B, 128, H, W].
        """
        B, C, H, W = x.shape

        # Extend boundaries to be perfectly divisible by window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))
            Hp, Wp = H + pad_b, W + pad_r
        else:
            Hp, Wp = H, W

        # Map to embed dim [B, 128, Hp, Wp]
        x = self.input_proj(x)

        # Channels-last [B, Hp, Wp, 128]
        x = x.permute(0, 2, 3, 1)

        # Resize flow if provided [B, Hp, Wp, 2]
        flow_trans = None
        if flow is not None:
            flow_resized = F.interpolate(
                flow, size=(Hp, Wp), mode="bilinear", align_corners=False
            )
            flow_trans = flow_resized.permute(0, 2, 3, 1)

        # Iterate Blocks linearly on H/W tensor using checkpointing
        for blk in self.blocks:
            if self.training and torch.is_grad_enabled():
                x = cp.checkpoint(blk, x, flow_trans, use_reentrant=False)
            else:
                x = blk(x, flow_trans)

        # Normalization
        x = self.norm(x)

        # Convert back [B, 128, Hp, Wp]
        out = x.permute(0, 3, 1, 2).contiguous()

        # Re-crop if padded
        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :H, :W].contiguous()

        out = self.output_proj(out)
        return out
