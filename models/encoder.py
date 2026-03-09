"""
GMTI-Net Encoder: Hybrid CNN + Swin Transformer
5-stage feature pyramid extractor.

Stage 1: Conv(3→32) + 2×ResBlock         → [B,32,H,W]
Stage 2: Conv(32→64,s2) + 2×ResBlock      → [B,64,H/2,W/2]
Stage 3: Conv(64→96,s2) + 2×ResBlock      → [B,96,H/4,W/4]
Stage 4: Conv(96→128,s2) + SwinTransformer → [B,128,H/8,W/8]
Stage 5: PatchMerge                        → [B,160,H/16,W/16]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import List, Tuple, Optional, Union


class ResidualBlock(nn.Module):
    """
    Standard ResBlock: Conv3x3 → GroupNorm → GELU → Conv3x3 → GroupNorm.

    Attributes:
        channels (int): Number of input and output channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Residual-enhanced features.
        """
        residual = x
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act(out + residual)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows.

    Args:
        x (torch.Tensor): Input tensor of shape [B, H, W, C].
        window_size (int): Size of the window.

    Returns:
        torch.Tensor: Windows of shape [B*num_windows, window_size, window_size, C].
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: int, H: int, W: int
) -> torch.Tensor:
    """
    Reverse window partition.

    Args:
        windows (torch.Tensor): Windows of shape [B*num_windows, window_size, window_size, C].
        window_size (int): Size of the window.
        H (int): Original height.
        W (int): Original width.

    Returns:
        torch.Tensor: Reconstructed feature map of shape [B, H, W, C].
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # [2, ws, ws]
        coords_flatten = torch.flatten(coords, 1)  # [2, ws*ws]
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # [2, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 2]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [N, N]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: [num_windows*B, N, C] where N = window_size^2
            mask: optional attention mask
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with window attention and shifted window attention."""

    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x, H, W):
        """
        Args:
            x: [B, H*W, C]
            H, W: spatial dimensions
        """
        B, L, C = x.shape
        assert L == H * W, f"Input length {L} != H*W {H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad to multiple of window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x.shape[1], x.shape[2]

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            # Compute attention mask for shifted windows
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_x = x
            attn_mask = None

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Reverse window partition
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerge(nn.Module):
    """Patch merging layer: 2× spatial downsampling + channel increase.

    Takes adjacent 2×2 patches and concatenates then projects.
    Input:  [B, C, H, W]
    Output: [B, 2*C - adjustment, H/2, W/2]
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduction = nn.Linear(4 * in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(4 * in_channels)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, out_channels, H/2, W/2]
        """
        B, C, H, W = x.shape
        # Pad if needed
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = x.shape[2], x.shape[3]

        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2, W/2, out_channels]
        x = x.permute(0, 3, 1, 2)  # [B, out_channels, H/2, W/2]
        return x


class HybridEncoder(nn.Module):
    """
    Hybrid CNN + Swin Transformer encoder.

    Returns multi-scale features at 5 levels:
        f1: [B, 32, H, W]
        f2: [B, 64, H/2, W/2]
        f3: [B, 96, H/4, W/4]
        f4: [B, 128, H/8, W/8]
        f5: [B, 160, H/16, W/16]
    """

    def __init__(
        self,
        in_channels: int = 3,
        stage_channels: Tuple[int, ...] = (32, 64, 96, 128, 160),
        swin_depth: int = 4,
        swin_heads: int = 6,
        swin_window_size: int = 8,
        swin_mlp_ratio: float = 4.0,
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = stage_channels

        # Stage 1: full resolution
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, c1),
            nn.GELU(),
            ResidualBlock(c1),
            ResidualBlock(c1),
        )

        # Stage 2: 1/2 resolution
        self.down2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False)
        self.stage2 = nn.Sequential(
            nn.GroupNorm(8, c2),
            nn.GELU(),
            ResidualBlock(c2),
            ResidualBlock(c2),
        )

        # Stage 3: 1/4 resolution
        self.down3 = nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False)
        self.stage3 = nn.Sequential(
            nn.GroupNorm(8, c3),
            nn.GELU(),
            ResidualBlock(c3),
            ResidualBlock(c3),
        )

        # Stage 4: 1/8 resolution — Swin Transformer
        self.down4 = nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=False)
        self.norm4_pre = nn.GroupNorm(8, c4)
        self.swin_blocks = nn.ModuleList()
        for i in range(swin_depth):
            shift = 0 if (i % 2 == 0) else swin_window_size // 2
            self.swin_blocks.append(
                SwinTransformerBlock(
                    dim=c4,
                    num_heads=swin_heads,
                    window_size=swin_window_size,
                    shift_size=shift,
                    mlp_ratio=swin_mlp_ratio,
                )
            )

        # Stage 5: 1/16 resolution — Patch merge
        self.patch_merge = PatchMerge(c4, c5)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale feature pyramid.

        Args:
            x (torch.Tensor): Input frames of shape [B, 3, H, W].

        Returns:
            List[torch.Tensor]: List of five feature tensors [f1, f2, f3, f4, f5].
        """
        # Stage 1
        f1 = self.stage1(x)  # [B, 32, H, W]

        # Stage 2
        f2 = self.down2(f1)
        f2 = self.stage2(f2)  # [B, 64, H/2, W/2]

        # Stage 3
        f3 = self.down3(f2)
        f3 = self.stage3(f3)  # [B, 96, H/4, W/4]

        # Stage 4 — Swin Transformer
        f4 = self.down4(f3)
        f4 = self.norm4_pre(f4)  # [B, 128, H/8, W/8]
        B, C, H4, W4 = f4.shape
        f4_seq = f4.flatten(2).transpose(1, 2)  # [B, H4*W4, 128]
        for blk in self.swin_blocks:
            f4_seq = blk(f4_seq, H4, W4)
        f4 = f4_seq.transpose(1, 2).view(B, C, H4, W4)  # [B, 128, H/8, W/8]

        # Stage 5 — Patch merge
        f5 = self.patch_merge(f4)  # [B, 160, H/16, W/16]

        return [f1, f2, f3, f4, f5]
