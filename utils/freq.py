import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_dct_basis(block_size: int, device: torch.device):
    """Compute DCT-II basis matrices for a given block size."""
    basis = np.zeros((block_size, block_size))
    for i in range(block_size):
        for j in range(block_size):
            if i == 0:
                basis[i, j] = 1 / np.sqrt(block_size)
            else:
                basis[i, j] = np.sqrt(2 / block_size) * np.cos(
                    np.pi * i * (2 * j + 1) / (2 * block_size)
                )
    return torch.from_numpy(basis).float().to(device)


def block_dct(img: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """
    Compute block-wise 2D DCT on an image.

    Args:
        img (torch.Tensor): [B, C, H, W]
        block_size (int): Size of the DCT block (default: 8)

    Returns:
        torch.Tensor: [B, C, H/block_size, W/block_size, block_size, block_size]
    """
    B, C, H, W = img.shape
    device = img.device

    # Pad to block size if necessary
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
        H, W = img.shape[2:]

    basis = get_dct_basis(block_size, device)

    # Reshape for block processing: [B, C, H/b, b, W/b, b]
    x = img.view(B, C, H // block_size, block_size, W // block_size, block_size)

    # DCT along W axis: x * basis^T
    x = torch.matmul(x, basis.t())

    # DCT along H axis: basis * x
    # Permute to bring H dimension to last for matmul: [B, C, H/b, W/b, b, b]
    x = x.permute(0, 1, 2, 4, 3, 5)
    x = torch.matmul(basis, x)

    return x  # [B, C, H/b, W/b, block_size, block_size]


def get_hf_mask(block_size: int, device: torch.device, cutoff: int = 4):
    """Create a mask for high-frequency DCT coefficients."""
    mask = torch.zeros((block_size, block_size), device=device)
    for i in range(block_size):
        for j in range(block_size):
            if i + j >= cutoff:  # Zig-zag heuristic for HF
                mask[i, j] = 1.0
    return mask
