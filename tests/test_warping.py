"""
tests/test_warping.py

Tests for models/warping.py:
  - backward_warp returns correct shape, no NaN, values in range
  - Single-pixel-shift: a unit pixel at (10,10) shifted by (dx,dy)
    lands at the expected output location.
  - flow_to_grid: zero flow produces grid in [-1, 1] bounds.
"""

import torch
import pytest
from models.warping import backward_warp, flow_to_grid


def test_backward_warp_shape_and_no_nan():
    """backward_warp output matches input shape and contains no NaN."""
    B, C, H, W = 2, 3, 64, 64
    img = torch.randn(B, C, H, W)
    flow = torch.randn(B, 2, H, W) * 5.0

    warped = backward_warp(img, flow)

    assert warped.shape == (B, C, H, W), f"Shape mismatch: {warped.shape}"
    assert not torch.isnan(warped).any(), "NaN in warped output"


def test_backward_warp_single_pixel_shift():
    """A white pixel at (r,c) warped by tracking offset (dx,dy) lands at (r-dy, c-dx).

    Using backward grid_sample logic: output[y,x] pulls from input[y+dy, x+dx].
    So a constant offset grid of (dx=2, dy=3) pulls the source pixel
    from (row=10, col=10) and places it into output at (row=7, col=8).
    """
    B, C, H, W = 1, 1, 64, 64
    img = torch.zeros(B, C, H, W)
    img[0, 0, 10, 10] = 1.0

    dx, dy = 2, 3
    flow = torch.zeros(B, 2, H, W)
    flow[0, 0] = float(dx)  # channel 0 = x-displacement
    flow[0, 1] = float(dy)  # channel 1 = y-displacement

    warped = backward_warp(img, flow)

    # The warped value at the destination should be close to 1.0
    val_at_dest = warped[0, 0, 10 - dy, 10 - dx].item()
    assert (
        val_at_dest > 0.9
    ), f"Expected warped pixel > 0.9 at ({10-dy},{10-dx}), got {val_at_dest:.4f}"


def test_zero_flow_preserves_image():
    """Zero flow should produce warped output equal to the input."""
    B, C, H, W = 1, 3, 32, 32
    img = torch.rand(B, C, H, W)
    flow = torch.zeros(B, 2, H, W)

    warped = backward_warp(img, flow)

    assert torch.allclose(
        warped, img, atol=1e-5
    ), f"Zero-flow warp should be identity. Max diff: {(warped - img).abs().max().item():.2e}"


def test_flow_to_grid_normalization():
    """Zero flow produces a grid with all values in [-1, 1]."""
    B, H, W = 1, 8, 8
    flow = torch.zeros(B, 2, H, W)
    grid = flow_to_grid(flow)

    assert grid.shape == (B, H, W, 2), f"Grid shape mismatch: {grid.shape}"
    assert grid.min().item() >= -1.0 - 1e-6
    assert grid.max().item() <= 1.0 + 1e-6


def test_backward_warp_half_precision():
    """backward_warp should produce no NaN even with fp16 input."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for fp16 warp test")

    device = torch.device("cuda")
    B, C, H, W = 1, 3, 64, 64
    img = torch.rand(B, C, H, W, dtype=torch.float16, device=device)
    flow = torch.rand(B, 2, H, W, dtype=torch.float16, device=device) * 4.0

    warped = backward_warp(img, flow)

    assert warped.dtype == torch.float16, "Output dtype should match input (fp16)"
    assert not torch.isnan(warped).any(), "NaN in fp16 warped output"


if __name__ == "__main__":
    test_backward_warp_shape_and_no_nan()
    test_backward_warp_single_pixel_shift()
    test_zero_flow_preserves_image()
    test_flow_to_grid_normalization()
    print("All warping tests passed.")
