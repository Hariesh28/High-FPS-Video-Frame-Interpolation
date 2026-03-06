"""
tests/test_occlusion_eval_train.py

Verifies that OcclusionNetwork respects the train/eval mask semantics:

  • Eval mode  → always returns the learned CNN mask only (geometry ignored).
  • Train mode + geom_mask=ones  → output equals learned mask (mask cancels).
  • Train mode + geom_mask=zeros → output is all zeros (mask blocks everything).
  • train output ≠ eval output when geom_mask differs from all-ones.
"""

import torch
import pytest
from models.occlusion import OcclusionNetwork


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_inputs(feat_channels=96, H=16, W=16):
    wL = torch.rand(1, feat_channels, H, W)
    wR = torch.rand(1, feat_channels, H, W)
    return wL, wR


# ── tests ─────────────────────────────────────────────────────────────────────


def test_eval_ignores_geom_mask():
    """In eval mode the geom mask must have no effect on the output."""
    net = OcclusionNetwork(feat_channels=96)
    net.eval()
    wL, wR = _make_inputs()

    with torch.no_grad():
        out_no_mask = net(wL, wR)
        out_ones = net(wL, wR, training_geom_mask=torch.ones_like(out_no_mask))
        out_zeros = net(wL, wR, training_geom_mask=torch.zeros_like(out_no_mask))

    assert torch.allclose(
        out_no_mask, out_ones, atol=1e-6
    ), "Eval: all-ones geom mask should not change output"
    assert torch.allclose(
        out_no_mask, out_zeros, atol=1e-6
    ), "Eval: all-zeros geom mask should not change output (ignored in eval)"


def test_train_geom_ones_equals_eval():
    """Train mode with all-ones geom mask must equal eval output."""
    net = OcclusionNetwork(feat_channels=96)
    wL, wR = _make_inputs()

    net.eval()
    with torch.no_grad():
        eval_out = net(wL, wR)

    net.train()
    geom_ones = torch.ones_like(eval_out)
    train_out = net(wL, wR, training_geom_mask=geom_ones)

    assert torch.allclose(
        train_out, eval_out, atol=1e-6
    ), "Train + all-ones geom mask should equal eval output"


def test_train_geom_zeros_gives_zero_output():
    """Train mode with all-zeros geom mask must produce all-zero output."""
    net = OcclusionNetwork(feat_channels=96)
    net.train()
    wL, wR = _make_inputs()

    geom_zeros = torch.zeros(1, 1, 16, 16)
    out = net(wL, wR, training_geom_mask=geom_zeros)

    assert torch.allclose(
        out, torch.zeros_like(out), atol=1e-7
    ), f"Train + all-zeros geom mask should zero the output. Max: {out.abs().max().item():.2e}"


def test_train_without_geom_mask_equals_learned_mask():
    """Train mode without geom mask falls back to the learned-only mask."""
    net = OcclusionNetwork(feat_channels=96)
    wL, wR = _make_inputs()

    net.eval()
    with torch.no_grad():
        eval_out = net(wL, wR)

    net.train()
    train_out = net(wL, wR, training_geom_mask=None)

    assert torch.allclose(
        train_out, eval_out, atol=1e-6
    ), "Train without geom mask should equal learned mask (same as eval)"


def test_geom_mask_resized_if_needed():
    """Geom mask at different resolution should be resized and applied correctly."""
    net = OcclusionNetwork(feat_channels=48)
    net.train()

    H, W = 32, 32
    wL = torch.rand(1, 48, H, W)
    wR = torch.rand(1, 48, H, W)

    # Provide geom mask at 2× resolution — module should resize down
    geom_big = torch.ones(1, 1, H * 2, W * 2)
    out_with = net(wL, wR, training_geom_mask=geom_big)

    net.eval()
    with torch.no_grad():
        out_eval = net(wL, wR)

    assert torch.allclose(
        out_with, out_eval, atol=1e-6
    ), "Upsampled all-ones geom mask should not alter the learned mask"


if __name__ == "__main__":
    test_eval_ignores_geom_mask()
    test_train_geom_ones_equals_eval()
    test_train_geom_zeros_gives_zero_output()
    test_train_without_geom_mask_equals_learned_mask()
    test_geom_mask_resized_if_needed()
    print("All occlusion tests passed.")
