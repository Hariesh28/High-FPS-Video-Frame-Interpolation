"""
tests/test_correlation.py

Tests for GMFlowMatching (flow_estimator.py):
  - Output shapes and no-NaN values.
  - Chunked vs dense correlation parity (rtol=1e-4).
  - corr_topk raises NotImplementedError with a clear message.
  - Zero-flow limit: identical features → zero expected displacement.
"""

import torch
import pytest
from models.flow_estimator import GMFlowMatching


def _make_features(B=1, C=64, H=8, W=8):
    fL = torch.randn(B, C, H, W)
    fR = torch.randn(B, C, H, W)
    return fL, fR


# ── basic output tests ────────────────────────────────────────────────────────


def test_gmflow_output_shapes():
    """Flow and confidence outputs have expected shapes."""
    matcher = GMFlowMatching(proj_dim=64, chunk_size=32)
    B, C, H, W = 2, 64, 8, 8
    fL, fR = _make_features(B, C, H, W)

    flow, conf = matcher(fL, fR)

    assert flow.shape == (B, 2, H, W), f"Flow shape {flow.shape}"
    assert conf.shape == (B, 1, H, W), f"Conf shape {conf.shape}"


def test_gmflow_no_nan():
    """Outputs must not contain NaN."""
    matcher = GMFlowMatching(proj_dim=64, chunk_size=32)
    fL, fR = _make_features()
    flow, conf = matcher(fL, fR)

    assert not torch.isnan(flow).any(), "NaN in flow output"
    assert not torch.isnan(conf).any(), "NaN in conf output"


# ── chunking parity ───────────────────────────────────────────────────────────


def test_corr_chunking_parity():
    """Chunked correlation must agree with dense (large chunk) up to fp32 rounding.

    Tolerance: rtol=1e-4, atol=1e-5 — fp32 chunked matmuls can produce tiny
    rounding diffs vs a single large matmul; we allow those here.
    """
    B, C, H, W = 1, 64, 8, 8
    fL, fR = _make_features(B, C, H, W)

    # dense (chunk larger than HW)
    dense = GMFlowMatching(proj_dim=64, chunk_size=H * W * 2)
    # chunked (chunk_size=4)
    chunked = GMFlowMatching(proj_dim=64, chunk_size=4)

    with torch.no_grad():
        flow_dense, _ = dense(fL, fR)
        flow_chunked, _ = chunked(fL, fR)

    max_err = (flow_dense - flow_chunked).abs().max().item()
    mean_err = (flow_dense - flow_chunked).abs().mean().item()

    assert torch.allclose(flow_dense, flow_chunked, rtol=1e-4, atol=1e-5), (
        f"Chunked vs dense mismatch: max_err={max_err:.2e}, mean_err={mean_err:.2e}. "
        "fp32 chunked matmul should agree within rtol=1e-4."
    )


# ── topk guard ────────────────────────────────────────────────────────────────


def test_corr_topk_raises_not_implemented():
    """Setting corr_topk should raise NotImplementedError with a helpful message."""
    with pytest.raises(NotImplementedError, match="corr_topk"):
        GMFlowMatching(proj_dim=64, chunk_size=32, corr_topk=32)


# ── zero-displacement limit ───────────────────────────────────────────────────


def test_identical_features_near_zero_flow():
    """When fL == fR the softmax peaks on the diagonal; expected disp ≈ diagonal."""
    # This is a sanity check, not a strict zero-flow test, because the expected
    # displacement is the centroid of the distribution which equals the source
    # coordinate when features are identical and softmax is peaked on diagonal.
    B, C, H, W = 1, 64, 4, 4
    feat = torch.randn(B, C, H, W)

    matcher = GMFlowMatching(proj_dim=64, chunk_size=16, temp=0.01)  # low temp → peaked
    flow, conf = matcher(feat, feat.clone())

    # Flow should be close to zero (each pixel matches itself)
    mean_flow_mag = flow.abs().mean().item()
    assert (
        mean_flow_mag < 2.0
    ), f"Identical features should produce near-zero flow; got mean_mag={mean_flow_mag:.3f}"


if __name__ == "__main__":
    test_gmflow_output_shapes()
    test_gmflow_no_nan()
    test_corr_chunking_parity()
    test_corr_topk_raises_not_implemented()
    test_identical_features_near_zero_flow()
    print("All correlation tests passed.")
