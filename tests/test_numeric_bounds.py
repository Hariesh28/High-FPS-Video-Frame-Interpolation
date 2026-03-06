import torch
import pytest
import os
from models.flow_estimator import GMFlowMatching, ConvexMaskUpsample, FlowEstimator
from models.occlusion import OcclusionNetwork
from torch.amp import GradScaler


def test_corr_chunking():
    """Verify chunked processing equates exactly to non-chunked outputs.

    Tolerance: rtol=1e-4, atol=1e-5 — fp32 chunked matmuls can produce tiny
    rounding diffs vs a single large matmul; we allow those here.
    """
    chunked_corr = GMFlowMatching(proj_dim=128, chunk_size=4)
    standard_corr = GMFlowMatching(proj_dim=128, chunk_size=1024)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1, 128, 4, 4 -> HW = 16
    fL = torch.randn(1, 128, 4, 4).to(device)
    fR = torch.randn(1, 128, 4, 4).to(device)

    flow_chunk, conf_chunk = chunked_corr(fL, fR)
    flow_std, conf_std = standard_corr(fL, fR)

    max_err = (flow_chunk - flow_std).abs().max().item()
    mean_err = (flow_chunk - flow_std).abs().mean().item()
    assert torch.allclose(flow_chunk, flow_std, rtol=1e-4, atol=1e-5), (
        f"Chunked vs dense correlation mismatch: max_err={max_err:.2e}, mean_err={mean_err:.2e}. "
        "fp32 chunked matmul should agree within rtol=1e-4."
    )


def test_convex_mask_channels():
    """Verify channel formulas."""
    kernel = 3
    upscale = 2
    convex = ConvexMaskUpsample(128, kernel=kernel, upscale=upscale)
    assert convex.mask_ch == (kernel**2) * (
        upscale**2
    ), "Subpixel mask alignment channel count mismatched."


def test_occlusion_training_inference():
    """Check geometric limits exist exclusively during training states."""
    occ = OcclusionNetwork(feat_channels=32)
    warp_L = torch.randn(1, 32, 64, 64)
    warp_R = torch.randn(1, 32, 64, 64)
    flow_lr = torch.randn(1, 2, 64, 64) * 0.1
    flow_rl = torch.randn(1, 2, 64, 64) * -0.1

    occ.train()
    mask_train = occ(warp_L, warp_R, flow_lr, flow_rl)

    occ.eval()
    mask_eval = occ(warp_L, warp_R, flow_lr, flow_rl)

    assert not torch.allclose(
        mask_train, mask_eval
    ), "Geom mask failed to deactivate across eval modes."


def test_resume_checkpoint():
    """Evaluate dict loading for scalar/optimizer artifacts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowEstimator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Simulate step
    opt.step()
    scaler.update()

    torch.save(
        {"optimizer": opt.state_dict(), "scaler": scaler.state_dict()},
        "temp_checkpoint.pth",
    )

    # Reload
    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler2 = GradScaler(enabled=torch.cuda.is_available())

    from utils.io import safe_torch_load

    ckpt = safe_torch_load(
        "temp_checkpoint.pth", map_location=device, weights_only=False
    )
    opt2.load_state_dict(ckpt["optimizer"])
    scaler2.load_state_dict(ckpt["scaler"])

    assert opt2.param_groups[0]["lr"] == opt.param_groups[0]["lr"]

    os.remove("temp_checkpoint.pth")
