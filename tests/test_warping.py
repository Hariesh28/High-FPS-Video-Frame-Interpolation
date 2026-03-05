import torch
from models.warping import backward_warp


def test_backward_warp():
    B, C, H, W = 2, 3, 64, 64
    img = torch.randn(B, C, H, W)
    flow = torch.randn(B, 2, H, W) * 5.0  # Random flows

    warped = backward_warp(img, flow)
    assert warped.shape == (B, C, H, W)
    assert not torch.isnan(warped).any()


if __name__ == "__main__":
    test_backward_warp()
    print("test_backward_warp passed")
