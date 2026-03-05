import torch
from models.warping import flow_to_grid, backward_warp


def test_flow_to_grid_and_warp():
    """Verify that a simple constant shift correctly shifts pixels."""
    B, C, H, W = 1, 1, 16, 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.zeros(B, C, H, W).to(device)

    # Add a white pixel exactly at coordinate [4, 4] -> y=4, x=4
    img[0, 0, 4, 4] = 1.0

    # Create flow shifting right by 2 -> dx=2, dy=0
    # Backward warp: source(x) = target(x + flow(x)).
    # So if target is looking at (6,4), it queries (6-dx, 4) = (4,4) finding the white pixel.
    flow = torch.zeros(B, 2, H, W).to(device)
    flow[:, 0, :, :] = -2.0  # flow towards -2 x
    flow[:, 1, :, :] = 0.0

    warped = backward_warp(img, flow)

    # Original was at 4,4. Shift is dx=-2 in source grid mapped.
    # The pixel will appear at x=6, y=4 in the target.
    # To be extremely precise, if flow[y,x] is the value pointing towards the source,
    # pixel at target (y=4, x=6) + flow (-2, 0) -> (4, 4).
    assert warped[0, 0, 4, 6] > 0.99, (
        "Affine shift warp reconstruction lost 1:1 mapping"
    )


def test_validity_mask():
    flow = torch.zeros(1, 2, 16, 16)
    flow[:, 0, :, :] = -100.0  # Force out of bounds
    grid = flow_to_grid(flow)

    # Because x is out of bounds, validity must mask this completely
    valid = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)
    assert not valid.any(), "Validity mask allowed OOB coordinate mapping."
