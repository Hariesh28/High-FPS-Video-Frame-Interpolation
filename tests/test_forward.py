import torch
from models.gmti_net import GMTINet


def test_forward_pass():
    model = GMTINet(
        swin_depth=2,
        swin_heads=4,
        swin_window_size=8,
        swin_mlp_ratio=4.0,
        flow_refinement_iters=2,
        use_deformable=False,  # test standard fallback
        transformer_blocks=2,
        transformer_heads=4,
        transformer_dim=64,
        transformer_mlp_ratio=4.0,
    )

    B, C, H, W = 2, 3, 128, 128
    L = torch.rand(B, C, H, W)
    R = torch.rand(B, C, H, W)

    with torch.no_grad():
        out, aux = model(L, R)

    assert out.shape == (B, C, H, W)
    assert "flow_lr" in aux
    assert not torch.isnan(out).any()


if __name__ == "__main__":
    test_forward_pass()
    print("test_forward_pass passed")
