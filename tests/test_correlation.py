import torch
from models.flow_estimator import GMFlowMatching


def test_gmflow_matching():
    matcher = GMFlowMatching(chunk_size=1024)
    B, C, H, W = 2, 64, 32, 32
    featL = torch.randn(B, C, H, W)
    featR = torch.randn(B, C, H, W)

    flow, conf = matcher(featL, featR)

    assert flow.shape == (B, 2, H, W)
    assert conf.shape == (B, 1, H, W)
    assert not torch.isnan(flow).any()
    assert not torch.isnan(conf).any()


if __name__ == "__main__":
    test_gmflow_matching()
    print("test_gmflow_matching passed")
