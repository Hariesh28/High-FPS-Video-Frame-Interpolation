import torch
import os
from inference import average_checkpoints


def test_average_checkpoints_basic(tmp_path):
    # Create two fake checkpoints with simple model tensors
    ckpt1 = {"model": {"w": torch.tensor([1.0, 2.0])}}
    ckpt2 = {"model": {"w": torch.tensor([3.0, 4.0])}}

    p1 = tmp_path / "ckpt1.pth"
    p2 = tmp_path / "ckpt2.pth"
    torch.save(ckpt1, str(p1))
    torch.save(ckpt2, str(p2))

    avg = average_checkpoints([str(p1), str(p2)], device="cpu")

    assert "w" in avg
    # Expect element-wise average (1+3)/2, (2+4)/2
    expected = torch.tensor([2.0, 3.0])
    assert torch.allclose(avg["w"], expected)


def test_average_checkpoints_with_ema_key(tmp_path):
    # Support checkpoints storing under 'ema' instead of 'model'
    ckpt1 = {"ema": {"w": torch.tensor([10.0])}}
    ckpt2 = {"ema": {"w": torch.tensor([14.0])}}

    p1 = tmp_path / "ckpt1.pth"
    p2 = tmp_path / "ckpt2.pth"
    torch.save(ckpt1, str(p1))
    torch.save(ckpt2, str(p2))

    avg = average_checkpoints([str(p1), str(p2)], device="cpu")
    assert "w" in avg
    assert torch.allclose(avg["w"], torch.tensor([12.0]))
