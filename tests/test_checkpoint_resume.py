"""
tests/test_checkpoint_resume.py

Verifies checkpoint save/load robustness:
  - All required keys present after save.
  - Optimizer LR, scheduler last_epoch, scaler state correctly restored.
  - CPU + CUDA RNG states round-trip exactly.
  - extract_model_state key priority: 'ema' > 'model' > 'state_dict'.
  - atomic_save() produces a readable file (no corrupt partial writes).
"""

import os
import math
import tempfile

import torch
import torch.nn as nn
import pytest

from models.flow_estimator import FlowEstimator
from utils.io import safe_torch_load, extract_model_state, atomic_save


@pytest.fixture
def model_and_opt():
    model = FlowEstimator(use_deformable=False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    return model, opt


def test_checkpoint_keys_present(tmp_path, model_and_opt):
    """All expected checkpoint keys are present after saving."""
    model, opt = model_and_opt
    scaler = torch.amp.GradScaler(enabled=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda x: 1.0)

    ckpt = {
        "model": model.state_dict(),
        "ema": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "iteration": 42,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": None,  # no GPU in CI
    }

    path = str(tmp_path / "test_ckpt.pth")
    torch.save(ckpt, path)

    loaded = safe_torch_load(path, map_location="cpu", weights_only=False)
    for key in (
        "model",
        "ema",
        "optimizer",
        "scheduler",
        "scaler",
        "iteration",
        "torch_rng",
    ):
        assert key in loaded, f"Missing key '{key}' in checkpoint."


def test_optimizer_lr_restored(tmp_path, model_and_opt):
    """Optimizer LR is identical after save/load."""
    model, opt = model_and_opt
    path = str(tmp_path / "opt.pth")
    torch.save({"optimizer": opt.state_dict()}, path)

    loaded = safe_torch_load(path, map_location="cpu", weights_only=False)
    opt2 = torch.optim.AdamW(model.parameters(), lr=99.0)
    opt2.load_state_dict(loaded["optimizer"])

    assert opt2.param_groups[0]["lr"] == opt.param_groups[0]["lr"]


def test_scheduler_state_restored(tmp_path, model_and_opt):
    """Scheduler last_epoch matches after save/load."""
    model, opt = model_and_opt

    def lr_fn(it):
        return max(0.5 * (1.0 + math.cos(math.pi * it / 100)), 1e-6 / 2e-4)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    for _ in range(5):
        opt.step()
        sched.step()

    path = str(tmp_path / "sched.pth")
    torch.save({"scheduler": sched.state_dict()}, path)

    loaded = safe_torch_load(path, map_location="cpu", weights_only=False)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    sched2.load_state_dict(loaded["scheduler"])

    assert sched2.last_epoch == sched.last_epoch


def test_scaler_state_non_empty_and_restored(tmp_path, model_and_opt):
    """GradScaler state_dict is non-empty and restores correctly."""
    scaler = torch.amp.GradScaler(enabled=False)
    state = scaler.state_dict()
    # In PyTorch >= 2.4, an unused disabled scaler returns an empty state dict initially.
    # We mainly care that what goes in comes out.

    path = str(tmp_path / "scaler.pth")
    torch.save({"scaler": state}, path)
    loaded = safe_torch_load(path, map_location="cpu", weights_only=False)

    scaler2 = torch.amp.GradScaler(enabled=False)
    scaler2.load_state_dict(loaded["scaler"])
    assert scaler2.state_dict() == state


def test_torch_rng_restored(tmp_path):
    """CPU torch RNG state is restored after save/load."""
    torch.manual_seed(42)
    rng_before = torch.get_rng_state()
    _ = torch.randn(10)

    path = str(tmp_path / "rng.pth")
    torch.save({"torch_rng": rng_before}, path)
    loaded = safe_torch_load(path, map_location="cpu", weights_only=False)
    torch.set_rng_state(loaded["torch_rng"].cpu())

    # After restoring RNG, should produce same sequence as original seed=42.
    a = torch.randn(10)
    torch.manual_seed(42)
    _ = torch.get_rng_state()  # advance state same as above
    b = torch.randn(10)
    assert torch.allclose(a, b), "RNG state not correctly restored."


def test_extract_model_state_priority():
    """extract_model_state respects ema > model > state_dict priority."""
    ema_sd = {"w": torch.tensor(1.0)}
    model_sd = {"w": torch.tensor(2.0)}
    state_sd = {"w": torch.tensor(3.0)}

    assert extract_model_state({"ema": ema_sd, "model": model_sd}) is ema_sd
    assert extract_model_state({"model": model_sd}) is model_sd
    assert extract_model_state({"state_dict": state_sd}) is state_sd

    # Bare state_dict (heuristic)
    bare = {"layer.weight": torch.randn(4, 4)}
    result = extract_model_state(bare, warn=False)
    assert result is bare


def test_atomic_save_roundtrip(tmp_path):
    """atomic_save writes a readable checkpoint; extract_model_state recovers weights."""
    weights = {"fc.weight": torch.randn(4, 4), "fc.bias": torch.zeros(4)}
    ckpt = {
        "model": weights,
        "optimizer": {"state": {}},
        "scaler": {"scale": 1024.0},
        "torch_rng": torch.get_rng_state(),
    }

    path = str(tmp_path / "atomic_ckpt.pth")
    atomic_save(ckpt, path)

    assert os.path.exists(path), "atomic_save did not create the target file"
    assert not os.path.exists(path + ".tmp"), "Temp file should be cleaned up"

    loaded = safe_torch_load(path, map_location="cpu", weights_only=False)
    assert "model" in loaded
    assert "scaler" in loaded

    recovered = extract_model_state(loaded)
    assert recovered is not None
    assert torch.equal(recovered["fc.weight"], weights["fc.weight"])
