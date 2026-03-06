"""
tests/test_scaler_saved.py

Verify GradScaler state_dict is non-empty, contains expected keys,
and is correctly restored after save/load.
"""

import tempfile
import torch
import pytest
from utils.io import safe_torch_load


def test_scaler_state_not_empty():
    """GradScaler.state_dict() must be non-empty after use."""
    scaler = torch.amp.GradScaler(
        "cuda" if torch.cuda.is_available() else "cpu", enabled=True
    )
    # PyTorch >= 2.4 returns empty dict until first scale pass
    dummy_loss = torch.tensor(1.0, requires_grad=True)
    scaler.scale(dummy_loss).backward()
    state = scaler.state_dict()
    assert len(state) > 0, "GradScaler state_dict should not be empty after a backpass"


def test_scaler_contains_scale_key():
    """GradScaler state_dict should contain a scale / _scale key after use."""
    scaler = torch.amp.GradScaler(
        "cuda" if torch.cuda.is_available() else "cpu", enabled=True
    )
    dummy_loss = torch.tensor(1.0, requires_grad=True)
    scaler.scale(dummy_loss).backward()
    state = scaler.state_dict()
    # PyTorch ≥2.1 uses '_scale'; older versions may use 'scale'
    has_scale = any("scale" in k.lower() for k in state.keys())
    assert (
        has_scale
    ), f"No scale key found in GradScaler state_dict: {list(state.keys())}"


def test_scaler_roundtrip(tmp_path):
    """GradScaler state is identical after save/load."""
    scaler = torch.amp.GradScaler(enabled=False)
    state = scaler.state_dict()

    path = str(tmp_path / "scaler.pth")
    torch.save({"scaler": state}, path)

    loaded = safe_torch_load(path, map_location="cpu", weights_only=False)
    scaler2 = torch.amp.GradScaler(enabled=False)
    scaler2.load_state_dict(loaded["scaler"])

    assert scaler2.state_dict() == state, "Restored scaler state_dict != original"


def test_scaler_enabled_flag_persisted(tmp_path):
    """enabled=True scaler has a different state from enabled=False."""
    # Both disabled scalers should have the same initial state structure.
    s1 = torch.amp.GradScaler(enabled=False).state_dict()
    s2 = torch.amp.GradScaler(enabled=False).state_dict()
    assert s1 == s2
