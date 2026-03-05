import os
import shutil
from types import SimpleNamespace
from pathlib import Path
from PIL import Image
import torch

from train import train, seed_everything


def create_dummy_dataset(root: Path, num_videos=2, frames_per_video=5, size=(32, 32)):
    root.mkdir(parents=True, exist_ok=True)
    for v in range(1, num_videos + 1):
        vid = root / f"vid_{v}"
        vid.mkdir(parents=True, exist_ok=True)
        for i in range(frames_per_video):
            img = Image.new(
                "RGB", size, color=(int(255 * (i / frames_per_video)), 0, 0)
            )
            fname = vid / f"frame_{i:06d}.png"
            img.save(str(fname))


def make_config(tmp_path):
    base = tmp_path
    data = {
        "train_dir": str(base / "train"),
        "val_dir": str(base / "val"),
        "crop_size": 16,
        "num_workers": 0,
    }

    training = {
        "batch_size": 1,
        "lr": 1e-4,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 1e-4,
        "max_iters": 2,
        "warmup_iters": 0,
        "final_lr": 1e-6,
        "ema_decay": 0.999,
        "amp": False,
        "val_freq": 10,
        "checkpoint_freq": 1,
        "clip_norm": 1.0,
    }

    model = {
        "encoder": {
            "swin_depth": 2,
            "swin_heads": 4,
            "swin_window_size": 8,
            "swin_mlp_ratio": 4.0,
        },
        "flow": {"refinement_iters": 2, "use_deformable": False},
        "transformer": {"blocks": 2, "heads": 4, "embed_dim": 64, "mlp_ratio": 4.0},
    }

    loss = {
        "charbonnier": 1.0,
        "laplacian": 1.0,
        "warping": 1.0,
        "bidirectional": 1.0,
        "smoothness": 1.0,
        "charbonnier_eps": 1e-3,
    }
    multiscale = {"scales": [1.0], "weights": [1.0]}

    cfg = {
        "data": data,
        "training": training,
        "model": model,
        "loss": loss,
        "multiscale": multiscale,
        "inference": {"multiscale": [1.0]},
    }

    return cfg


def test_smoke_train_and_resume(tmp_path):
    # Prepare dataset
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    create_dummy_dataset(train_dir, num_videos=2, frames_per_video=5)
    create_dummy_dataset(val_dir, num_videos=1, frames_per_video=4)

    cfg = make_config(tmp_path)

    # ensure clean checkpoints
    cp_dir = Path("checkpoints")
    if cp_dir.exists():
        shutil.rmtree(cp_dir)

    # Seed
    seed_everything(123)

    # First run: short training (max_iters=2)
    args = SimpleNamespace(max_iters=cfg["training"]["max_iters"], resume=None)
    train(cfg, args)

    # Check that checkpoints exist
    assert cp_dir.exists(), "checkpoints directory not created"
    latest = cp_dir / "latest.pth"
    assert latest.exists(), "latest.pth not found"

    # Use weights_only=False to load full checkpoint contents (rng, optimizer, etc.)
    ckpt = torch.load(str(latest), map_location="cpu", weights_only=False)
    assert "model" in ckpt and "iteration" in ckpt
    iter1 = ckpt["iteration"]

    # Resume run: continue to two more iterations
    args2 = SimpleNamespace(max_iters=iter1 + 2, resume=str(latest))
    train(cfg, args2)

    # Check updated latest
    ckpt2 = torch.load(
        str(cp_dir / "latest.pth"), map_location="cpu", weights_only=False
    )
    assert ckpt2["iteration"] >= iter1

    # Clean up
    shutil.rmtree(cp_dir)
