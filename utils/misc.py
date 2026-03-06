# utils/misc.py
"""
Shared training utilities: seeding, git revision, and environment logging.

These functions are imported by train.py and any other script that needs
reproducibility or run-tracking helpers.  They are extracted from train.py
so that validate.py and inference.py can reuse them without circular deps.
"""

import os
import sys
import json
import logging
import random
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch

logger = logging.getLogger("utils.misc")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def seed_everything(seed: int = 2026, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed:          Integer seed for all RNGs.
        deterministic: If True, engage cudnn deterministic mode and disable
                       benchmark autotuning (~10-20% throughput reduction).

    .. note::
        Even with ``deterministic=True`` a small number of cuDNN kernels use
        non-deterministic algorithms (e.g., atomicAdd-based reductions).  Use
        this flag for debugging and final reproduction; keep it off for
        large-scale training to maintain throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def make_worker_init_fn(base_seed: int):
    """Return a DataLoader ``worker_init_fn`` that seeds each worker uniquely.

    Each worker gets ``seed = base_seed + worker_id`` so that data augmentation
    across workers is independent but reproducible when ``base_seed`` is fixed.
    This is especially important when ``--deterministic`` is used.
    """

    def _worker_init(worker_id: int) -> None:
        worker_seed = int(base_seed) + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _worker_init


# ---------------------------------------------------------------------------
# Git revision
# ---------------------------------------------------------------------------


def _get_git_rev() -> str:
    """Return the current git HEAD commit hash, or 'unknown' on failure."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Environment logging
# ---------------------------------------------------------------------------


def log_environment(writer=None, run_dir: str | None = None) -> dict:
    """Log environment information to stdout, TensorBoard, and run_info.json.

    Args:
        writer:  Optional SummaryWriter; if provided, adds an ``add_text`` entry.
        run_dir: Optional directory; if provided, writes ``run_info.json`` there.

    Returns:
        A dict with all logged fields (date, python, torch, cuda, …).
    """
    try:
        import torchvision

        tv_version = torchvision.__version__
    except ImportError:
        tv_version = "not installed"

    info: dict = {
        "date": datetime.now(timezone.utc).isoformat(),
        "python": "{}.{}.{}".format(*sys.version_info[:3]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "N/A",
        "cudnn": (
            str(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else "N/A"
        ),
        "torchvision": tv_version,
        "git_rev": _get_git_rev(),
    }

    # ---- stdout ----
    print("=" * 60)
    print("  Environment")
    for k, v in info.items():
        print(f"  {k:<14}: {v}")
    print("=" * 60)

    # ---- TensorBoard ----
    if writer is not None:
        try:
            env_md = "\n".join(f"**{k}**: {v}" for k, v in info.items())
            writer.add_text("environment", env_md, global_step=0)
        except Exception as exc:
            logger.warning("TensorBoard add_text failed: %s", exc)

    # ---- JSON ----
    if run_dir is not None:
        try:
            os.makedirs(run_dir, exist_ok=True)
            json_path = os.path.join(run_dir, "run_info.json")
            with open(json_path, "w") as fh:
                json.dump(info, fh, indent=2)
            print(f"[misc] Environment info saved to {json_path}")
        except Exception as exc:
            logger.warning("Failed to write run_info.json: %s", exc)

    return info
