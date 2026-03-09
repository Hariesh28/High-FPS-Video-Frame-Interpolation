"""Safe I/O helpers for checkpoint loading and saving.

Provides:
  safe_torch_load     — robust torch.load wrapper across PyTorch versions.
  extract_model_state — robust key lookup: 'ema' → 'model' → 'state_dict'.
  atomic_save         — write to tmp then os.replace to avoid partial files.
  prune_checkpoints   — delete old iter_*.pth beyond keep_last retention.
"""

import glob
import os
import re
import logging
import warnings
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def safe_torch_load(
    path: str, map_location=None, weights_only: Optional[bool] = None
) -> Any:
    """Load a checkpoint robustly across PyTorch versions.

    Args:
        path:         file path to load.
        map_location: passed to torch.load.
        weights_only: If True, request weights-only safe load.
                      If False, request full unpickle.
                      If None, prefer weights-only when available.

    Returns:
        The object returned by torch.load.

    Notes:
        On PyTorch versions that don't accept the weights_only kwarg this
        will fall back to the legacy torch.load signature.
    """
    if weights_only is None:
        weights_only = True

    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Older torch that doesn't accept weights_only
        return torch.load(path, map_location=map_location)


def extract_model_state(
    ckpt: dict,
    preferred_keys: tuple = ("ema", "model", "state_dict"),
    warn: bool = True,
) -> Optional[dict]:
    """Robustly extract model weights from a checkpoint dict.

    Tries keys in order: 'ema' → 'model' → 'state_dict'.
    A bare checkpoint (the dict IS the state_dict) is also handled.

    Args:
        ckpt:          The checkpoint dict returned by torch.load.
        preferred_keys: Ordered keys to try.
        warn:          If True, emit a warning (not an error) when no key found.

    Returns:
        An OrderedDict / dict of model parameters, or None if not found.
    """
    if not isinstance(ckpt, dict):
        warnings.warn(
            "extract_model_state: checkpoint is not a dict. "
            "Returning it as-is (may be a raw state_dict)."
        )
        return ckpt  # type: ignore[return-value]

    for key in preferred_keys:
        if key in ckpt:
            return ckpt[key]

    # Heuristic: if none of the known keys exist but the dict looks like a
    # state_dict (first value is a tensor), treat it as a direct state_dict.
    first_val = next(iter(ckpt.values()), None)
    if isinstance(first_val, torch.Tensor):
        if warn:
            warnings.warn(
                "extract_model_state: checkpoint appears to be a raw state_dict "
                "(no 'ema'/'model'/'state_dict' key). Using it directly."
            )
        return ckpt

    if warn:
        warnings.warn(
            f"extract_model_state: none of {preferred_keys} found in checkpoint keys "
            f"({list(ckpt.keys())}). Returning None."
        )
    return None


# ---------------------------------------------------------------------------
# Atomic checkpoint save
# ---------------------------------------------------------------------------


def atomic_save(obj: Any, path: str) -> None:
    """Save ``obj`` to a temporary file then atomically rename to ``path``."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        logger.error(f"Failed to save {path}: {e}")
        raise


# ---------------------------------------------------------------------------
# Checkpoint retention / pruning
# ---------------------------------------------------------------------------


def prune_checkpoints(ckpt_dir: str = "checkpoints", keep_last: int = 5) -> None:
    """Delete old numbered checkpoints beyond the keep-last retention window.

    Retention policy:
      - Always keep the ``keep_last`` most-recent ``iter_*.pth`` files.
      - Always preserve ``latest.pth``, ``best_ema.pth`` (untouched).
      - Numbered files are sorted by their iteration number (not file name,
        which is important for iterative naming like ``iter_10000.pth``).

    Args:
        ckpt_dir:  Directory that contains the checkpoint files.
        keep_last: Number of recent numbered checkpoints to retain.
    """
    pattern = os.path.join(ckpt_dir, "iter_*.pth")
    files = glob.glob(pattern)

    def _iter_num(fn: str) -> int:
        m = re.search(r"iter_(\d+)\.pth", os.path.basename(fn))
        return int(m.group(1)) if m else -1

    files_sorted = sorted(files, key=_iter_num)
    to_delete = files_sorted[:-keep_last] if len(files_sorted) > keep_last else []
    for path in to_delete:
        try:
            os.remove(path)
            logger.info(f"Pruned checkpoint: {os.path.basename(path)}")
        except OSError as exc:
            logger.warning(f"Could not prune {path}: {exc}")
