"""Safe I/O helpers for checkpoint loading.

Provides safe_torch_load which tries to use torch.load weights_only
where appropriate and falls back to older torch.load signatures.
"""

from typing import Any
import torch


def safe_torch_load(
    path: str, map_location=None, weights_only: bool | None = None
) -> Any:
    """Load a checkpoint robustly across PyTorch versions.

    Args:
        path: file path to load.
        map_location: passed to torch.load.
        weights_only: If True, request weights-only safe load. If False,
            request full unpickle. If None, prefer weights-only when
            available.

    Returns:
        The object returned by torch.load.

    Notes:
        On PyTorch versions that don't accept the weights_only kwarg this
        will fall back to the legacy torch.load signature.
    """
    # Choose preferred behavior if not explicitly provided
    if weights_only is None:
        weights_only = True

    # Try calling torch.load with the desired kwarg; fall back when not supported
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Older torch that doesn't accept weights_only
        return torch.load(path, map_location=map_location)
