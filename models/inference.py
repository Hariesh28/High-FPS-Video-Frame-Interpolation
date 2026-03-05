import torch
import torch.nn.functional as F


def apply_transform_pair(L, R, t):
    """Applies geometric T transforms for self-ensembling."""
    if t == "none":
        return L, R
    elif t == "hflip":
        return L.flip(-1), R.flip(-1)
    elif t == "vflip":
        return L.flip(-2), R.flip(-2)
    elif t == "hvflip":
        return L.flip(-2, -1), R.flip(-2, -1)
    elif t == "reverse":
        return R, L
    else:
        raise ValueError(f"Unknown transform {t}")


def inverse_transform(pred, t):
    """Reverts geometric T transforms for aggregated prediction matching."""
    if t == "none" or t == "reverse":
        return pred
    elif t == "hflip":
        return pred.flip(-1)
    elif t == "vflip":
        return pred.flip(-2)
    elif t == "hvflip":
        return pred.flip(-2, -1)
    else:
        raise ValueError(f"Unknown transform {t}")


def self_ensemble_infer(
    model, L, R, transforms=["none", "hflip", "vflip", "hvflip", "reverse"]
):
    """
    Robust self-ensembling averaging multi-angle and reversed forward passes
    to maximize PSNR through structural symmetry.
    """
    preds = []

    # Optional multi-scale inference can be wrapped around this loop
    for t in transforms:
        Lt, Rt = apply_transform_pair(L, R, t)
        with torch.no_grad():
            pred_t, _ = model(Lt, Rt)

        pred_inv = inverse_transform(pred_t, t)
        preds.append(pred_inv)

    return torch.stack(preds, dim=0).mean(dim=0)
