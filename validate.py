"""
GMTI-Net Validation Script

Evaluates model on validation set.
Reports per-video and overall PSNR and SSIM.

Usage:
    python validate.py
    python validate.py --checkpoint checkpoints/best_model.pth
"""

import os
import math
import argparse
import yaml
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.gmti_net import GMTINet
from datasets.ntire_dataset import NTIREDataset

try:
    from pytorch_msssim import ssim as compute_ssim

    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False
    print("[Warning] pytorch_msssim not installed, SSIM will not be computed")


def compute_psnr(pred, gt):
    """Exact PSNR formula per user request: -10 * log10(MSE) with RGB in [0, 1]."""
    mse = F.mse_loss(pred, gt)
    if mse < 1e-10:
        return 100.0
    return -10 * math.log10(mse.item())


def validate(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Validate] Using device: {device}")

    # Dataset
    val_dataset = NTIREDataset(
        root=config["data"]["val_dir"],
        mode="val",
        crop_size=config["data"]["crop_size"],
        augment=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Model
    model = GMTINet(
        swin_depth=config["model"]["encoder"]["swin_depth"],
        swin_heads=config["model"]["encoder"]["swin_heads"],
        swin_window_size=config["model"]["encoder"]["swin_window_size"],
        swin_mlp_ratio=config["model"]["encoder"]["swin_mlp_ratio"],
        flow_refinement_iters=config["model"]["flow"]["refinement_iters"],
        use_deformable=config["model"]["flow"]["use_deformable"],
        transformer_blocks=config["model"]["transformer"]["blocks"],
        transformer_heads=config["model"]["transformer"]["heads"],
        transformer_dim=config["model"]["transformer"]["embed_dim"],
        transformer_mlp_ratio=config["model"]["transformer"]["mlp_ratio"],
    ).to(device)

    # Load checkpoint
    ckpt_path = args.checkpoint or "checkpoints/best_ema.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = (
            "checkpoints/best_model.pth"  # Fallback to manual best if ema not found
        )

    if os.path.exists(ckpt_path):
        # Use the safe loader helper which prefers weights-only loads when
        # available but falls back to full loads on older torch versions.
        from utils.io import safe_torch_load

        ckpt = safe_torch_load(ckpt_path, map_location=device, weights_only=True)
        # Prioritize EMA weights for validation
        state_dict = ckpt.get("ema", ckpt.get("model"))
        if state_dict is not None:
            model.load_state_dict(state_dict)
        else:
            print(f"[Validate] Warning: no model weights found in {ckpt_path}")
        print(f"[Validate] Loaded checkpoint: {ckpt_path}")
        if "iteration" in ckpt:
            print(f"[Validate] Checkpoint iteration: {ckpt['iteration']}")
    else:
        print(f"[Validate] No checkpoint found at {ckpt_path}, using random weights")

    model.eval()

    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for i, (L, M, R) in enumerate(tqdm(val_loader, desc="Validating")):
            L = L.to(device)
            M = M.to(device)
            R = R.to(device)

            pred = model.inference(L, R).clamp(0, 1)

            # PSNR
            psnr = compute_psnr(pred, M)
            psnr_values.append(psnr)

            # SSIM
            if HAS_MSSSIM:
                ssim_val = compute_ssim(
                    pred, M, data_range=1.0, size_average=True
                ).item()
                ssim_values.append(ssim_val)

    # Report results
    avg_psnr = np.mean(psnr_values)
    print(f"\n{'=' * 50}")
    print(f"Validation Results ({len(psnr_values)} samples)")
    print(f"{'=' * 50}")
    print(f"  PSNR:  {avg_psnr:.3f} dB (±{np.std(psnr_values):.3f})")
    if ssim_values:
        avg_ssim = np.mean(ssim_values)
        print(f"  SSIM:  {avg_ssim:.4f} (±{np.std(ssim_values):.4f})")
    print(f"  Min PSNR: {np.min(psnr_values):.3f}")
    print(f"  Max PSNR: {np.max(psnr_values):.3f}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="GMTI-Net Validation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    validate(config, args)


if __name__ == "__main__":
    main()
