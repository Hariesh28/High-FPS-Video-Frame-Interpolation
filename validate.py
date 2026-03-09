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
from utils.io import safe_torch_load, extract_model_state

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
        from utils.io import safe_torch_load, extract_model_state

        ckpt = safe_torch_load(ckpt_path, map_location=device, weights_only=True)
        state_dict = extract_model_state(ckpt, warn=True)
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
    lpips_values = []

    # Init LPIPS
    loss_fn_vgg = None
    try:
        import lpips

        loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)
        loss_fn_vgg.eval()
    except ImportError:
        print(
            "[Warning] lpips not installed. Run `pip install lpips`. LPIPS will not be computed."
        )

    # CSV headers
    val_results = []

    # Notebook-friendly tqdm
    is_notebook = "ipykernel" in sys.modules or "google.colab" in sys.modules
    with torch.no_grad():
        # Adjust dataset to return paths if possible, else just keep counter
        for i, (L, M, R) in enumerate(
            tqdm(
                val_loader,
                desc="Validating",
                dynamic_ncols=True,
                mininterval=5.0 if is_notebook else 0.1,
            )
        ):
            L = L.to(device)
            M = M.to(device)
            R = R.to(device)

            pred = model.inference(L, R).clamp(0, 1)

            # PSNR
            psnr = compute_psnr(pred, M)
            psnr_values.append(psnr)

            # SSIM
            ssim_val = float("nan")
            if HAS_MSSSIM:
                ssim_val = compute_ssim(
                    pred, M, data_range=1.0, size_average=True
                ).item()
                ssim_values.append(ssim_val)

            # LPIPS
            lpips_val = float("nan")
            if loss_fn_vgg is not None:
                # LPIPS expects [-1, 1] range
                pred_scaled = pred * 2.0 - 1.0
                M_scaled = M * 2.0 - 1.0
                lpips_val = loss_fn_vgg(pred_scaled, M_scaled).item()
                lpips_values.append(lpips_val)

            frame_name = f"{i:04d}"
            val_results.append((frame_name, psnr, ssim_val, lpips_val))

    # Report results
    avg_psnr = np.mean(psnr_values)
    print(f"\n{'=' * 50}")
    print(f"Validation Results ({len(psnr_values)} samples)")
    print(f"{'=' * 50}")
    print(f"  PSNR:  {avg_psnr:.3f} dB (±{np.std(psnr_values):.3f})")
    if ssim_values:
        avg_ssim = np.mean(ssim_values)
        print(f"  SSIM:  {avg_ssim:.4f} (±{np.std(ssim_values):.4f})")
    if lpips_values:
        avg_lpips = np.mean(lpips_values)
        print(f"  LPIPS: {avg_lpips:.4f} (±{np.std(lpips_values):.4f})")
    print(f"  Min PSNR: {np.min(psnr_values):.3f}")
    print(f"  Max PSNR: {np.max(psnr_values):.3f}")
    print(f"{'=' * 50}")

    # Save CSV
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        csv_path = os.path.join(args.outdir, "val_summary.csv")
        with open(csv_path, "w") as f:
            f.write("frame,PSNR,SSIM,LPIPS\n")
            for frame_tgt, p, s, l in val_results:
                f.write(f"{frame_tgt},{p:.4f},{s:.4f},{l:.4f}\n")
        print(f"Saved validation summary to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="GMTI-Net Validation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    validate(config, args)


if __name__ == "__main__":
    main()
