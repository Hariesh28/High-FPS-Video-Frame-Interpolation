"""
GMTI-Net Training Script

Training schedule per spec:
- AdamW optimizer (lr=2e-4, β1=0.9, β2=0.999, wd=1e-4)
- Cosine decay with 2000 warmup, final lr=1e-6
- Batch size 16, gradient clipping=1.0
- EMA (decay=0.9999)
- AMP mixed precision
- Checkpoint every 5000 iterations
- Validation every 2000 iterations

Usage:
    python train.py
    python train.py --max_iters 100  # Short test run
"""

import os
import sys
import math
import copy
import argparse
import yaml
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.gmti_net import GMTINet
from losses.flow_losses import CombinedLoss
from datasets.ntire_dataset import NTIREDataset


def seed_everything(seed=42):
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # For reproducibility, disable the cuDNN benchmark autotuner which can
    # introduce non-determinism by selecting different algorithms per input size.
    torch.backends.cudnn.benchmark = False


import torchvision.utils as vutils


class EMAModel:
    """Exponential Moving Average of model parameters using deepcopy."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            if param.requires_grad:
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def get_model(self):
        return self.ema_model


def get_cosine_schedule(optimizer, warmup_iters, total_iters, final_lr):
    """Cosine decay with linear warmup."""

    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return current_iter / max(warmup_iters, 1)
        progress = (current_iter - warmup_iters) / max(total_iters - warmup_iters, 1)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale so that we decay from 1.0 to final_lr/initial_lr
        return max(cosine_decay, final_lr / optimizer.defaults["lr"])

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_psnr(pred, gt):
    """Exact PSNR formula per user request: -10 * log10(MSE) with RGB in [0, 1]."""
    mse = F.mse_loss(pred, gt)
    if mse < 1e-10:
        return 100.0
    psnr = -10 * math.log10(mse.item())
    return psnr


def flow_to_color(flow):
    """Convert flow [B, 2, H, W] to RGB for visualization."""
    # Simple normalization for debug visualization
    B, _, H, W = flow.shape
    rad = torch.sqrt(torch.sum(flow**2, dim=1, keepdim=True))
    rad_max = rad.view(B, -1).max(-1)[0].view(B, 1, 1, 1) + 1e-5
    flow_norm = flow / rad_max

    # Map dx to red, dy to green
    rgb = torch.zeros(B, 3, H, W, device=flow.device)
    rgb[:, 0] = (flow_norm[:, 0] + 1) / 2.0
    rgb[:, 1] = (flow_norm[:, 1] + 1) / 2.0
    return rgb


def train(config, args):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # ---- Dataset ----
    train_dataset = NTIREDataset(
        root=config["data"]["train_dir"],
        mode="train",
        crop_size=config["data"]["crop_size"],
        augment=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = NTIREDataset(
        root=config["data"]["val_dir"],
        mode="val",
        crop_size=config["data"]["crop_size"],
        augment=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ---- Model ----
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

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model parameters: {num_params / 1e6:.2f}M")

    # ---- Loss ----
    criterion = CombinedLoss(
        w_charb=config["loss"]["charbonnier"],
        w_lap=config["loss"]["laplacian"],
        w_warp=config["loss"]["warping"],
        w_bidir=config["loss"]["bidirectional"],
        w_smooth=config["loss"]["smoothness"],
        charb_eps=config["loss"]["charbonnier_eps"],
        multiscale_scales=config["multiscale"]["scales"],
        multiscale_weights=config["multiscale"]["weights"],
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
        weight_decay=config["training"]["weight_decay"],
    )

    max_iters = args.max_iters or config["training"]["max_iters"]
    accumulate_steps = config["training"].get("accumulate_steps", 1)

    scheduler = get_cosine_schedule(
        optimizer,
        config["training"]["warmup_iters"],
        max_iters,
        config["training"]["final_lr"],
    )

    # ---- EMA ----
    ema = EMAModel(model, decay=config["training"]["ema_decay"])

    # ---- AMP ----
    use_amp = config["training"]["amp"] and torch.cuda.is_available()
    # Correct GradScaler construction. Do not pass device name as positional arg.
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ---- TensorBoard & Dirs ----
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    writer = SummaryWriter("logs")

    # ---- Resume checkpoint ----
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        # Use the safe loader helper so behavior is consistent across torch versions
        from utils.io import safe_torch_load

        ckpt = safe_torch_load(args.resume, map_location=device, weights_only=False)

        # Load model weights (support both 'model' and 'ema' keys)
        state_dict = ckpt.get("model", ckpt.get("ema", None))
        if state_dict is None:
            raise KeyError(f"No model weights found in checkpoint: {args.resume}")
        model.load_state_dict(state_dict)

        # Restore optimizer/scheduler/scaler if present
        if "optimizer" in ckpt and hasattr(optimizer, "load_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and hasattr(scheduler, "load_state_dict"):
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                # Some schedulers may not be fully serializable across versions
                print("[Train] Warning: failed to load scheduler state; continuing")
        if "scaler" in ckpt and hasattr(scaler, "load_state_dict"):
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                print("[Train] Warning: failed to load GradScaler state; continuing")

        # Restore RNG states if present
        if "torch_rng" in ckpt:
            try:
                torch.set_rng_state(ckpt["torch_rng"].cpu())
            except Exception:
                print("[Train] Warning: failed to restore torch RNG state")
        if (
            "cuda_rng" in ckpt
            and ckpt["cuda_rng"] is not None
            and torch.cuda.is_available()
        ):
            try:
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in ckpt["cuda_rng"]]
                )
            except Exception:
                print("[Train] Warning: failed to restore CUDA RNG state")

        start_iter = ckpt.get("iteration", 0)
        print(f"[Train] Resumed from iteration {start_iter}")

    # ---- Training loop ----
    model.train()
    data_iter = iter(train_loader)
    best_psnr = 0.0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(range(start_iter, max_iters), desc="Training", dynamic_ncols=True)

    for iteration in pbar:
        # Get batch
        try:
            L, M, R = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            L, M, R = next(data_iter)

        L = L.to(device, non_blocking=True)
        M = M.to(device, non_blocking=True)
        R = R.to(device, non_blocking=True)

        # Forward pass
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred, aux = model(L, R)
            loss, loss_dict = criterion(pred, M, L, R, aux)

            # Scale loss for gradient accumulation
            loss = loss / accumulate_steps

        import math

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(
                f"[Warning] Non-finite loss {loss_value} at iteration {iteration + 1}. Skipping batch."
            )
            optimizer.zero_grad(set_to_none=True)
            # Do not update GradScaler when skipping a batch (no step taken).
            continue
            continue

        # Backward pass
        scaler.scale(loss).backward()

        grad_norm = 0.0
        # Optimization step after accumulating gradients
        if (iteration + 1) % accumulate_steps == 0:
            scaler.unscale_(optimizer)

            # Exact gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["training"].get("clip_norm", 1.0)
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            # EMA update
            ema.update(model)

        # Compute PSNR
        with torch.no_grad():
            psnr = compute_psnr(pred.clamp(0, 1), M)

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            loss=f"{loss.item() * accumulate_steps:.4f}",
            psnr=f"{psnr:.2f}",
            lr=f"{current_lr:.2e}",
        )

        writer.add_scalar("train/loss", loss.item() * accumulate_steps, iteration)
        writer.add_scalar("train/psnr", psnr, iteration)
        writer.add_scalar("train/lr", current_lr, iteration)
        if grad_norm > 0:
            writer.add_scalar("train/grad_norm", grad_norm, iteration)

        for k, v in loss_dict.items():
            writer.add_scalar(f"train/loss_{k}", v, iteration)

        # ---- Flow Visualization Debug Tool ----
        if (iteration + 1) % 1000 == 0:
            with torch.no_grad():
                B_vis = min(4, L.shape[0])  # Save up to 4 items in batch
                vis_grid = torch.cat(
                    [
                        L[:B_vis],
                        flow_to_color(aux["flow_lm"][:B_vis]),
                        aux["warped_img_L"][:B_vis],
                        pred[:B_vis].clamp(0, 1),
                        M[:B_vis],
                        R[:B_vis],
                    ],
                    dim=0,
                )  # [6*B, 3, H, W]
                vutils.save_image(
                    vis_grid,
                    f"visualizations/iter_{iteration + 1:06d}.jpg",
                    nrow=B_vis,
                    normalize=False,
                )

        # ---- Validation ----
        val_freq = config["training"]["val_freq"]
        if (iteration + 1) % val_freq == 0:
            val_model = ema.get_model()
            val_model.eval()

            val_psnr_sum = 0
            val_count = 0
            with torch.no_grad():
                for vL, vM, vR in val_loader:
                    vL = vL.to(device)
                    vM = vM.to(device)
                    vR = vR.to(device)
                    vpred = val_model.inference(vL, vR)
                    val_psnr_sum += compute_psnr(vpred.clamp(0, 1), vM)
                    val_count += 1

            val_psnr = val_psnr_sum / max(val_count, 1)
            writer.add_scalar("val/psnr", val_psnr, iteration)
            print(f"\n[Val] Iter {iteration + 1} | PSNR: {val_psnr:.2f}")

            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                ckpt_data = {
                    "ema": val_model.state_dict(),
                    "iteration": iteration + 1,
                    "psnr": val_psnr,
                }
                tmp_best = "checkpoints/.tmp_best_ema.pth"
                torch.save(ckpt_data, tmp_best)
                os.replace(tmp_best, "checkpoints/best_ema.pth")
                print(f"[Val] New best EMA PSNR: {best_psnr:.2f}")

        # ---- Checkpoint ----
        ckpt_freq = config["training"]["checkpoint_freq"]
        if (iteration + 1) % ckpt_freq == 0:
            import random, numpy as np, subprocess

            try:
                git_rev = (
                    subprocess.check_output(["git", "rev-parse", "HEAD"])
                    .decode("ascii")
                    .strip()
                )
            except Exception:
                git_rev = "unknown"

            # Save RNG states conditionally to maintain portability across CPU-only runs
            cuda_rng_state = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )

            checkpoint_data = {
                "model": model.state_dict(),
                "ema": ema.get_model().state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "iteration": iteration + 1,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": cuda_rng_state,
                "np_rng": np.random.get_state(),
                "py_rng": random.getstate(),
                "git_rev": git_rev,
                "config": config,
            }
            tmp_path = f"checkpoints/.tmp_iter_{iteration + 1}.pth"
            final_path = f"checkpoints/iter_{iteration + 1}.pth"
            torch.save(checkpoint_data, tmp_path)
            os.replace(tmp_path, final_path)

            # Atomic sync of latest as well
            tmp_latest = "checkpoints/.tmp_latest.pth"
            torch.save(checkpoint_data, tmp_latest)
            os.replace(tmp_latest, "checkpoints/latest.pth")

    writer.close()
    print(f"[Train] Complete! Best val PSNR: {best_psnr:.2f}")


def main():
    parser = argparse.ArgumentParser(description="GMTI-Net Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument(
        "--max_iters", type=int, default=None, help="Override max iterations"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint to resume from"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = args.seed or config["training"].get("seed", 42)
    seed_everything(seed)

    train(config, args)


if __name__ == "__main__":
    main()
