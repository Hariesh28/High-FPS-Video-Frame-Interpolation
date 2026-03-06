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
import json
import random
import argparse
import subprocess
import yaml
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
from utils.io import (
    safe_torch_load,
    extract_model_state,
    atomic_save,
    prune_checkpoints,
)
from utils.misc import (
    seed_everything,
    make_worker_init_fn,
    log_environment,
    _get_git_rev,
)


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

    # ---- TensorBoard & Dirs (create early for log_environment) ----
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    writer = SummaryWriter("logs")

    # ---- Environment logging (before dataset/model construction) ----
    log_environment(writer, run_dir="logs")

    # Debug visualisation frequency
    vis_freq = getattr(args, "vis_freq", 1000)

    # ---- Dataset & Sampler ----
    stage_idx = getattr(args, "stage", 1) - 1
    stages = config.get("curriculum", {}).get("stages", [])
    if 0 <= stage_idx < len(stages):
        dataset_names = stages[stage_idx].get("datasets", ["ntire"])
    else:
        dataset_names = ["ntire"]

    from datasets.mixed import MixedDataset
    from datasets.vimeo90k import Vimeo90KDataset
    from datasets.adobe240 import Adobe240Dataset
    from datasets.hard_sampler import FlowMagnitudeWeightedSampler, PSNRBiasedSampler

    ds_list = []
    w_list = []
    mix_weights = config.get("dataset_mixing", {}).get("weights", {})

    for name in dataset_names:
        if name.lower() == "ntire":
            try:
                ds_list.append(
                    NTIREDataset(
                        root=config["data"]["train_dir"],
                        mode="train",
                        crop_size=config["data"]["crop_size"],
                        augment=True,
                    )
                )
                w_list.append(mix_weights.get("ntire", 1.0))
            except FileNotFoundError:
                print(
                    f"[Warning] NTIRE dataset not found at {config['data']['train_dir']}. Skipping."
                )
        elif name.lower() == "vimeo90k":
            try:
                ds_list.append(
                    Vimeo90KDataset(
                        root="data/vimeo_triplet",
                        split="train",
                        crop_size=config["data"]["crop_size"],
                        augment=True,
                    )
                )
                w_list.append(mix_weights.get("vimeo90k", 0.3))
            except FileNotFoundError:
                print(
                    f"[Warning] Vimeo90K dataset not found at data/vimeo_triplet. Skipping."
                )
        elif name.lower() == "adobe240":
            try:
                ds_list.append(
                    Adobe240Dataset(
                        root="data/adobe240",
                        split="train",
                        crop_size=config["data"]["crop_size"],
                        augment=True,
                    )
                )
                w_list.append(mix_weights.get("adobe240", 0.1))
            except FileNotFoundError:
                print(
                    f"[Warning] Adobe240 dataset not found at data/adobe240. Skipping."
                )

    if not ds_list:
        raise RuntimeError(
            "No datasets were successfully loaded. Please check your data directions."
        )

    sampler_mode = config.get("hard_sampler", {}).get("mode", "off")
    is_psnr_biased = sampler_mode == "psnr_biased"

    class IndexWrapper(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            return (*self.ds[i], i)

    if is_psnr_biased:
        ds_list = [IndexWrapper(ds) for ds in ds_list]

    seed = config["training"].get("seed", 42)
    train_dataset, sampler = MixedDataset.build(
        datasets=ds_list,
        weights=w_list,
        num_samples=100_000,
        generator=torch.Generator().manual_seed(seed),
    )

    if sampler_mode == "flow_magnitude":
        sampler = FlowMagnitudeWeightedSampler(
            train_dataset,
            num_samples=100_000,
            alpha=config["hard_sampler"].get("flow_mag_alpha", 0.5),
            min_weight=config["hard_sampler"].get("flow_mag_min_weight", 1e-4),
        )
    elif is_psnr_biased:
        sampler = PSNRBiasedSampler(
            train_dataset,
            num_samples=100_000,
            temperature=config["hard_sampler"].get("psnr_temperature", 3.0),
            init_psnr=config["hard_sampler"].get("psnr_init", 32.0),
        )

    worker_init = make_worker_init_fn(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=config["data"].get("num_workers", 0) > 0,
        worker_init_fn=worker_init,
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
        num_workers=min(2, config["data"].get("num_workers", 2)),
        pin_memory=True,
        persistent_workers=config["data"].get("num_workers", 0) > 0,
        worker_init_fn=make_worker_init_fn(config["training"].get("seed", 42) + 9999),
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
        # MSE term directly optimises PSNR = -10*log10(MSE). Expected: +0.05-0.2 dB.
        w_mse=config["loss"].get("mse", 0.1),
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
    # Read from ema.decay (new section) with fallback to training.ema_decay
    ema_decay = config.get("ema", {}).get("decay") or config["training"].get(
        "ema_decay", 0.9999
    )
    ema = EMAModel(model, decay=ema_decay)
    print(f"[Train] EMA decay: {ema_decay}")

    # ---- AMP ----
    use_amp = config["training"]["amp"] and torch.cuda.is_available()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ---- Dirs (already created at top of train()) ----

    # ---- Resume checkpoint ----
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = safe_torch_load(args.resume, map_location=device, weights_only=False)

        state_dict = extract_model_state(ckpt, warn=True)
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
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        idx = None
        if len(batch) == 4:
            L, M, R, idx = batch
        else:
            L, M, R = batch

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
            mse_per_sample = F.mse_loss(pred.clamp(0, 1), M, reduction="none").mean(
                dim=[1, 2, 3]
            )
            psnr_per_sample = -10 * torch.log10(torch.clamp(mse_per_sample, min=1e-10))
            psnr = psnr_per_sample.mean().item()

        # Update PSNR sampler history
        if is_psnr_biased and idx is None and iteration == start_iter:
            print(
                "[Warning] PSNRBiasedSampler requires index but Dataset did not return it."
            )
        elif is_psnr_biased and idx is not None:
            sampler.update_psnr_batch(idx.tolist(), psnr_per_sample.tolist())
            if (iteration + 1) % 1000 == 0:
                sampler.refresh_weights()

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
        if (iteration + 1) % vis_freq == 0:
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
            try:
                git_rev = _get_git_rev()
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
            final_path = f"checkpoints/iter_{iteration + 1}.pth"
            atomic_save(checkpoint_data, final_path)
            atomic_save(checkpoint_data, "checkpoints/latest.pth")

            # Prune old checkpoints (keep last N)
            prune_checkpoints(
                ckpt_dir="checkpoints",
                keep_last=config["training"].get("keep_last_checkpoints", 5),
            )

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
    parser.add_argument(
        "--stage", type=int, default=None, help="Curriculum stage index"
    )
    parser.add_argument(
        "--patch_size", type=int, default=None, help="Override patch/crop size"
    )
    parser.add_argument(
        "--save_freq", type=int, default=None, help="Override checkpoint save freq"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable cudnn deterministic mode (slower, fully reproducible). "
        "Sets cudnn.deterministic=True, cudnn.benchmark=False.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: save flow visualisations every 100 iters instead of 1000.",
    )
    args = parser.parse_args()

    # Debug vis frequency
    args.vis_freq = 100 if args.debug else 1000

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.max_iters is not None:
        config["training"]["max_iters"] = args.max_iters
    if getattr(args, "patch_size", None) is not None:
        config["data"]["crop_size"] = args.patch_size
    if getattr(args, "save_freq", None) is not None:
        config["training"]["checkpoint_freq"] = args.save_freq
    if getattr(args, "stage", None) is not None:
        stages = config.get("curriculum", {}).get("stages", [])
        for s in stages:
            if s.get("stage") == args.stage:
                print(f"[Curriculum] Stage {args.stage}: {s.get('description', '')}")
                if "patch_size" in s and getattr(args, "patch_size", None) is None:
                    config["data"]["crop_size"] = s["patch_size"]
                if "max_iters" in s and getattr(args, "max_iters", None) is None:
                    config["training"]["max_iters"] = s["max_iters"]
                if "lr" in s:
                    config["training"]["lr"] = s["lr"]

    seed = args.seed or config["training"].get("seed", 42)
    seed_everything(seed, deterministic=getattr(args, "deterministic", False))

    train(config, args)


if __name__ == "__main__":
    main()
