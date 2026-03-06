"""
GMTI-Net Inference Pipeline

Self-ensemble:
    identity, flip H, flip V, flip HV → average outputs
    Gain: +0.05–0.1 PSNR

Checkpoint averaging:
    Average last 5 checkpoint weights

Multi-scale inference:
    Scales 1.0, 1.25 → average predictions

Usage:
    python inference.py --input_dir val --output_dir results
    python inference.py --input_dir val --output_dir results --self_ensemble --multiscale
"""

import os
import glob
import argparse
import yaml
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from models.gmti_net import GMTINet
from utils.io import safe_torch_load, extract_model_state


def load_image(path, device="cuda"):
    """Load image as [1, 3, H, W] tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = torch.from_numpy(np.array(img)).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)
    return img


def save_image(tensor, path):
    """Save [1, 3, H, W] tensor as PNG."""
    img = tensor.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def self_ensemble(model, L, R):
    """Test-time augmentation with 4 transforms.

    Average predictions from: identity, flip-H, flip-V, flip-HV.
    """
    preds = []

    transforms = [
        (lambda x: x, lambda x: x),  # identity
        (lambda x: torch.flip(x, [3]), lambda x: torch.flip(x, [3])),  # flip H
        (lambda x: torch.flip(x, [2]), lambda x: torch.flip(x, [2])),  # flip V
        (lambda x: torch.flip(x, [2, 3]), lambda x: torch.flip(x, [2, 3])),  # flip HV
    ]

    for fwd, inv in transforms:
        L_t = fwd(L)
        R_t = fwd(R)
        pred_t = model.inference(L_t, R_t)
        pred = inv(pred_t)
        preds.append(pred)

    return torch.stack(preds).mean(dim=0)


def multiscale_inference(model, L, R, scales=[1.0, 1.25], ensemble=False):
    """Multi-scale inference: run at multiple scales and average.

    Args:
        model: GMTINet model
        L, R: [1, 3, H, W] input frames
        scales: list of scale factors
        ensemble: whether to use self-ensemble at each scale
    """
    B, C, H, W = L.shape
    preds = []

    for scale in scales:
        if scale != 1.0:
            Hs, Ws = int(H * scale), int(W * scale)
            # Ensure divisible by 16
            Hs = (Hs // 16) * 16
            Ws = (Ws // 16) * 16
            L_s = F.interpolate(L, size=(Hs, Ws), mode="bilinear", align_corners=False)
            R_s = F.interpolate(R, size=(Hs, Ws), mode="bilinear", align_corners=False)
        else:
            L_s, R_s = L, R

        if ensemble:
            pred_s = self_ensemble(model, L_s, R_s)
        else:
            pred_s = model.inference(L_s, R_s)

        # Resize back to original
        if scale != 1.0:
            pred_s = F.interpolate(
                pred_s, size=(H, W), mode="bilinear", align_corners=False
            )

        preds.append(pred_s)

    return torch.stack(preds).mean(dim=0)


def average_checkpoints(checkpoint_paths, device="cuda"):
    """Average model weights from multiple checkpoints."""
    avg_state = None

    from utils.io import safe_torch_load

    for path in checkpoint_paths:
        ckpt = safe_torch_load(path, map_location=device, weights_only=True)
        state = extract_model_state(ckpt, warn=True)
        if state is None:
            raise KeyError(f"No model weights found in checkpoint: {path}")

        if avg_state is None:
            avg_state = OrderedDict()
            for k, v in state.items():
                avg_state[k] = v.float()
        else:
            for k, v in state.items():
                avg_state[k] += v.float()

    # Average
    n = len(checkpoint_paths)
    for k in avg_state:
        avg_state[k] /= n

    return avg_state


def load_model_for_inference(checkpoint_path, config_path=None, device="cuda"):
    """Load a GMTINet model from a checkpoint, ready for inference.

    This is the primary entry-point used by the Colab notebook (Cell 15)
    and any external scripts that need a pre-built inference model.

    Args:
        checkpoint_path: Path to .pth checkpoint (supports avg_ema / best_ema / latest).
        config_path:     Path to config.yaml.  If None, looks for config.yaml in the
                         current directory and then in the checkpoint's parent directory.
        device:          'cuda' or 'cpu'.

    Returns:
        (model, cfg) — model is in eval mode on the requested device.
    """
    import yaml, os

    # ── resolve config ──────────────────────────────────────────────────────
    if config_path is None:
        candidates = [
            "config.yaml",
            os.path.join(os.path.dirname(str(checkpoint_path)), "..", "config.yaml"),
        ]
        config_path = next((c for c in candidates if os.path.isfile(c)), None)
    if config_path is None or not os.path.isfile(str(config_path)):
        raise FileNotFoundError(
            "config.yaml not found. Pass config_path= explicitly or place it in the working directory."
        )

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ── build model ─────────────────────────────────────────────────────────
    enc = cfg["model"]["encoder"]
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = GMTINet(
        swin_depth=enc["swin_depth"],
        swin_heads=enc["swin_heads"],
        swin_window_size=enc["swin_window_size"],
        swin_mlp_ratio=enc["swin_mlp_ratio"],
        flow_refinement_iters=cfg["model"]["flow"]["refinement_iters"],
        use_deformable=cfg["model"]["flow"]["use_deformable"],
        transformer_blocks=cfg["model"]["transformer"]["blocks"],
        transformer_heads=cfg["model"]["transformer"]["heads"],
        transformer_dim=cfg["model"]["transformer"]["embed_dim"],
        transformer_mlp_ratio=cfg["model"]["transformer"]["mlp_ratio"],
    ).to(device)

    # ── load weights ────────────────────────────────────────────────────────
    ckpt = safe_torch_load(
        str(checkpoint_path), map_location=str(device), weights_only=False
    )
    state = extract_model_state(ckpt, warn=True)
    if state is None:
        raise KeyError(f"No model weights found in checkpoint: {checkpoint_path}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(
            f"[load_model_for_inference] Missing keys ({len(missing)}): {missing[:5]} ..."
        )
    if unexpected:
        print(
            f"[load_model_for_inference] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ..."
        )

    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[load_model_for_inference] Loaded {checkpoint_path}  ({n_params/1e6:.2f} M params)  device={device}"
    )
    return model, cfg


def inference(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Using device: {device}")

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

    # Load checkpoint(s)
    if args.avg_checkpoints:
        ckpt_paths = sorted(glob.glob("checkpoints/iter_*.pth"))[
            -args.avg_checkpoints :
        ]
        print(f"[Inference] Averaging {len(ckpt_paths)} checkpoints")
        avg_state = average_checkpoints(ckpt_paths, device)
        model.load_state_dict(avg_state)
    else:
        ckpt_path = args.checkpoint or "checkpoints/best_ema.pth"
        if not os.path.exists(ckpt_path):
            ckpt_path = "checkpoints/best_model.pth"

        print(f"[Inference] Loading: {ckpt_path}")
        ckpt = safe_torch_load(ckpt_path, map_location=device, weights_only=True)
        state_dict = extract_model_state(ckpt, warn=True)
        if state_dict is None:
            raise KeyError(f"No model weights found in checkpoint: {ckpt_path}")
        model.load_state_dict(state_dict)

    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    # Find video directories
    vid_dirs = sorted(glob.glob(os.path.join(args.input_dir, "vid_*")))

    for vid_dir in tqdm(vid_dirs, desc="Processing videos"):
        vid_name = os.path.basename(vid_dir)
        out_vid_dir = os.path.join(args.output_dir, vid_name)
        os.makedirs(out_vid_dir, exist_ok=True)

        frames = sorted(glob.glob(os.path.join(vid_dir, "*.png")))
        if len(frames) < 2:
            continue

        # Predict intermediate frame between consecutive pairs
        with torch.no_grad():
            for i in range(len(frames) - 1):
                L = load_image(frames[i], device)
                R = load_image(frames[i + 1], device)

                if args.self_ensemble and args.multiscale:
                    scales = config["inference"]["multiscale"]
                    pred = multiscale_inference(
                        model, L, R, scales=scales, ensemble=True
                    )
                elif args.self_ensemble:
                    pred = self_ensemble(model, L, R)
                elif args.multiscale:
                    scales = config["inference"]["multiscale"]
                    pred = multiscale_inference(model, L, R, scales=scales)
                else:
                    pred = model.inference(L, R)

                out_path = os.path.join(out_vid_dir, f"pred_{i:04d}_{i + 1:04d}.png")
                save_image(pred, out_path)

    print(f"[Inference] Results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="GMTI-Net Inference")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input_dir", type=str, default="val")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--self_ensemble", action="store_true", help="Use self-ensemble (TTA)"
    )
    parser.add_argument(
        "--multiscale", action="store_true", help="Use multi-scale inference"
    )
    parser.add_argument(
        "--avg_checkpoints",
        type=int,
        default=0,
        help="Average last N checkpoints (0=disabled)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    inference(config, args)


if __name__ == "__main__":
    main()
