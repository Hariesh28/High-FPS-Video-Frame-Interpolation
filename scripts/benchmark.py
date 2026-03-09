import os
import sys
import torch
import torch.nn.functional as F
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from models.gmti_net import GMTINet
from datasets.ntire_dataset import NTIREDataset
from torch.utils.data import DataLoader


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    if mse < 1e-10:
        return 100.0
    return -10 * torch.log10(mse).item()


def average_checkpoints(checkpoint_paths, device):
    """
    Weight-average multiple checkpoints for a "snapshot ensemble" effect.
    """
    if not checkpoint_paths:
        return None

    avg_state = None
    count = len(checkpoint_paths)

    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get("ema", ckpt.get("model", ckpt))

        if avg_state is None:
            avg_state = state
        else:
            for k in avg_state.keys():
                avg_state[k] += state[k]

    for k in avg_state.keys():
        if avg_state[k].is_floating_point():
            avg_state[k] /= count
        else:
            avg_state[k] //= count

    return avg_state


class BenchmarkInference:
    def __init__(self, model, device, self_ensemble=False, scales=[1.0]):
        self.model = model
        self.device = device
        self.self_ensemble = self_ensemble
        self.scales = scales

    def _inference_single(self, L, R, scale=1.0):
        if scale != 1.0:
            B, C, H, W = L.shape
            H_new, W_new = int(H * scale), int(W * scale)
            H_new, W_new = (H_new // 16) * 16, (W_new // 16) * 16
            L_s = F.interpolate(
                L, size=(H_new, W_new), mode="bilinear", align_corners=False
            )
            R_s = F.interpolate(
                R, size=(H_new, W_new), mode="bilinear", align_corners=False
            )
            pred_s, aux = self.model(L_s, R_s)
            pred = F.interpolate(
                pred_s, size=(H, W), mode="bilinear", align_corners=False
            )
            conf = (aux["conf_lr"] + aux["conf_rl"]) / 2.0
            conf = F.interpolate(
                conf, size=(H, W), mode="bilinear", align_corners=False
            )
            return pred, conf

        pred, aux = self.model(L, R)
        conf = (aux["conf_lr"] + aux["conf_rl"]) / 2.0
        return pred, conf

    def _inference_ensemble(self, L, R, scale=1.0):
        if not self.self_ensemble:
            pred, _ = self._inference_single(L, R, scale)
            return pred

        all_preds = []
        all_confs = []

        # 1. Original
        p, c = self._inference_single(L, R, scale)
        all_preds.append(p)
        all_confs.append(c)

        # 2. H-Flip
        p, c = self._inference_single(torch.flip(L, [3]), torch.flip(R, [3]), scale)
        all_preds.append(torch.flip(p, [3]))
        all_confs.append(torch.flip(c, [3]))

        # 3. V-Flip
        p, c = self._inference_single(torch.flip(L, [2]), torch.flip(R, [2]), scale)
        all_preds.append(torch.flip(p, [2]))
        all_confs.append(torch.flip(c, [2]))

        # 4. HV-Flip
        p, c = self._inference_single(
            torch.flip(L, [2, 3]), torch.flip(R, [2, 3]), scale
        )
        all_preds.append(torch.flip(p, [2, 3]))
        all_confs.append(torch.flip(c, [2, 3]))

        # Confidence-Weighted Averaging
        # result = sum(P_i * C_i) / sum(C_i)
        preds_stack = torch.stack(all_preds)  # [4, B, 3, H, W]
        confs_stack = torch.stack(all_confs)  # [4, B, 1, H, W]

        # Soft-maxing confidence weights across the ensemble members for stability
        weights = torch.softmax(confs_stack, dim=0)  # [4, B, 1, H, W]

        return torch.sum(preds_stack * weights, dim=0)

    @torch.no_grad()
    def __call__(self, L, R):
        results = []
        for s in self.scales:
            results.append(self._inference_ensemble(L, R, s))
        return torch.stack(results).mean(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="+",
        required=True,
        help="One or more checkpoints to average",
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--self_ensemble", action="store_true")
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.0, 1.25],
        help="Scales to average (e.g. 1.0 1.25)",
    )
    parser.add_argument("--data_dir", type=str, default="data/NTIRE/val")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

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

    if len(args.checkpoint) > 1:
        print(f"Averaging {len(args.checkpoint)} checkpoints...")
        sd = average_checkpoints(args.checkpoint, device)
    else:
        ckpt = torch.load(args.checkpoint[0], map_location=device)
        sd = ckpt.get("ema", ckpt.get("model", ckpt))

    model.load_state_dict(sd)
    model.eval()

    dataset = NTIREDataset(
        root=args.data_dir, mode="val", crop_size=None, augment=False
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    infer = BenchmarkInference(model, device, args.self_ensemble, args.scales)

    psnrs = []
    print(f"Benchmarking: {args.checkpoint}")
    print(f"Self-Ensemble: {args.self_ensemble}, Scales: {args.scales}")

    for L, M, R in tqdm(loader):
        L, M, R = L.to(device), M.to(device), R.to(device)
        pred = infer(L, R).clamp(0, 1)
        psnrs.append(compute_psnr(pred, M))

    print(f"\nFinal PSNR: {np.mean(psnrs):.4f} dB")


if __name__ == "__main__":
    main()
