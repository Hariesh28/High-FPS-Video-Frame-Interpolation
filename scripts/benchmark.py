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
            # Ensure divisible by 16
            H_new = (H_new // 16) * 16
            W_new = (W_new // 16) * 16
            L_s = F.interpolate(
                L, size=(H_new, W_new), mode="bilinear", align_corners=False
            )
            R_s = F.interpolate(
                R, size=(H_new, W_new), mode="bilinear", align_corners=False
            )
            pred_s = self.model.inference(L_s, R_s)
            pred = F.interpolate(
                pred_s, size=(H, W), mode="bilinear", align_corners=False
            )
            return pred
        return self.model.inference(L, R)

    def _inference_ensemble(self, L, R, scale=1.0):
        if not self.self_ensemble:
            return self._inference_single(L, R, scale)

        # Original
        preds = [self._inference_single(L, R, scale)]

        # H-Flip
        preds.append(
            torch.flip(
                self._inference_single(torch.flip(L, [3]), torch.flip(R, [3]), scale),
                [3],
            )
        )

        # V-Flip
        preds.append(
            torch.flip(
                self._inference_single(torch.flip(L, [2]), torch.flip(R, [2]), scale),
                [2],
            )
        )

        # HV-Flip
        preds.append(
            torch.flip(
                self._inference_single(
                    torch.flip(L, [2, 3]), torch.flip(R, [2, 3]), scale
                ),
                [2, 3],
            )
        )

        return torch.stack(preds).mean(0)

    @torch.no_grad()
    def __call__(self, L, R):
        results = []
        for s in self.scales:
            results.append(self._inference_ensemble(L, R, s))
        return torch.stack(results).mean(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--self_ensemble", action="store_true")
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0])
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

    ckpt = torch.load(args.checkpoint, map_location=device)
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
