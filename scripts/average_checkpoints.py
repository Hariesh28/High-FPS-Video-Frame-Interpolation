import torch
from utils.io import safe_torch_load


def average_checkpoints(paths, key="ema", outpath="avg_ema.pth"):
    avg_sd = None
    n = 0
    for p in paths:
        ckpt = safe_torch_load(p, map_location="cpu", weights_only=True)
        sd = ckpt.get(key, ckpt.get("model", None))
        if sd is None:
            raise KeyError(f"no '{key}' or 'model' found in checkpoint {p}")
        if avg_sd is None:
            avg_sd = {k: v.clone().float() for k, v in sd.items()}
        else:
            for k in avg_sd:
                avg_sd[k] += sd[k].float()
        n += 1

    for k in avg_sd:
        avg_sd[k] /= n

    torch.save({key: avg_sd}, outpath)
    print(f"Averaged {n} checkpoints into {outpath} successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs", nargs="+", required=True, help="List of checkpoint paths"
    )
    parser.add_argument("--output", type=str, default="avg_ema.pth")
    args = parser.parse_args()
    average_checkpoints(args.inputs, outpath=args.output)
