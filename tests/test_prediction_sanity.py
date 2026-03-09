import os
import glob
import re
import argparse
import yaml
import json
import torch
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import zipfile
from models.gmti_net import GMTINet
from utils.io import safe_torch_load, extract_model_state


def load_image(path, target_size=(1080, 1920)):
    """Load image and ensure target size [H, W]."""
    img = Image.open(path).convert("RGB")
    # Always resize to target size to be absolute sure
    if img.size != (target_size[1], target_size[0]):
        img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)

    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return img_tensor


def create_submission_zip(results_dir, output_zip):
    """Create a zip where vid_* folders are at the top level."""
    print(f"Creating submission zip: {output_zip}")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        vid_dirs = sorted([d for d in os.listdir(results_dir) if d.startswith("vid_")])
        for vid_name in vid_dirs:
            vid_path = os.path.join(results_dir, vid_name)
            for root, _, files in os.walk(vid_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Archive name: vid_k/filename.png
                    archive_name = os.path.join(vid_name, file)
                    zf.write(file_path, archive_name)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="GMTI-Net Inference on Test Sequences")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pth")
    parser.add_argument("--test_dir", type=str, default="data/NTIRE/test")
    parser.add_argument("--out_dir", type=str, default="results/test_predictions")
    parser.add_argument("--zip_name", type=str, default="Submission.zip")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load model
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

    if os.path.exists(args.checkpoint):
        ckpt = safe_torch_load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = extract_model_state(ckpt)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(
            f"Warning: No checkpoint found at {args.checkpoint}. Using random weights."
        )

    model.eval()

    vid_dirs = sorted(glob.glob(os.path.join(args.test_dir, "vid_*")))
    os.makedirs(args.out_dir, exist_ok=True)

    target_h, target_w = 1080, 1920

    for vid_dir in tqdm(vid_dirs, desc="Inference"):
        frames = sorted(glob.glob(os.path.join(vid_dir, "*.png")))
        if len(frames) != 2:
            print(f"Skipping {vid_dir}: expected 2 frames, found {len(frames)}")
            continue

        # Extract frame numbers
        nums = [int(re.search(r"(\d+)", os.path.basename(f)).group(1)) for f in frames]
        mid_num = sum(nums) // 2

        output_name = f"frame_{mid_num:06d}.png"
        vid_name = os.path.basename(vid_dir)
        vid_out_dir = os.path.join(args.out_dir, vid_name)
        os.makedirs(vid_out_dir, exist_ok=True)

        I0 = load_image(frames[0], (target_h, target_w)).to(device)
        I1 = load_image(frames[1], (target_h, target_w)).to(device)

        with torch.no_grad():
            pred = model.inference(I0, I1)
            pred = pred.clamp(0, 1).squeeze(0)  # [3, H, W]

        # Extra safety resize
        if pred.shape[1:] != (target_h, target_w):
            pred = TF.resize(
                pred, (target_h, target_w), interpolation=TF.InterpolationMode.LANCZOS
            )

        # Save
        pred_arr = (pred.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pred_img = Image.fromarray(pred_arr)
        pred_img.save(os.path.join(vid_out_dir, output_name))

    # All done, create zip
    create_submission_zip(args.out_dir, args.zip_name)


if __name__ == "__main__":
    main()
