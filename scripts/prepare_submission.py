import os
import glob
import torch
import shutil
import subprocess
import zipfile
import sys
from tqdm import tqdm
from PIL import Image


def find_best_model(ckpt_dir="checkpoints"):
    print(f"[*] Searching for best model in {ckpt_dir}...")
    # Priority 1: Specifically requested stage_2_final.pth
    s2_final = os.path.join(ckpt_dir, "stage_2_final.pth")
    if os.path.exists(s2_final):
        print(f"[*] Found specifically requested model: {s2_final}")
        return s2_final

    # Priority 2: Find best model by PSNR
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    best_psnr = -1.0
    best_path = None

    for path in ckpts:
        if "final" in path:
            continue  # Skip final markers if we didn't find the requested one
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            psnr = ckpt.get("psnr", -1.0)
            if psnr > best_psnr:
                best_psnr = psnr
                best_path = path
        except Exception as e:
            pass

    if best_path:
        print(
            f"[*] Automatically found best model: {best_path} (PSNR: {best_psnr:.4f})"
        )
        return best_path

    # Priority 3: Default fallback
    default_path = os.path.join(ckpt_dir, "best_ema.pth")
    print(f" [!] Falling back to default: {default_path}")
    return default_path


def run_inference(model_path, input_dir, output_dir):
    print(f"[*] Running inference on {input_dir} using {model_path}...")
    cmd = [
        "python",
        "inference.py",
        "--config",
        "config.yaml",
        "--input_dir",
        input_dir,
        "--output_dir",
        output_dir,
        "--checkpoint",
        model_path,
        "--self_ensemble",
        "--multiscale",
    ]

    # Run and stream output
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()

    if process.returncode != 0:
        print(f" [!] Error: Inference failed with code {process.returncode}")
        sys.exit(1)


def verify_and_zip(result_dir, zip_name="Submission.zip"):
    print(f"[*] Verifying resolution and zipping {result_dir}...")

    # Check resolution of the first image found
    image_paths = glob.glob(os.path.join(result_dir, "vid_*", "*.png"))
    if not image_paths:
        print(" [!] Error: No result images found!")
        sys.exit(1)

    with Image.open(image_paths[0]) as img:
        w, h = img.size
        print(f"[*] Detected resolution: {w}x{h}")
        if w != 1920 or h != 1080:
            print(f" [!] Warning: Resolution is {w}x{h}, but NTIRE requires 1920x1080!")
        else:
            print(" [OK] Resolution matches NTIRE requirements.")

    # Create Zip
    # Structure: Submission.zip -> vid_k/frame_x.png
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        vid_dirs = sorted(glob.glob(os.path.join(result_dir, "vid_*")))
        for vid_dir in tqdm(vid_dirs, desc="Zipping videos"):
            vid_name = os.path.basename(vid_dir)
            files = glob.glob(os.path.join(vid_dir, "*.png"))
            for f in files:
                # Add file to zip at path: vid_name/filename
                arcname = os.path.join(vid_name, os.path.basename(f))
                zipf.write(f, arcname)

    print(f"[*] Successfully created {zip_name}")


def main():
    # Setup paths
    input_dir = "data/NTIRE/test"
    if not os.path.exists(input_dir):
        print(f" [!] Warning: {input_dir} not found. Using 'val' instead.")
        input_dir = "data/NTIRE/val"

    output_dir = "temp_submission"
    zip_name = "Submission.zip"

    # Clean prev run
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # 1. Find best model
    best_model = find_best_model()

    # 2. Run Inference
    run_inference(best_model, input_dir, output_dir)

    # 3. Zip and Verify
    verify_and_zip(output_dir, zip_name)

    print("\n" + "=" * 40)
    print(" SUBMISSION READY: Submission.zip")
    print("=" * 40)


if __name__ == "__main__":
    main()
