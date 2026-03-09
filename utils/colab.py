import os
import sys


def setup_colab():
    """
    Sets up the Google Colab environment:
    1. Mounts Google Drive.
    2. Installs missing dependencies.
    3. Adds the project root to sys.path.
    """
    try:
        from google.colab import drive

        print("[Colab] Mounting Google Drive...")
        drive.mount("/content/drive")
        IS_COLAB = True
    except ImportError:
        IS_COLAB = False
        print("[Colab] Not running in Google Colab.")
        return False

    # Install specific dependencies if missing
    import subprocess

    print("[Colab] Checking dependencies...")
    subprocess.run("pip install -q lpips==0.1.4 pytorch-msssim==0.3.7", shell=True)

    # Add project to path
    project_root = "/content/drive/MyDrive/GMTI_Net"  # Standard assumption
    if os.path.exists(project_root):
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        os.chdir(project_root)
        print(f"[Colab] Working directory set to: {project_root}")
    else:
        print(
            f"[Colab] Warning: Project root {project_root} not found. Please ensure your code is in MyDrive/GMTI_Net"
        )

    return True


def get_colab_paths(config_paths):
    """
    Adjusts paths if running in Colab.
    """
    if "google.colab" in sys.modules:
        drive_root = "/content/drive/MyDrive/GMTI_Data"
        return {
            "train_dir": os.path.join(drive_root, "NTIRE/train"),
            "val_dir": os.path.join(drive_root, "NTIRE/val"),
            "checkpoint_dir": os.path.join(drive_root, "checkpoints"),
            "log_dir": os.path.join(drive_root, "logs"),
        }
    return config_paths
