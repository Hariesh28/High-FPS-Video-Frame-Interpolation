import torch
import torch.nn.functional as F
import yaml
from models.gmti_net import GMTINet
from losses.flow_losses import CombinedLoss


def test_v3_pro_refined_smoke():
    print("Starting Refined V3.1 Pro Smoke Test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config to get weights
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize model
    model = GMTINet(use_deformable=True, flow_refinement_iters=8, num_hypotheses=3).to(
        device
    )
    print("Model initialized successfully.")

    # Dummy data (0-1 range for VGG stability)
    B, C, H, W = 2, 3, 256, 256
    L = torch.rand(B, C, H, W).to(device)
    R = torch.rand(B, C, H, W).to(device)
    GT = torch.rand(B, C, H, W).to(device)

    print("Running forward pass with temp=0.5...")
    # Forward pass
    pred, aux = model(L, R, temp=0.5)
    print(f"Forward pass completed. Pred shape: {pred.shape}")

    # Verify log_sigma (Heteroscedastic)
    if "sigma_lr" in aux:
        print(f"Log-Sigma LR mean: {aux['sigma_lr'].mean().item():.4f}")
        # Note: log_sigma can be negative, that's fine.

    # Initialize loss
    criterion = CombinedLoss(
        w_charb=config["loss"]["charbonnier"],
        w_lap=config["loss"]["laplacian"],
        w_freq=config["loss"].get("freq", 0.50),
        w_warp=config["loss"]["warping"],
        w_bidir=config["loss"]["bidirectional"],
        w_smooth=config["loss"]["smoothness"],
        w_mse=config["loss"].get("mse", 1.0),
        w_hetero=config["loss"].get("heteroscedastic", 0.1),
        w_multi=config["loss"].get("multi_hypothesis", 0.05),
        w_perceptual=config["loss"].get("perceptual", 0.005),
        w_accel=1e-4,
    ).to(device)

    print("Computing loss with progress=0.9 (DCT ramp phase)...")
    total_loss, loss_dict = criterion(pred, GT, L, R, aux, progress=0.9)

    print(f"Loss computed. Total: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")

    # Final checks
    assert not torch.isnan(total_loss), "Loss is NaN!"
    print("Smoke Test PASSED!")


if __name__ == "__main__":
    test_v3_pro_refined_smoke()
