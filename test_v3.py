import torch
from models.gmti_net import GMTINet
from losses.flow_losses import CombinedLoss
import yaml
import os


def test_v3_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize V3 model
    model = GMTINet(
        swin_depth=config["model"]["encoder"]["swin_depth"],
        swin_heads=config["model"]["encoder"]["swin_heads"],
        flow_refinement_iters=config["model"]["flow"]["refinement_iters"],
    ).to(device)

    # Dummy inputs
    L = torch.randn(1, 3, 256, 256).to(device)
    R = torch.randn(1, 3, 256, 256).to(device)
    M = torch.randn(1, 3, 256, 256).to(device)

    # Forward pass
    print("Running forward pass...")
    pred, aux = model(L, R)

    print(f"Prediction shape: {pred.shape}")
    print(f"Aux keys: {aux.keys()}")

    # Loss computation
    print("\nRunning loss computation...")
    criterion = CombinedLoss(
        w_freq=config["loss"]["freq"],
        w_mse=1.0,
        multiscale_scales=config["multiscale"]["scales"],
        multiscale_weights=config["multiscale"]["weights"],
    ).to(device)

    loss, loss_dict = criterion(pred, M, L, R, aux)
    print(f"Total loss: {loss.item()}")

    # Test BenchmarkInference (Self-Ensemble + Soft-Confidence)
    from scripts.benchmark import BenchmarkInference

    print("\nTesting BenchmarkInference (Self-Ensemble + Conf Weighting)...")
    infer = BenchmarkInference(model, device, self_ensemble=True, scales=[1.0, 1.25])
    ensemble_pred = infer(L, R)
    print(f"Ensemble prediction shape: {ensemble_pred.shape}")

    print("\nV3 Sanity Check Passed!")


if __name__ == "__main__":
    test_v3_pipeline()
