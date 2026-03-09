import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects flow_uv to be [H, W, 2] or [2, H, W]
    """
    if isinstance(flow_uv, torch.Tensor):
        if flow_uv.dim() == 4:  # [B, 2, H, W]
            flow_uv = flow_uv[0]
        if flow_uv.shape[0] == 2:
            flow_uv = flow_uv.permute(1, 2, 0)
        flow_uv = flow_uv.detach().cpu().numpy()

    assert flow_uv.ndim == 3, f"Expected 3D flow array, got {flow_uv.ndim}D"
    H, W, _ = flow_uv.shape
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow_uv[:, :, 0], flow_uv[:, :, 1])
    if clip_flow is not None:
        mag = np.clip(mag, 0, clip_flow)
    hsv[:, :, 0] = ang * 180 / np.pi / 2
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def visualize_kernels(kernels, num_samples=4):
    """
    Visualizes learned 5x5 kernels.
    kernels: [B, 25, H, W]
    Returns a grid of kernel heatmaps.
    """
    B, K, H, W = kernels.shape
    ks = int(np.sqrt(K))  # 5

    # Pick the center pixel kernels
    centerY, centerX = H // 2, W // 2
    k_samples = kernels[:num_samples, :, centerY, centerX].view(-1, ks, ks)

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        im = axes[i].imshow(k_samples[i].detach().cpu().numpy(), cmap="viridis")
        axes[i].set_title(f"Kernel Sample {i}")
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    return fig


def visualize_dct_spectrum(img_tensor):
    """
    Computes and visualizes the average DCT energy spectrum.
    img_tensor: [B, 3, H, W]
    """
    from utils.freq import block_dct

    with torch.no_grad():
        dct = block_dct(img_tensor)  # [B, 3, H, W]
        # Average across batch and channels
        spectrum = torch.abs(dct).mean(dim=(0, 1))
        spectrum_log = torch.log1p(spectrum)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(spectrum_log.cpu().numpy(), cmap="magma")
    ax.set_title("DCT Energy Spectrum (Log Scale)")
    plt.colorbar(im, ax=ax)
    return fig


def create_analytics_grid(L, R, pred, GT, aux, it):
    """
    Creates a comprehensive debugging grid saved to disk.
    """
    import torchvision.utils as vutils

    # Convert tensors to CPU/numpy for complex plotting
    B_vis = min(2, L.shape[0])

    # 1. Image Row: L, Pred, GT, R, Error
    with torch.no_grad():
        error = (
            torch.abs(pred - GT).mean(1, keepdim=True).repeat(1, 3, 1, 1) * 5.0
        )  # Boost error for visibility
        row1 = torch.cat(
            [L[:B_vis], pred[:B_vis], GT[:B_vis], R[:B_vis], error[:B_vis].clamp(0, 1)],
            dim=0,
        )

        # 2. Flow Row: Flow_LM_Color, Flow_RM_Color, Conf_LM, Conf_RM, Occ_Mask
        flm = (
            torch.from_numpy(
                np.stack([flow_to_color(f) for f in aux["flow_lm"][:B_vis]])
            )
            .permute(0, 3, 1, 2)
            .to(L.device)
            .float()
            / 255.0
        )
        frm = (
            torch.from_numpy(
                np.stack([flow_to_color(f) for f in aux["flow_rm"][:B_vis]])
            )
            .permute(0, 3, 1, 2)
            .to(L.device)
            .float()
            / 255.0
        )
        clm = aux["conf_lr"][:B_vis].repeat(1, 3, 1, 1)
        crm = aux["conf_rl"][:B_vis].repeat(1, 3, 1, 1)
        occ = aux["fused_mask"][:B_vis].repeat(1, 3, 1, 1)
        row2 = torch.cat([flm, frm, clm, crm, occ], dim=0)

    grid = torch.cat([row1, row2], dim=0)  # [20, 3, H, W] if B_vis=2
    vutils.save_image(grid, f"visualizations/analytics_iter_{it:06d}.png", nrow=5)
