"""
GMTI-Net Flow Estimator

Global correlation volume → softmax matching → expected displacement
Deformable flow refinement → full-resolution refinement
Learned middle flow projection for F_LM, F_RM

Tensor flow:
    Encoder stage5 features [B,160,H/16,W/16]
    → 4D correlation [B, h*w, h*w]
    → coarse flow [B,2,H/16,W/16]
    → refined flow [B,2,H,W]
    → middle flows F_LM, F_RM [B,2,H,W]

FP32 safety: correlation softmax + all coordinate arithmetic
run in forced fp32 (autocast disabled). Convex mask softmax
also runs fp32. Both are cached on device to reduce alloc overhead.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

# Try to import deformable conv, fall back to standard conv
try:
    from torchvision.ops import deform_conv2d

    HAS_DEFORMABLE = True
except ImportError:
    HAS_DEFORMABLE = False
    warnings.warn(
        "torchvision.ops.deform_conv2d not found. Falling back to standard convolutions."
    )


class FlowConfidence(nn.Module):
    """Predict confidence map from features + flow.

    Output: [B,1,H,W] in [0,1] via sigmoid.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class DeformableRefinementBlock(nn.Module):
    """Flow refinement using deformable convolution (or standard conv fallback).

    Predicts residual flow update: F = F + ΔF
    """

    def __init__(self, in_channels, use_deformable=True):
        super().__init__()
        self.use_deformable = use_deformable and HAS_DEFORMABLE

        if self.use_deformable:
            # Offset prediction for deformable conv
            self.offset_conv = nn.Sequential(
                nn.Conv2d(in_channels + 2, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 18, 3, padding=1),  # 9 offsets × 2 = 18
            )
            nn.init.constant_(self.offset_conv[-1].weight, 0.0)
            nn.init.constant_(self.offset_conv[-1].bias, 0.0)

            self.weight = nn.Parameter(
                torch.randn(in_channels, in_channels, 3, 3) * 0.01
            )
            self.bias_param = nn.Parameter(torch.zeros(in_channels))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels + 2, in_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
            )

        # Flow residual head
        self.flow_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1),
        )

    def forward(self, feat, flow):
        """
        Args:
            feat: [B, C, H, W] context features
            flow: [B, 2, H, W] current flow estimate
        Returns:
            refined_flow: [B, 2, H, W]
        """
        inp_cat = torch.cat([feat, flow], dim=1)

        if self.use_deformable:
            offset = self.offset_conv(inp_cat)
            out = deform_conv2d(
                feat,
                offset,
                self.weight,
                self.bias_param,
                padding=1,
            )
        else:
            out = self.conv(inp_cat)

        delta_flow = self.flow_head(out)
        return flow + delta_flow


class GMFlowMatching(nn.Module):
    """Robust correlation volume matching with dynamic OOM fallback.

    Implements:
    - Feature Projection Channel Alignment
    - L2 Normalization with Epsilon Clamp
    - Explicit float32 projection & Softmax Bounds
    - Cached coordinate grid per (H, W, device) to avoid repeated allocation
    - Optional top-k correlation (NOT YET IMPLEMENTED — see corr_topk)
    """

    def __init__(
        self, proj_dim=128, chunk_size=1024, temp=0.1, clamp_val=50.0, corr_topk=None
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.temp = temp
        self.clamp_val = clamp_val
        self.scale = math.sqrt(proj_dim)

        # top-k correlation placeholder
        if corr_topk is not None:
            raise NotImplementedError(
                f"corr_topk={corr_topk} is not supported in this build.\n"
                "  → To use global matching (recommended), set corr_topk=null in config.yaml\n"
                "  → To implement sparse top-k: replace this guard with a two-stage\n"
                "    coarse→top-k path in GMFlowMatching.forward().\n"
                "    Reference: GMFlow sparse variant (top-k nearest neighbor in feature space).\n"
                "    TODO: for very large images (>1080p) top-k will be necessary to avoid OOM."
            )
        self.corr_topk = corr_topk

        # Cache: maps (H, W, device_str) → precomputed coords tensor
        self._coord_cache: dict = {}

    def _get_coords(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return cached [H*W, 2] coordinate grid for (H, W, device)."""
        key = (H, W, str(device))
        if key not in self._coord_cache:
            xs = torch.arange(W, dtype=torch.float32, device=device)
            ys = torch.arange(H, dtype=torch.float32, device=device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            coords = torch.stack(
                [grid_x.flatten(), grid_y.flatten()], dim=-1
            )  # [HW, 2]
            self._coord_cache[key] = coords
        return self._coord_cache[key]

    def forward(self, fL, fR):
        """
        Args:
            fL, fR: [B, C, H, W] projected features
        Returns:
            flow: [B, 2, H, W] expected displacement
            conf: [B, 1, H, W] confidence
        """
        B, C, H, W = fL.shape
        HW = H * W

        # 1. L2 Normalize with clamp
        fLn = fL / (fL.norm(dim=1, keepdim=True).clamp_min(1e-6))
        fRn = fR / (fR.norm(dim=1, keepdim=True).clamp_min(1e-6))

        # 2. Flatten for batch matmul
        fL_flat = fLn.view(B, C, HW).permute(0, 2, 1)  # [B, HW, C]
        fR_flat = fRn.view(B, C, HW)  # [B, C, HW]

        # 3. Get cached coords
        coords = self._get_coords(H, W, fL.device)  # [HW, 2]

        def run_with_chunk(chunk_size):
            fL32 = fL_flat.float()
            fR32 = fR_flat.float()
            flows = []
            max_probs = []
            for i in range(0, HW, chunk_size):
                q = fL32[:, i : i + chunk_size, :]  # [B, chunk, C]
                corr_chunk = torch.matmul(q, fR32) / self.scale  # [B, chunk, HW]
                corr_chunk = (corr_chunk / self.temp).clamp(
                    -self.clamp_val, self.clamp_val
                )
                prob = torch.softmax(corr_chunk, dim=-1)  # [B, chunk, HW]
                disp_chunk = prob @ coords  # [B, chunk, 2]
                flows.append(disp_chunk)

                # Confidence: maximum matching probability for each source pixel
                max_prob_chunk = prob.max(dim=-1)[0]  # [B, chunk]
                max_probs.append(max_prob_chunk)

            expected_pos = torch.cat(flows, dim=1)  # [B, HW, 2]
            max_probs_all = torch.cat(max_probs, dim=1)  # [B, HW]

            flow_out = expected_pos.view(B, H, W, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]
            conf_out = max_probs_all.view(B, 1, H, W)  # [B, 1, H, W]
            return flow_out, conf_out

        # Disable autocast inside matching block — correlation + softmax must be fp32
        active_chunk = self.chunk_size
        oom_fallback_triggered = False
        with torch.amp.autocast("cuda", enabled=False):
            try:
                expected_pos, conf = run_with_chunk(self.chunk_size)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                active_chunk = max(128, self.chunk_size // 2)
                oom_fallback_triggered = True
                try:
                    expected_pos, conf = run_with_chunk(active_chunk)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    active_chunk = max(64, active_chunk // 2)
                    expected_pos, conf = run_with_chunk(active_chunk)

        # OOM watchdog: log when fallback triggers so user can tune corr_chunk_size
        if oom_fallback_triggered:
            peak_mb = (
                torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0
            )
            import warnings

            warnings.warn(
                f"[GMFlowMatching] OOM triggered: fell back to chunk_size={active_chunk} "
                f"(was {self.chunk_size}). "
                f"Peak GPU memory: {peak_mb:.0f} MB. "
                f"Consider setting corr_chunk_size={active_chunk} in config.yaml to avoid this overhead.",
                RuntimeWarning,
                stacklevel=2,
            )
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        expected_pos = expected_pos.to(fL.dtype)
        conf = conf.to(fL.dtype)

        # Source coords for displacement (reuse cached grid)
        src_coords = (
            coords.view(H, W, 2).permute(2, 0, 1).unsqueeze(0).to(fL.dtype)
        )  # [1,2,H,W]
        flow = expected_pos - src_coords

        return flow, conf


class ConvexMaskUpsample(nn.Module):
    """Upsample flow using a learned convex combination of local neighbors.

    Channel assert: mask_ch must equal kernel² × upscale² at init time.
    Mask softmax runs in fp32 (autocast disabled) to prevent rounding that
    could shift learned weights and degrade sub-pixel flow quality.
    """

    def __init__(self, in_channels, kernel=3, upscale=2):
        super().__init__()
        self.kernel = kernel
        self.upscale = upscale
        self.mask_ch = (kernel * kernel) * (upscale * upscale)

        # Fail fast: catch accidental misconfiguration before any training step.
        assert self.mask_ch == (kernel**2) * (upscale**2), (
            f"ConvexMaskUpsample: mask_ch={self.mask_ch} ≠ "
            f"kernel²×upscale²={kernel**2}×{upscale**2}={kernel**2 * upscale**2}. "
            "Check kernel and upscale arguments."
        )

        self.mask_net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.mask_ch, 1),
        )

    def forward(self, flow_coarse, feat_coarse):
        B, C, Hc, Wc = flow_coarse.shape
        up = self.upscale
        k = self.kernel
        k2 = k * k
        Hs, Ws = Hc * up, Wc * up

        # Mask softmax must run in fp32 to prevent rounding-induced weight bias.
        with torch.amp.autocast("cuda", enabled=False):
            fc32 = feat_coarse.float()
            raw_mask = self.mask_net(fc32)  # [B, k2*up2, Hc, Wc]
            mask = raw_mask.view(B, k2, up * up, Hc, Wc)
            mask = torch.softmax(mask, dim=1)  # Softmax across k*k neighbors
            mask = mask.view(B, k2, up, up, Hc, Wc).permute(
                0, 1, 4, 2, 5, 3
            )  # [B, k2, Hc, up, Wc, up]
            mask = mask.contiguous().view(B, 1, k2, Hs, Ws)  # [B, 1, k2, Hs, Ws]

            # Extract k*k local neighborhood windows around coarse flow
            flow32 = flow_coarse.float()
            flow_unfold = F.unfold(
                flow32, kernel_size=k, padding=k // 2
            )  # [B, 2*k2, Hc*Wc]
            flow_unfold = flow_unfold.view(B, 2, k2, Hc, Wc)

            # Repeat for upsampled spatial grid
            flow_unfold = (
                flow_unfold.unsqueeze(4).unsqueeze(6).repeat(1, 1, 1, 1, up, 1, up)
            )
            flow_unfold = flow_unfold.view(B, 2, k2, Hs, Ws)

            # Convex combination
            flow_up = (mask * flow_unfold).sum(dim=2)  # [B, 2, Hs, Ws]

        return flow_up.to(flow_coarse.dtype) * float(up)


class MiddleFlowProjection(nn.Module):
    """Project bidirectional flows to intermediate time t=0.5.

    Contextualized by input sequence (flow maps + mean magnitudes + scaled feature).
    """

    def __init__(self, in_channels):
        super().__init__()
        hidden = 128
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 4, 3, padding=1),  # Output: ΔF_LM (2) + ΔF_RM (2)
        )

    def forward(self, flow_lr, flow_rl, feat):
        """
        Args:
            flow_lr: [B, 2, H, W] forward flow L→R
            flow_rl: [B, 2, H, W] backward flow R→L
            feat: [B, C, H, W] context features (usually fine level)
        """
        inp = torch.cat(
            [
                flow_lr,
                flow_rl,
                flow_lr.abs().mean(dim=1, keepdim=True),
                flow_rl.abs().mean(dim=1, keepdim=True),
                feat,
            ],
            dim=1,
        )

        delta = self.net(inp)
        delta_lm = delta[:, :2]
        delta_rm = delta[:, 2:]

        flow_lm = 0.5 * flow_lr + delta_lm
        flow_rm = 0.5 * flow_rl + delta_rm
        return flow_lm, flow_rm


class FlowRefinementStage(nn.Module):
    """Refines flow at a specific scale using context features."""

    def __init__(self, feat_dim, coarse_feat_dim, use_deformable=True):
        super().__init__()
        # Convex Mask upsampler from prior coarser stage features
        self.upsample = ConvexMaskUpsample(coarse_feat_dim)

        # Deformable refinement block predicting residual
        self.refine = DeformableRefinementBlock(feat_dim, use_deformable=use_deformable)

    def forward(self, flow_coarse, feat_coarse, feat_fine):
        """
        Args:
            flow_coarse: [B, 2, H/s, W/s] flow from previous coarser stage
            feat_coarse: [B, C', H/s, W/s] features at coarser stage (to drive mask)
            feat_fine: [B, C, H/(s/2), W/(s/2)] features at current finer stage
        Returns:
            flow_refined: [B, 2, H/(s/2), W/(s/2)]
        """
        # 1. Upsample flow rigorously using convex combination mask
        flow_up = self.upsample(flow_coarse, feat_coarse)

        # 2. Apply deformable refinement with context features at current scale
        flow_refined = self.refine(feat_fine, flow_up)
        return flow_refined


class FlowEstimator(nn.Module):
    """Complete optical flow estimation pipeline.

    1. Global correlation at 1/16 scale (GMFlow styled, robust numerics)
    2. Coarse-to-fine multi-scale refinement (1/8 -> 1/4 -> 1/2) with Convex Upsampling
    3. Final upsample to full res
    4. Middle flow projection
    """

    def __init__(
        self,
        use_deformable=True,
    ):
        super().__init__()

        # Feature projection to restrict dimensional compute explosion
        self.proj = nn.Sequential(
            nn.Conv2d(160, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.corr_proj_dim = 128

        self.correlation = GMFlowMatching(chunk_size=1024)

        # Coarse-to-fine refinement stages
        self.refine_8 = FlowRefinementStage(
            128, 160, use_deformable
        )  # Feats: f4(128), driving f5(160)
        self.refine_4 = FlowRefinementStage(
            96, 128, use_deformable
        )  # Feats: f3(96), driving f4(128)
        self.refine_2 = FlowRefinementStage(
            64, 96, use_deformable
        )  # Feats: f2(64), driving f3(96)

        # Final upsample from 1/2 to full-res
        self.final_upsample = ConvexMaskUpsample(64)

        # Full resolution flow confidence
        self.confidence_full = FlowConfidence(32 + 2)  # f1 is 32 channels

        # Middle flow projection: F_LR(2) + F_RL(2) + abs(2) + f_s1_feat(32) = 38
        self.middle_flow = MiddleFlowProjection(in_channels=38)

    def _estimate_single_flow(self, feats_src, feats_trg):
        """Estimate unidirectional flow from src to trg using feature pyramids.

        Args:
            feats_src: list of [f1, f2, f3, f4, f5] from encoder
            feats_trg: list of [f1, f2, f3, f4, f5] from encoder
        Returns:
            flow: [B, 2, H, W]
            conf: [B, 1, H, W]
        """
        f_s1, f_s2, f_s3, f_s4, f_s5 = feats_src
        f_t1, f_t2, f_t3, f_t4, f_t5 = feats_trg

        # 1. Base flow from correlation at 1/16 scale
        f_s5_proj = self.proj(f_s5)
        f_t5_proj = self.proj(f_t5)
        # correlation returns (flow, confidence)
        flow_16, conf_16 = self.correlation(f_s5_proj, f_t5_proj)  # [B, 2, H/16, W/16]

        # 2. Coarse-to-fine Refinement (using Convex Mask Upsampling)
        flow_8 = self.refine_8(flow_16, f_s5, f_s4)  # [B, 2, H/8, W/8]
        flow_4 = self.refine_4(flow_8, f_s4, f_s3)  # [B, 2, H/4, W/4]
        flow_2 = self.refine_2(flow_4, f_s3, f_s2)  # [B, 2, H/2, W/2]

        # 3. Final convex upsample to full resolution (H, W)
        flow_full = self.final_upsample(flow_2, f_s2)

        # 4. Confidence map prediction at full res
        conf_inp = torch.cat([f_s1, flow_full], dim=1)
        conf_full = self.confidence_full(conf_inp)

        return flow_full, conf_full

    def forward(self, features_L, features_R):
        """
        Args:
            features_L: list of [f1, f2, f3, f4, f5] from encoder(L)
            features_R: list of [f1, f2, f3, f4, f5] from encoder(R)
        Returns:
            flow_lr: [B, 2, H, W]
            flow_rl: [B, 2, H, W]
            flow_lm: [B, 2, H, W]
            flow_rm: [B, 2, H, W]
            conf_lr: [B, 1, H, W]
            conf_rl: [B, 1, H, W]
        """
        # Forward flow estimation
        flow_lr, conf_lr = self._estimate_single_flow(features_L, features_R)

        # Backward flow estimation
        flow_rl, conf_rl = self._estimate_single_flow(features_R, features_L)

        # Middle flow projection
        # Passing f1 (32 channels) from the L->R frame sequence as the semantic context
        flow_lm, flow_rm = self.middle_flow(flow_lr, flow_rl, features_L[0])

        return flow_lr, flow_rl, flow_lm, flow_rm, conf_lr, conf_rl
