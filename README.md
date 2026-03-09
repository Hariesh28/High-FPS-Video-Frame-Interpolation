# GMTI-Net V3.1 Pro

**GMTI-Net** (Global Motion-guided Transformer Interpolation Network) is a state-of-the-art Video Frame Interpolation (VFI) system designed for the NTIRE 2026 Challenge. Version 3.1 'Pro' introduces advanced motion modeling and stability refinements to maximize PSNR.

---

## ✨ V3.1 Pro Features

- **Quadratic Motion**: Models local acceleration (second-order motion) for parabolic trajectories.
- **Multi-Hypothesis Flow ($K=3$)**: Predicts multiple candidates with confidence-weighted fusion.
- **Frequency-Focal Loss**: Band-weighted DCT supervision with high-frequency annealing.
- **Heteroscedastic Uncertainty**: Stable log-sigma formulation for variance-weighted regression.
- **Structural Stability**: Normalized VGG features to ensure MSE-dominant (PSNR) optimization.
- **Advanced Inference**: Snapshot ensembling and post-hoc linear color calibration.

---

## 🚀 Getting Started

### 1. Requirements

Python 3.9+, CUDA 11.8+.

```bash
pip install -r requirements.txt
```

### 2. Quick Smoke Test

Verify the architecture and numeric stability without heavy datasets:

```bash
python tests/test_v3_pro_refined_smoke.py
```

### 3. Training Curriculum

GMTI-Net is trained in three stages for optimal convergence:

1. **Stage 1 (Base)**: Core reconstruction with Charbonnier & Laplacian losses.
2. **Stage 2 (Pro Warmup)**: Introduction of flow-refinement and multi-hypothesis candidates.
3. **Stage 3 (Pro Fine-tune)**: High-PSNR focus with DCT annealing, heteroscedastic loss, and temperature sharpening.

```bash
bash scripts/launch.sh full  # Executes the full curriculum
```

---

## 🧪 Testing Infrastructure

We maintain a professional testing suite to ensure numeric stability and performance.

```bash
pytest tests/ -v
```

| Test File                            | Description                                           |
| ------------------------------------ | ----------------------------------------------------- |
| `tests/test_v3_pro_refined_smoke.py` | Numeric stability of Pro losses and annealing logic.  |
| `tests/test_correlation.py`          | GMFlow matching and chunking parity at 1/16 scale.    |
| `tests/test_numeric_bounds.py`       | FP32 guards and occlusion modulation checks.          |
| `tests/test_checkpoint_resume.py`    | Reliability of the optimizer/scaler state round-trip. |

---

## 📁 Repository Structure

```text
models/
  gmti_net.py         Full model assembly and orchestration.
  flow_estimator.py   GMFlow matching + Pro heads (Quadratic, K-Flow).
  warping.py          Dual-warping (Image+Feature) and kernels (FP32).
  encoder.py          Hybrid CNN-Swin hierarchical encoder.
  transformer.py      Global motion-guided fusion transformer.
  decoder.py          Frequency-aware Laplacian decoder.

losses/
  flow_losses.py      CombinedLoss + Pro supervisions (DCT, Hetero).
  reconstruction.py   Base spatial reconstruction losses.

scripts/
  benchmark.py        Final inference engine (Snapshots, Color Calib).
  prepare_submission.py Generates NTIRE-formatted ZIP files.
  launch.sh           SLURM/Local training launch utility.

datasets/             Optimized loader implementations for VFI.
tests/                Comprehensive unit and smoke tests.
config.yaml           Professional hyperparameter configuration.
```

---

## 🏆 Submission

To generate a final submission for NTIRE 2026:

```bash
python scripts/benchmark.py --input_dir data/test --output_dir results --snapshots checkpoints/snap_*.pth
python scripts/prepare_submission.py --results_dir results --output Submission.zip
```
