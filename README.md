# GMTI-Net VFI

**GMTI-Net** (Global Motion-guided Transformer Interpolation Network) — state-of-the-art Video Frame Interpolation for the NTIRE 2026 Challenge.

---

## ⚡ Quick Smoke Test (no dataset needed)

```bash
pip install -r requirements.txt
python -c "
import torch
from models.gmti_net import GMTINet
m = GMTINet(swin_depth=2, swin_heads=4, transformer_blocks=2, transformer_dim=64).eval()
L, R = torch.rand(1,3,128,128), torch.rand(1,3,128,128)
with torch.no_grad():
    pred = m.inference(L, R)
print('Output shape:', pred.shape, '| NaN:', torch.isnan(pred).any().item())
"
```

---

## 🚀 Running Locally

### 1. Requirements

Python 3.9+, CUDA 11.8+ (or CPU fallback).

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **PyTorch ≥ 2.6 note**: `torch.load` defaults to `weights_only=True`. The codebase uses `utils.io.safe_torch_load` which handles this transparently. Pin `torch==2.5.1` for maximum reproducibility.

### 2. Dataset Setup

```text
data/
  train/
    vid_1/
      frame_000001.png
      ...
  val/
    ...
```

Edit `config.yaml` → `data.train_dir` / `data.val_dir`.

For multi-dataset pretraining (Section 12 of the system spec):

```python
from datasets import NTIREDataset, Vimeo90KDataset, Adobe240Dataset, MixedDataset
from torch.utils.data import DataLoader

ntire  = NTIREDataset("data/ntire/train",  mode="train", crop_size=256)
vimeo  = Vimeo90KDataset("data/vimeo",     split="train", crop_size=256)
adobe  = Adobe240Dataset("data/adobe240",  split="train", crop_size=256)

mixed, sampler = MixedDataset.build(
    datasets=[vimeo, adobe, ntire],
    weights=[0.50, 0.30, 0.20],   # exact proportions via WeightedRandomSampler
    num_samples=100_000,
)
loader = DataLoader(mixed, batch_size=16, sampler=sampler, drop_last=True)
```

### 3. Training

```bash
# Dev run (fast, single GPU)
bash scripts/launch.sh light

# NTIRE full run (see config knobs below first)
bash scripts/launch.sh full

# Resume from checkpoint
bash scripts/launch.sh resume checkpoints/latest.pth

# Debug (deterministic, frequent visualisations)
bash scripts/launch.sh debug
```

Or directly:

```bash
python train.py --config config.yaml [--max_iters N] [--resume PATH] [--deterministic] [--debug]
```

### 4. Validation & Inference

```bash
python validate.py --config config.yaml --checkpoint checkpoints/best_ema.pth

# Full NTIRE submission pipeline (checkpoint avg + self-ensemble + multi-scale)
bash scripts/submit.sh val submission 5
```

---

## ☁️ Running on Google Colab

### Cell 1 — Mount Drive & set up

```python
from google.colab import drive
drive.mount('/content/drive')
!cp "/content/drive/MyDrive/GMTI-Net.zip" .
!unzip -q GMTI-Net.zip -d GMTI-Net
%cd GMTI-Net
!pip install -r requirements.txt -q
```

### Cell 2 — Smoke test

```python
!python -c "
import torch; from models.gmti_net import GMTINet
m = GMTINet(swin_depth=2, swin_heads=4, transformer_blocks=2, transformer_dim=64).eval()
L, R = torch.rand(1,3,128,128), torch.rand(1,3,128,128)
with torch.no_grad(): pred = m.inference(L, R)
print('OK — shape:', pred.shape)
"
```

### Cell 3 — Run tests

```python
!pytest tests/ -v --tb=short
```

### Cell 4 — Train

```python
!python train.py --config config.yaml --max_iters 10000
```

### Cell 5 — TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir logs
```

### Cell 6 — Back up to Drive

```python
!cp -r checkpoints "/content/drive/MyDrive/GMTI-Net-Checkpoints"
!cp -r visualizations "/content/drive/MyDrive/GMTI-Net-Visualizations"
!cp logs/run_info.json "/content/drive/MyDrive/GMTI-Net-run_info.json"
```

---

## ⚙️ Configuration Knobs

| Key                         | Default                     | Description                                                        |
| --------------------------- | --------------------------- | ------------------------------------------------------------------ |
| `model.encoder.swin_depth`  | **2** (dev) / **4** (NTIRE) | Swin Transformer depth. Set to 4 for full-quality                  |
| `model.encoder.swin_heads`  | **4** (dev) / **6** (NTIRE) | Swin heads. Set to 6 when swin_depth=4                             |
| `training.proj_dim`         | 128                         | Correlation feature projection dim. ↑ = better matching, more VRAM |
| `training.corr_chunk_size`  | 1024                        | Rows per chunk in GMFlowMatching. ↓ on OOM                         |
| `training.corr_temp`        | 0.1                         | Softmax temperature. Try 0.05–0.20                                 |
| `training.corr_topk`        | null                        | Future top-k correlation. null = disabled                          |
| `training.accumulate_steps` | 4                           | Gradient accumulation. **LR is kept constant** regardless          |
| `training.amp`              | true                        | AMP mixed precision. Disabled automatically on CPU                 |
| `training.ema_decay`        | 0.9999                      | EMA model shadow decay                                             |
| `inference.multiscale`      | [1.0, 1.25]                 | Scales for multi-scale test-time inference                         |
| `inference.checkpoint_avg`  | 5                           | Number of last checkpoints to average at inference                 |

### Light Dev vs NTIRE Full Run

| Setting      | Light Dev   | NTIRE Full            |
| ------------ | ----------- | --------------------- |
| `swin_depth` | 2           | **4**                 |
| `swin_heads` | 4           | **6**                 |
| `max_iters`  | 50 000      | **300 000**           |
| GPU          | 1× RTX 3090 | 4× RTX 4090 / 2× A100 |
| Crop size    | 256         | 256 → 512 (staged)    |

---

## 🔬 Determinism

```bash
python train.py --deterministic
```

Forces `cudnn.deterministic=True` and `cudnn.benchmark=False`. Expect **10–20% throughput reduction** on convolution-heavy ops. Use for debugging or exact reproducibility testing only.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

| Test file                       | Covers                                                       |
| ------------------------------- | ------------------------------------------------------------ |
| `test_correlation.py`           | GMFlow matching, shapes, no NaN                              |
| `test_warping.py`               | Backward warp, no NaN                                        |
| `test_forward.py`               | Full model forward, output shape                             |
| `test_flow_to_grid_and_warp.py` | Grid generation correctness                                  |
| `test_numeric_bounds.py`        | Chunking parity, convex channel assert, occlusion modes      |
| `test_smoke_train_resume.py`    | 2-iter train, checkpoint, resume                             |
| `test_average_checkpoints.py`   | Checkpoint averaging math                                    |
| `test_occlusion_eval_train.py`  | Eval uses only learned mask; train uses geometric modulation |
| `test_checkpoint_resume.py`     | Full round-trip: optimizer, scheduler, scaler, RNG           |
| `test_scaler_saved.py`          | GradScaler state persistence                                 |
| `test_mixed_dataset.py`         | MixedDataset proportions, deterministic sampling             |
| `test_datasets.py`              | Vimeo90K + Adobe240 item shapes, value ranges                |

---

## 📁 Repository Layout

```text
models/
  encoder.py          Hybrid CNN + Swin encoder
  flow_estimator.py   GMFlow correlation, convex upsample, middle flow
  warping.py          Backward warp (fp32-guarded) + dual warping
  occlusion.py        Occlusion CNN + geometric mask
  transformer.py      Motion-guided window transformer fusion
  decoder.py          Frequency-aware decoder + Laplacian merge
  gmti_net.py         Full model assembly

losses/
  reconstruction.py   Charbonnier + Laplacian pyramid
  flow_losses.py      Warping + bidirectional + smoothness + CombinedLoss

datasets/
  ntire_dataset.py    NTIRE triplet dataset
  vimeo90k.py         Vimeo-90K triplet dataset
  adobe240.py         Adobe-240FPS triplet dataset
  mixed.py            MixedDataset (WeightedRandomSampler)

utils/
  io.py               safe_torch_load + extract_model_state helpers

train.py              Training loop (EMA, AMP, env logging, determinism)
validate.py           Evaluation (PSNR, SSIM)
inference.py          Full inference pipeline (self-ensemble, multi-scale)
config.yaml           All hyperparameters
scripts/launch.sh     Local launch (light / full / resume / debug / smoke)
scripts/submit.sh     NTIRE submission pipeline
```
