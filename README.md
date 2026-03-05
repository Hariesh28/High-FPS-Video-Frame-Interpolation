# GMTI-Net VFI

This repository implements **GMTI-Net**, a state-of-the-art Video Frame Interpolation (VFI) system. The implementation has been rigorously optimized for stable, single-GPU training, robust performance on the NTIRE dataset, and features a clean Window Transformer Fusion & Global Correlation pipeline.

---

## 🚀 Running Locally

### 1. Requirements

Ensure you have Python 3.9+ and CUDA 11.8+ installed.

```bash
# Clone the repository and cd into it
# Create a virtual environment
python -m venv venv
# Activate the environment
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Compatibility note

PyTorch 2.6 changed the default behavior of torch.load to be "weights-only" which
restricts unpickling to tensors and a small safe-list of globals. This repository
is defensive and requests weights-only loads when only model weights are required,
and explicitly requests full checkpoint loading when resuming training (to restore
optimizer/scaler/RNG state). For reproducible runs you can pin a tested version
such as `torch==2.5.1` in your environment. See `requirements.txt` for guidance.

### 2. Dataset Setup

Extract the NTIRE dataset. Make sure the structure looks like:

```text
data/
  train/
    vid_1/
      frame_000001.png
      ...
  val/
    ...
```

Update `config.yaml` to point to your `train` and `val` directories.

### 3. Training & Validation

To start training with the optimized pipeline (GMFlow correlation, deepcopy EMA, strict gradient clipping, Amp):

```bash
python train.py --config config.yaml
```

**Features during training:**

- Checkpoints (incl. EMA shadow weights) are saved to `checkpoints/`.
- Training metrics (Loss, exact PSNR, Gradient Norm) stream to TensorBoard (`logs/`).
- Flow debug visualization grids are saved to `visualizations/iter_XXXX000.jpg` every 1,000 steps to detect silent failures (e.g. ghosting, pure black flows).

To strictly evaluate a checkpoint (which uses EMA weights by default):

```bash
python validate.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

To run inference predicting an intermediate test frame:

```bash
# L.png and R.png are the left and right context frames
python inference.py --left L.png --right R.png --output M_pred.png --checkpoint checkpoints/best_model.pth
```

---

## ☁️ Running on Google Colab

The project is natively compatible with Google Colab's standard T4 or A100 environments.

### Step-by-Step

1. Zip your project folder (`GMTI-Net.zip`) and upload it to your Google Drive.
2. Open a new Notebook on Google Colab, and change your runtime type to **T4 GPU** or higher.
3. Run the following cells in order:

**Cell 1: Mount Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2: Copy and Extract the Project**

```python
# Assuming you uploaded GMTI-Net.zip to the root of your Google Drive
!cp "/content/drive/MyDrive/GMTI-Net.zip" .
!unzip -q GMTI-Net.zip -d GMTI-Net
%cd GMTI-Net
```

**Cell 3: Install Dependencies**

```python
# PyTorch is pre-installed in Colab, but you need a few extras
!pip install -r requirements.txt
```

**Cell 4: Run Training**

```python
# Note: You can upload your dataset zip to Drive and extract it here too.
# Edit config.yaml beforehand or use sed commands to patch paths if needed.
!python train.py --config config.yaml
```

**Cell 5: Monitor Progress (TensorBoard in Colab)**

```python
%load_ext tensorboard
%tensorboard --logdir logs
```

**Bonus: Zip Checkpoints / Visualizations automatically to your drive**

```python
# Safely copy artifacts back to permanent storage!
!cp -r checkpoints "/content/drive/MyDrive/GMTI-Net-Checkpoints"
!cp -r visualizations "/content/drive/MyDrive/GMTI-Net-Visualizations"
```
