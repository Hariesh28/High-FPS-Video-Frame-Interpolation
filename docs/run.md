## Running GMTI-Net (local + Colab)

This document explains how to run the GMTI-Net repository locally (Windows PowerShell) and on Google Colab. It also describes where you will see progress, which artifacts are produced, and common troubleshooting tips.

Sections:

- Local (Windows PowerShell)
- Google Colab
- What progress you see (console / TensorBoard / artifacts)
- Quick smoke checks and tests
- Troubleshooting and reproducibility notes

---

## Local (Windows PowerShell)

Prerequisites

- Python 3.9+ (3.10 / 3.11 are fine)
- A GPU and matching CUDA drivers (optional but recommended). If no GPU, the code runs on CPU.
- Git (optional)

Create and activate a virtual environment (PowerShell):

```powershell
# Create a venv in the repo root
python -m venv .venv

# Activate the venv (PowerShell)
.\.venv\Scripts\Activate.ps1

# If your PowerShell blocks activation, set this once (user scope):
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Notes on PyTorch: the repository is defensive about torch.load behavior across versions.
If you want strict reproducibility, pin a tested version (for example `torch==2.5.1`) in `requirements.txt` before installing.

Edit the config

- Open `config.yaml` and set `data.train_dir` and `data.val_dir` to your dataset paths.
- For quick local debugging set smaller values (e.g. `training.max_iters: 10`, `training.checkpoint_freq: 1`).

Run a short training session (smoke run):

```powershell
python train.py --config config.yaml --max_iters 10
```

Resume training from latest checkpoint:

```powershell
python train.py --config config.yaml --resume checkpoints/latest.pth
```

Run validation on a checkpoint:

```powershell
python validate.py --config config.yaml --checkpoint checkpoints/best_ema.pth
```

Run inference and save predictions:

```powershell
python inference.py --config config.yaml --input_dir val --output_dir results --checkpoint checkpoints/best_ema.pth

# Or average the last 5 checkpoints and use the averaged weights
python inference.py --config config.yaml --input_dir val --output_dir results --avg_checkpoints 5
```

Run the tests (quick smoke):

```powershell
# Run the smoke train/resume test only
python -m pytest tests/test_smoke_train_resume.py -q

# Run the full pytest suite
python -m pytest -q
```

---

## Google Colab (quick notebook steps)

Notes: choose a GPU runtime (T4, P100, or A100). Save large artifacts (checkpoints/visualizations) back to your Google Drive.

Example Colab cell sequence (copy into a notebook):

1. Mount Drive and copy repo (or clone/upload it)

```python
from google.colab import drive
drive.mount('/content/drive')

# If you uploaded a zip to Drive, copy & extract it
!cp "/content/drive/MyDrive/GMTI-Net.zip" .
!unzip -q GMTI-Net.zip -d GMTI-Net
%cd GMTI-Net
```

2. Install dependencies

```python
!pip install -r requirements.txt
# Optionally install a specific torch wheel that matches the runtime CUDA version.
# Example (adjust for the desired CUDA):
#!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. (Optional) copy a small dataset from Drive for testing

```python
!cp -r /content/drive/MyDrive/NTIRE_subset ./data
```

4. Run training or smoke test

```python
!python train.py --config config.yaml --max_iters 20
```

5. View TensorBoard inside the notebook

```python
%load_ext tensorboard
%tensorboard --logdir logs
```

6. Save outputs back to Drive

```python
!cp -r checkpoints "/content/drive/MyDrive/GMTI-Net-Checkpoints"
!cp -r visualizations "/content/drive/MyDrive/GMTI-Net-Visualizations"
```

---

## What progress you will see

Console (stdout)

- The training loop prints a tqdm progress bar with a small per-iteration postfix displaying:
  - loss
  - psnr (computed exactly using -10\*log10(MSE))
  - lr

Example tqdm postfix (approx):

```
Training:  50%|█████     | 5/10 [00:10<00:10, loss=1.2489, psnr=9.07, lr=1.00e-06]
```

TensorBoard

- TensorBoard is written to `logs/` by `torch.utils.tensorboard.SummaryWriter`.
- Scalars available:
  - train/loss
  - train/psnr
  - train/lr
  - train/grad_norm (when computed)
  - train/loss\_<component> for individual loss terms
  - val/psnr

Artifacts on disk

- `checkpoints/` contains checkpoint files saved atomically:
  - `iter_{i}.pth` — checkpoint at iteration i
  - `latest.pth` — atomic copy of latest checkpoint
  - `best_ema.pth` — EMA weights when validation improved

Each checkpoint contains:

- `model` (state_dict)
- `ema` (state_dict)
- `optimizer` (state_dict)
- `scheduler` (state_dict)
- `scaler` (GradScaler state)
- `iteration` (int)
- RNG states (torch_rng, cuda_rng if present), numpy RNG and python RNG

- `visualizations/` stores flow visualization grids (saved every 1000 iters by default)
- `results/` contains inference outputs (when you run `inference.py`)

---

## Quick smoke checks (what to run right now)

- Unit tests: `python -m pytest -q` (should pass with the repo changes applied)
- Smoke training & resume: `python -m pytest tests/test_smoke_train_resume.py -q`

---

## Troubleshooting and tips

- DataLoader pin_memory warning when no GPU:
  - Warning: `pin_memory` argument is set as true but no accelerator is found.
  - This is harmless. To silence it set `data.num_workers=0` or `pin_memory=False` in config.

- torch.load weights-only / unpickle errors (PyTorch 2.6+):
  - PyTorch 2.6 changed `torch.load` default to weights-only safe loads which restrict unpickling.
  - The codebase includes `utils/io.py::safe_torch_load` that prefers weights-only loads where appropriate
    (inference/averaging) and explicitly requests full loads where necessary (training resume). You should
    not need to change anything in normal use.
  - If you hit an UnpicklingError when loading a trusted checkpoint, you can set the environment variable
    `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` to force full loads by default (only do this with trusted checkpoints).

- GPU OOM / memory problems:
  - Reduce `training.batch_size` or `training.crop_size` in `config.yaml`.
  - Use gradient accumulation: `training.accumulate_steps`.

- Reproducibility:
  - We save RNG states in checkpoints and `train.py` restores them when resuming.
  - For exact reproduction across runs, pin the exact PyTorch and other library versions.

---

## Additional utilities

- `utils/io.py::safe_torch_load(path, map_location, weights_only)` is a small helper which the repo uses.
  - Use it when writing new scripts that load checkpoints; it centralizes the weights-only fallback logic.

- `scripts/average_checkpoints.py` provides a small CLI to average checkpoints by weight and save an averaged file.

---

If you'd like, I can add a ready-to-run Colab notebook (.ipynb) with the cells above and an example small dataset so you can click-run it immediately. Or I can add `docs/colab.ipynb` to this repo.
