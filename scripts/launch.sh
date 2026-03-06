#!/usr/bin/env bash
# GMTI-Net Launch Script
# Usage: bash scripts/launch.sh [mode]
# Modes: light (default), full, resume, debug

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT"

MODE="${1:-light}"

echo "=============================="
echo " GMTI-Net Launch — mode: $MODE"
echo " Working dir: $PROJECT"
echo "=============================="

case "$MODE" in

  # ── Light dev config: single GPU, swin_depth=2, quick iteration ──────────
  light)
    echo "[light] Single-GPU dev run (swin_depth=2, 50k iters)"
    python train.py \
      --config config.yaml \
      --max_iters 50000 \
      --seed 2026
    ;;

  # ── Full NTIRE run: edit config.yaml to set swin_depth=4, swin_heads=6 ──
  # Recommended: 4×RTX4090 or 2×A100.
  # Pretraining schedule: 300k → 200k mixed → 100k NTIRE fine-tune.
  # Set swin_depth=4, swin_heads=6 in config.yaml before running.
  full)
    echo "[full] NTIRE full-run (edit config.yaml: swin_depth=4, swin_heads=6)"
    python train.py \
      --config config.yaml \
      --max_iters 300000 \
      --seed 2026
    ;;

  # ── Resume from latest checkpoint ────────────────────────────────────────
  resume)
    CKPT="${2:-checkpoints/latest.pth}"
    echo "[resume] Resuming from $CKPT"
    python train.py \
      --config config.yaml \
      --resume "$CKPT" \
      --seed 2026
    ;;

  # ── Debug mode: frequent visualisations, deterministic ───────────────────
  debug)
    echo "[debug] Debug run (20 iters, deterministic, vis every 10)"
    python train.py \
      --config config.yaml \
      --max_iters 20 \
      --deterministic \
      --debug \
      --seed 42
    ;;

  # ── Smoke test (no dataset needed) ───────────────────────────────────────
  smoke)
    echo "[smoke] CPU forward-pass smoke test"
    python -c "
import torch
from models.gmti_net import GMTINet
m = GMTINet(swin_depth=2, swin_heads=4, transformer_blocks=2, transformer_dim=64).eval()
L = torch.rand(1, 3, 128, 128)
R = torch.rand(1, 3, 128, 128)
with torch.no_grad():
    pred = m.inference(L, R)
assert pred.shape == (1, 3, 128, 128), f'Bad shape: {pred.shape}'
assert not torch.isnan(pred).any(), 'NaN in output'
print('Smoke test PASSED — output shape:', pred.shape)
"
    ;;

  *)
    echo "Unknown mode '$MODE'. Choose: light | full | resume | debug | smoke"
    exit 1
    ;;
esac
