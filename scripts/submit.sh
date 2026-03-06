#!/usr/bin/env bash
# GMTI-Net NTIRE Submission Script
# Runs full inference pipeline:
#   1. Average last N EMA checkpoints
#   2. Self-ensemble (4 flips)
#   3. Multi-scale inference (1.0, 1.25)
#   4. Save results to OUTPUT_DIR
#
# Usage:
#   bash scripts/submit.sh [INPUT_DIR] [OUTPUT_DIR] [N_CHECKPOINTS]
#
# Defaults:
#   INPUT_DIR    = val
#   OUTPUT_DIR   = submission
#   N_CHECKPOINTS = 5

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT"

INPUT_DIR="${1:-val}"
OUTPUT_DIR="${2:-submission}"
N_CKPTS="${3:-5}"

echo "=============================="
echo " GMTI-Net NTIRE Submission"
echo " Input:        $INPUT_DIR"
echo " Output:       $OUTPUT_DIR"
echo " Avg checkpts: $N_CKPTS"
echo "=============================="

# ── Step 1: Full self-ensemble + multi-scale + checkpoint averaging ─────────
python inference.py \
  --config config.yaml \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --self_ensemble \
  --multiscale \
  --avg_checkpoints "$N_CKPTS"

echo ""
echo "[submit] Results saved to $OUTPUT_DIR"
echo ""

# ── Step 2: Quick PSNR sanity check (if GT available) ──────────────────────
# Uncomment if you have ground truth frames available alongside inputs.
# python validate.py \
#   --config config.yaml \
#   --checkpoint checkpoints/best_ema.pth

echo "[submit] Done. Zip $OUTPUT_DIR and upload to the NTIRE challenge portal."
