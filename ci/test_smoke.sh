#!/bin/bash
# NTIRE 2026 GMTI-Net VFI CI Smoke Test
# Fails fast on regression

echo "Running GMTI-Net Smoke Tests..."
python -m pytest tests/test_numeric_bounds.py tests/test_flow_to_grid_and_warp.py tests/test_warping.py tests/test_correlation.py tests/test_forward.py -q

if [ $? -eq 0 ]; then
    echo "Smoke tests passed! Safe to train."
    exit 0
else
    echo "Regression detected. Do not train."
    exit 1
fi
