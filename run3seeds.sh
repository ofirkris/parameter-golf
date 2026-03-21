#!/bin/bash
# =============================================================
# Parameter Golf — 3-Seed Validation (required for submission)
# Runs 3 seeds sequentially for statistical significance (p < 0.01)
# =============================================================
set -e

for seed in 42 1337 2024; do
    echo "=========================================="
    echo "=== SEED $seed ==="
    echo "=========================================="
    bash run8x.sh $seed
    echo ""
done

echo "=== All 3 seeds complete ==="
echo "Check logs/ for results. Need ≥0.005 nat improvement over SOTA and p < 0.01"
grep "final_int6_sliding_window_s64_exact" logs/run_8gpu_seed*.log
