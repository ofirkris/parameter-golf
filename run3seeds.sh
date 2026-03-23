#!/bin/bash
set -e
for seed in 42 1337 2024; do
    echo "=========================================="
    echo "=== SEED $seed ==="
    echo "=========================================="
    bash run8x.sh $seed
done
echo "=== All 3 seeds complete ==="
grep "final_roundtrip_exact\|ttt:done.*ttt_val_bpb" logs/run_8gpu_seed*.log || true
