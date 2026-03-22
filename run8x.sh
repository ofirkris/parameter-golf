#!/bin/bash
# Full 8xH100 submission run (10 min)
set -e
SEED=${1:-1337}
mkdir -p /dev/shm/data/datasets /dev/shm/data/tokenizers
cp -rn /mnt/vol/upstream/data/datasets/fineweb10B_sp1024 /dev/shm/data/datasets/ 2>/dev/null || true
cp -n /mnt/vol/upstream/data/tokenizers/fineweb_1024_bpe.model /dev/shm/data/tokenizers/ 2>/dev/null || true
cd /mnt/vol/ours
DATA_PATH=/dev/shm/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/dev/shm/data/tokenizers/fineweb_1024_bpe.model \
SEED=$SEED \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/run_8gpu_seed${SEED}.log
