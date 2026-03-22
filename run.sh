#!/bin/bash
# Quick 1xH100 test (500 steps, 5 min)
set -e
mkdir -p /dev/shm/data/datasets /dev/shm/data/tokenizers
cp -rn /mnt/vol/upstream/data/datasets/fineweb10B_sp1024 /dev/shm/data/datasets/ 2>/dev/null || true
cp -n /mnt/vol/upstream/data/tokenizers/fineweb_1024_bpe.model /dev/shm/data/tokenizers/ 2>/dev/null || true
cd /mnt/vol/ours
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=/dev/shm/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/dev/shm/data/tokenizers/fineweb_1024_bpe.model \
MAX_WALLCLOCK_SECONDS=300 \
ITERATIONS=500 \
VAL_LOSS_EVERY=100 \
TRAIN_LOG_EVERY=25 \
EVAL_STRIDE=64 \
python3 train_gpt.py 2>&1 | tee logs/test_1gpu.log
