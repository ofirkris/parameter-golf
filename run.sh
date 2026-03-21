#!/bin/bash
# =============================================================
# Parameter Golf — Single GPU Test Run
# Tests combo on 1xH100, ~5 min quick validation
# =============================================================
set -e

# Copy data to RAM for maximum I/O speed
echo "Loading data into RAM..."
mkdir -p /dev/shm/data/datasets /dev/shm/data/tokenizers
cp -r /workspace/data/datasets/fineweb10B_sp1024 /dev/shm/data/datasets/
cp /workspace/data/tokenizers/fineweb_1024_bpe.model /dev/shm/data/tokenizers/
echo "Data loaded to /dev/shm ($(du -sh /dev/shm/data | cut -f1) in RAM)"

cd /workspace/ours

# Quick test: 500 steps, 3 min cap, frequent logging
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=/dev/shm/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/dev/shm/data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=11 \
XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
SWA_ENABLED=0 \
ROPE_DIMS=16 \
LN_SCALE=1 \
LATE_QAT=1 \
TTT_ENABLED=1 \
BIGRAM_VOCAB_SIZE=10240 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 \
MAX_WALLCLOCK_SECONDS=300 \
ITERATIONS=500 \
VAL_LOSS_EVERY=100 \
TRAIN_LOG_EVERY=25 \
EVAL_STRIDE=64 \
python train_gpt.py 2>&1 | tee logs/test_1gpu.log
