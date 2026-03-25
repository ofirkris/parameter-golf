#!/bin/bash
# Full 8xH100 submission run — every knob pinned explicitly
set -e
SEED=${1:-1337}
mkdir -p /dev/shm/data/datasets /dev/shm/data/tokenizers
cp -rn /mnt/vol/upstream/data/datasets/fineweb10B_sp1024 /dev/shm/data/datasets/ 2>/dev/null || true
cp -n /mnt/vol/upstream/data/tokenizers/fineweb_1024_bpe.model /dev/shm/data/tokenizers/ 2>/dev/null || true
cd /mnt/vol/ours

DATA_PATH=/dev/shm/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/dev/shm/data/tokenizers/fineweb_1024_bpe.model \
SEED=$SEED \
NUM_LAYERS=10 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3.0 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 \
WARMDOWN_ITERS=3000 \
MAX_WALLCLOCK_SECONDS=600 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WEIGHT_DECAY=0.04 \
BIGRAM_VOCAB_SIZE=10240 \
XSA_LAST_N=10 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VALUE_RESIDUAL=1 \
VE_ENABLED=0 \
VE_DIM=128 \
VE_LAYERS=8,9 \
LATE_QAT=1 \
QAT_THRESHOLD=0.15 \
TTT_EPOCHS=30 \
TTT_LR=0.0005 \
TTT_BATCH_SEQS=32 \
USE_DWA=0 \
NGRAM_CACHE=1 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/run_8gpu_seed${SEED}.log
