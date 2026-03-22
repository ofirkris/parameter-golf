#!/bin/bash
# =============================================================
# Parameter Golf — Spot Instance Setup
# Works on any bare Ubuntu + NVIDIA GPU instance (Verda, RunPod, Lambda, etc.)
# Data goes to $VOL (block volume / network volume / persistent storage)
# Run: curl -sL https://raw.githubusercontent.com/Tap-Mobile/parameter-golf/main/setup.sh | bash
# =============================================================
set -e

# Auto-detect volume mount — check common locations
if mountpoint -q /mnt/vol 2>/dev/null; then
    VOL="/mnt/vol"
elif mountpoint -q /workspace 2>/dev/null; then
    VOL="/workspace"
else
    # Try to find and mount an unformatted block device (vdb, vdc, sdb, etc.)
    FOUND_DEV=""
    for dev in /dev/vdb /dev/vdc /dev/sdb /dev/sdc /dev/nvme1n1; do
        if [ -b "$dev" ] && ! mount | grep -q "$dev"; then
            FOUND_DEV="$dev"
            break
        fi
    done
    if [ -n "$FOUND_DEV" ]; then
        echo "=== Mounting block volume $FOUND_DEV ==="
        mkdir -p /mnt/vol
        blkid "$FOUND_DEV" >/dev/null 2>&1 || mkfs.ext4 "$FOUND_DEV"
        mount "$FOUND_DEV" /mnt/vol
        VOL="/mnt/vol"
    elif [ -d "/workspace" ]; then
        VOL="/workspace"
    else
        echo "WARN: No volume found, using /root/pgolf (NOT persistent!)"
        VOL="/root/pgolf"
    fi
fi
mkdir -p "$VOL"
echo "Volume: $VOL ($(df -h $VOL | tail -1 | awk '{print $4}') free)"

# ---- Phase 1: System packages ----
echo "=== Phase 1: System packages ==="
if ! command -v pip3 &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq python3-pip python3-venv git 2>&1 | tail -3
fi

# ---- Phase 2: PyTorch nightly (always upgrade to latest) ----
echo "=== Phase 2: PyTorch nightly ==="
pip3 install --break-system-packages --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cu130 2>&1 | tail -3 || \
pip3 install --break-system-packages --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cu128 2>&1 | tail -3
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"

# ---- Phase 3: Python deps ----
echo "=== Phase 3: Dependencies ==="
pip3 install --break-system-packages -q zstandard sentencepiece huggingface-hub datasets numpy tqdm 2>&1 | tail -3

# ---- Phase 4: Flash Attention (always try latest) ----
echo "=== Phase 4: Flash Attention ==="
pip3 install --break-system-packages --upgrade flash-attn --no-build-isolation 2>&1 | tail -5 || {
    echo "pip flash-attn failed, trying FA3 source build..."
    pip3 install --break-system-packages ninja packaging 2>&1 | tail -2
    cd /tmp
    rm -rf flash-attention
    git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention/hopper && python3 setup.py install 2>&1 | tail -5 || echo "FA3 source failed"
    cd /tmp/flash-attention && pip3 install --break-system-packages . 2>&1 | tail -5 || echo "FA2 also failed — using SDPA"
    cd "$VOL"
}
python3 -c "
fa = 'NONE (SDPA fallback)'
try:
    from flash_attn_interface import flash_attn_func; fa = 'FA3'
except:
    try:
        from flash_attn import flash_attn_func; import flash_attn; fa = f'FA2 v{flash_attn.__version__}'
    except: pass
print(f'FlashAttn: {fa}')
"

# ---- Phase 5: Data (persistent on volume) ----
echo "=== Phase 5: Training data ==="
DATA_DIR="$VOL/data/datasets/fineweb10B_sp1024"
TOK_DIR="$VOL/data/tokenizers"

if [ -d "$DATA_DIR" ] && [ "$(ls $DATA_DIR/fineweb_train_*.bin 2>/dev/null | wc -l)" -ge 80 ]; then
    echo "Data already on volume ($(du -sh $DATA_DIR | cut -f1)), skipping download"
else
    echo "Downloading training data (~16GB)..."
    mkdir -p "$VOL/upstream"
    cd "$VOL/upstream"
    [ -d ".git" ] || git clone --depth 1 https://github.com/openai/parameter-golf.git .
    python3 data/cached_challenge_fineweb.py
    mkdir -p "$VOL/data/datasets" "$VOL/data/tokenizers"
    cp -r data/datasets/fineweb10B_sp1024 "$VOL/data/datasets/"
    cp data/tokenizers/fineweb_1024_bpe.model "$TOK_DIR/"
    cp data/tokenizers/fineweb_1024_bpe.vocab "$TOK_DIR/" 2>/dev/null || true
    echo "Data download complete"
fi

# ---- Phase 6: Our code (persistent on volume) ----
echo "=== Phase 6: Code ==="
cd "$VOL"
if [ -d "$VOL/ours/.git" ]; then
    cd "$VOL/ours" && git pull 2>&1 | tail -3 && cd "$VOL"
else
    git clone https://github.com/Tap-Mobile/parameter-golf.git ours 2>&1 | tail -3 || {
        echo "Private repo failed, using upstream"
        git clone --depth 1 https://github.com/openai/parameter-golf.git ours
    }
fi
mkdir -p "$VOL/ours/logs"

# ---- Phase 7: Load data to RAM ----
echo "=== Phase 7: Loading data to RAM (/dev/shm) ==="
mkdir -p /dev/shm/data/datasets /dev/shm/data/tokenizers
cp -r "$DATA_DIR" /dev/shm/data/datasets/
cp "$TOK_DIR/fineweb_1024_bpe.model" /dev/shm/data/tokenizers/
echo "Data in RAM: $(du -sh /dev/shm/data | cut -f1)"

# ---- Phase 8: Verify ----
echo ""
echo "=========================================="
echo "=== SETUP COMPLETE ==="
echo "=========================================="
python3 -c "
import torch
print(f'PyTorch:   {torch.__version__}')
print(f'CUDA:      {torch.version.cuda}')
print(f'GPUs:      {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')
fa = 'NONE (SDPA fallback)'
try:
    from flash_attn_interface import flash_attn_func; fa = 'FA3 (best)'
except:
    try:
        from flash_attn import flash_attn_func; import flash_attn; fa = f'FA2 (v{flash_attn.__version__})'
    except: pass
print(f'FlashAttn: {fa}')
print(f'bf16:      {torch.cuda.is_bf16_supported()}')
print(f'compile:   {hasattr(torch, \"compile\")}')
"
echo ""
echo "Volume:     $VOL"
echo "Code:       $VOL/ours/train_gpt.py"
echo "Data (RAM): /dev/shm/data/datasets/fineweb10B_sp1024"
echo "Tokenizer:  /dev/shm/data/tokenizers/fineweb_1024_bpe.model"
echo "Train shards: $(ls /dev/shm/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l)"
echo "Val shards:   $(ls /dev/shm/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin | wc -l)"
echo ""
echo "Quick test (1 GPU, 5 min):"
echo "  cd $VOL/ours && CUDA_VISIBLE_DEVICES=0 DATA_PATH=/dev/shm/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/dev/shm/data/tokenizers/fineweb_1024_bpe.model TORCH_COMPILE=0 MAX_WALLCLOCK_SECONDS=300 ITERATIONS=500 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=25 python3 train_gpt.py"
echo ""
echo "Full 8xH100 run (10 min):"
echo "  cd $VOL/ours && DATA_PATH=/dev/shm/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/dev/shm/data/tokenizers/fineweb_1024_bpe.model TORCH_COMPILE=0 torchrun --standalone --nproc_per_node=8 train_gpt.py"
