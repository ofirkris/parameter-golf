#!/bin/bash
# =============================================================
# Parameter Golf — RunPod Setup Script
# Run once after spinning up a pod with 50GB Network Volume at /workspace
# Template: CUDA 13.x base (we install PyTorch nightly ourselves)
# =============================================================
set -e

echo "=== Phase 1: PyTorch nightly + CUDA 13.0 ==="
# Latest nightly with CUDA 13 for best torch.compile and SDPA performance
# CUDA 13.2 is latest toolkit (March 2026), PyTorch nightly has cu130 builds
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 2>&1 | tail -5

echo "=== Phase 2: Flash Attention 3 (Hopper/H100) ==="
# FA3 gives ~2x speedup over SDPA on H100
# Try the Dao-AILab Hopper build first (provides flash_attn_interface)
pip install flash-attn --no-build-isolation 2>&1 | tail -5 || true

# If flash_attn_interface not available, try building from source
python -c "from flash_attn_interface import flash_attn_func; print('FA3: OK')" 2>/dev/null || {
    echo "FA3 not in pip package, trying source build..."
    pip install ninja packaging 2>&1 | tail -2
    cd /tmp
    git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git 2>/dev/null || true
    cd flash-attention/hopper
    python setup.py install 2>&1 | tail -5 || echo "WARN: FA3 source build failed"
    cd /workspace
}

echo "=== Phase 3: Other dependencies ==="
pip install zstandard sentencepiece huggingface-hub datasets numpy tqdm 2>&1 | tail -3

echo "=== Phase 4: Download data (one-time, persists on network volume) ==="
DATA_DIR=/workspace/data/datasets/fineweb10B_sp1024

if [ -d "$DATA_DIR" ] && [ "$(ls $DATA_DIR/fineweb_train_*.bin 2>/dev/null | wc -l)" -ge 80 ]; then
    echo "Data already on network volume, skipping download"
else
    echo "Downloading training data..."
    mkdir -p /workspace/upstream
    cd /workspace/upstream
    if [ ! -d ".git" ]; then
        git clone --depth 1 https://github.com/openai/parameter-golf.git .
    fi
    python data/cached_challenge_fineweb.py
    # Move data to persistent location
    mkdir -p /workspace/data/datasets /workspace/data/tokenizers
    cp -r data/datasets/fineweb10B_sp1024 /workspace/data/datasets/
    cp data/tokenizers/fineweb_1024_bpe.model /workspace/data/tokenizers/
    cp data/tokenizers/fineweb_1024_bpe.vocab /workspace/data/tokenizers/ 2>/dev/null || true
    echo "Data download complete"
fi

echo "=== Phase 5: Clone our repo ==="
cd /workspace
if [ ! -d "/workspace/ours" ]; then
    git clone https://github.com/Tap-Mobile/parameter-golf.git ours 2>/dev/null || \
    git clone https://github.com/openai/parameter-golf.git ours
else
    cd /workspace/ours && git pull && cd /workspace
fi
mkdir -p /workspace/ours/logs

echo "=== Phase 6: Verify full stack ==="
python -c "
import torch, sentencepiece, zstandard, numpy
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.version.cuda}')
print(f'GPUs:     {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')
print(f'compile:  {hasattr(torch, \"compile\")}')

fa = 'NONE (SDPA fallback)'
try:
    from flash_attn_interface import flash_attn_func
    fa = 'FA3 (flash_attn_interface) — BEST'
except:
    try:
        from flash_attn import flash_attn_func
        import flash_attn
        fa = f'FA2 (v{flash_attn.__version__})'
    except:
        pass
print(f'FlashAttn: {fa}')
print(f'bf16:     {torch.cuda.is_bf16_supported()}')
"

echo ""
echo "=== Setup complete ==="
echo "Train shards: $(ls /workspace/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l)"
echo "Val shards:   $(ls /workspace/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin | wc -l)"
echo ""
echo "Next steps:"
echo "  Quick test:  bash /workspace/ours/run.sh"
echo "  Full run:    bash /workspace/ours/run8x.sh"
echo "  3-seed:      bash /workspace/ours/run3seeds.sh"
