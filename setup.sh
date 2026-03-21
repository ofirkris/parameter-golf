#!/bin/bash
# =============================================================
# Parameter Golf — RunPod Setup Script
# Run once after spinning up a pod with 50GB Network Volume at /workspace
# =============================================================
set -e

echo "=== Phase 1: Install dependencies ==="
pip install zstandard sentencepiece huggingface-hub datasets numpy tqdm 2>&1 | tail -3

# Flash Attention (H100 sm_90)
pip install flash-attn --no-build-isolation 2>&1 | tail -3 || echo "WARN: flash-attn failed, will use SDPA fallback"

echo "=== Phase 2: Download data (one-time, persists on network volume) ==="
DATA_DIR=/workspace/data/datasets/fineweb10B_sp1024
TOK_DIR=/workspace/data/tokenizers

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

echo "=== Phase 3: Clone our repo ==="
if [ ! -d "/workspace/ours" ]; then
    cd /workspace
    git clone https://github.com/Tap-Mobile/parameter-golf.git ours 2>/dev/null || \
    git clone https://github.com/openai/parameter-golf.git ours
fi

echo "=== Phase 4: Verify ==="
python -c "
import torch, sentencepiece, zstandard, numpy
print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')
try:
    from flash_attn_interface import flash_attn_func
    print('flash_attn_3 (FA3): OK')
except:
    try:
        from flash_attn import flash_attn_func
        import flash_attn
        print(f'flash_attn_2: OK (v{flash_attn.__version__})')
    except:
        print('flash_attn: NOT AVAILABLE (using SDPA fallback)')
"

echo ""
echo "=== Setup complete ==="
echo "Train shards: $(ls /workspace/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l)"
echo "Val shards:   $(ls /workspace/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin | wc -l)"
echo ""
echo "Next: run ./run.sh (1xH100 test) or ./run8x.sh (8xH100 full)"
