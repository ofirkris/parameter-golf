#!/bin/bash
# Paste this entire block into a fresh spot instance terminal.
# It sets up everything without needing access to the private repo.

set -e

# ---- Volume ----
if mountpoint -q /mnt/vol 2>/dev/null; then VOL="/mnt/vol"
elif mountpoint -q /workspace 2>/dev/null; then VOL="/workspace"
else
  for dev in /dev/vdb /dev/vdc /dev/sdb /dev/nvme1n1; do
    if [ -b "$dev" ] && ! mount | grep -q "$dev"; then
      mkdir -p /mnt/vol; blkid "$dev" &>/dev/null || mkfs.ext4 "$dev"; mount "$dev" /mnt/vol; VOL="/mnt/vol"; break
    fi
  done
fi
[ -z "$VOL" ] && VOL="/root/pgolf" && echo "WARN: no volume"
mkdir -p "$VOL"
echo "VOL=$VOL ($(df -h $VOL | tail -1 | awk '{print $4}') free)"

# ---- System deps ----
command -v pip3 &>/dev/null || { apt-get update -qq && apt-get install -y -qq python3-pip git; }

# ---- PyTorch nightly (always latest) ----
pip3 install --break-system-packages --pre -U torch --index-url https://download.pytorch.org/whl/nightly/cu130 2>&1 | tail -3 || \
pip3 install --break-system-packages --pre -U torch --index-url https://download.pytorch.org/whl/nightly/cu128 2>&1 | tail -3

# ---- Deps ----
pip3 install --break-system-packages -q -U zstandard sentencepiece huggingface-hub datasets numpy tqdm

# ---- Flash Attention (latest) ----
pip3 install --break-system-packages -U flash-attn --no-build-isolation 2>&1 | tail -5 || echo "FA pip failed, SDPA fallback"

# ---- Data (skip if already on volume) ----
DATA="$VOL/upstream/data/datasets/fineweb10B_sp1024"
if [ "$(ls $DATA/fineweb_train_*.bin 2>/dev/null | wc -l)" -lt 80 ]; then
  cd "$VOL"; [ -d upstream/.git ] || git clone --depth 1 https://github.com/openai/parameter-golf.git upstream
  cd "$VOL/upstream" && python3 data/cached_challenge_fineweb.py
fi

# ---- Our code (pull latest) ----
cd "$VOL"
[ -d ours/.git ] && (cd ours && git pull) || git clone https://github.com/Tap-Mobile/parameter-golf.git ours 2>/dev/null || echo "Private repo needs auth — SCP train_gpt.py manually"
mkdir -p "$VOL/ours/logs"

# ---- Data to RAM ----
mkdir -p /dev/shm/data/datasets /dev/shm/data/tokenizers
cp -r "$VOL/upstream/data/datasets/fineweb10B_sp1024" /dev/shm/data/datasets/
cp "$VOL/upstream/data/tokenizers/fineweb_1024_bpe.model" /dev/shm/data/tokenizers/

# ---- Verify ----
python3 -c "
import torch; fa='SDPA'
try:
 from flash_attn_interface import flash_attn_func; fa='FA3'
except:
 try:
  from flash_attn import flash_attn_func; fa='FA2'
 except: pass
print(f'torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)} fa={fa}')
"
echo "Shards: $(ls /dev/shm/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin|wc -l)/80"
echo "Code: $VOL/ours/train_gpt.py"
echo "READY. Run: cd $VOL/ours && CUDA_VISIBLE_DEVICES=0 DATA_PATH=/dev/shm/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/dev/shm/data/tokenizers/fineweb_1024_bpe.model TORCH_COMPILE=0 python3 train_gpt.py"
