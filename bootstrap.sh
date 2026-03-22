#!/bin/bash
# =============================================================
# Parameter Golf - Verda Startup Script
# Paste into "Create new Startup Script" on verda.com
# Runs automatically on every new spot instance boot.
# Logs to /var/log/pgolf-setup.log - tail it after SSH-ing in.
# =============================================================

exec > >(tee -a /var/log/pgolf-setup.log) 2>&1
echo "=== Parameter Golf setup started at $(date) ==="

# ---- Volume ----
VOL=""
if mountpoint -q /mnt/vol 2>/dev/null; then VOL="/mnt/vol"
elif mountpoint -q /workspace 2>/dev/null; then VOL="/workspace"
else
  for dev in /dev/vdb /dev/vdc /dev/sdb /dev/nvme1n1; do
    if [ -b "$dev" ] && ! mount | grep -q "$dev"; then
      mkdir -p /mnt/vol
      blkid "$dev" &>/dev/null || mkfs.ext4 "$dev"
      mount "$dev" /mnt/vol
      VOL="/mnt/vol"
      break
    fi
  done
fi
[ -z "$VOL" ] && VOL="/root/pgolf" && echo "WARN: no persistent volume found!"
mkdir -p "$VOL"
echo "Volume: $VOL ($(df -h $VOL 2>/dev/null | tail -1 | awk '{print $4}') free)"

# ---- System packages ----
echo "=== Installing system packages ==="
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git screen 2>&1 | tail -3

# ---- PyTorch nightly ----
echo "=== Installing PyTorch nightly ==="
pip3 install --break-system-packages --pre -U torch --index-url https://download.pytorch.org/whl/nightly/cu130 2>&1 | tail -5 || \
pip3 install --break-system-packages --pre -U torch --index-url https://download.pytorch.org/whl/nightly/cu128 2>&1 | tail -5 || \
echo "WARN: PyTorch install failed"

# ---- Python deps ----
echo "=== Installing Python dependencies ==="
pip3 install --break-system-packages -q -U zstandard sentencepiece huggingface-hub datasets numpy tqdm 2>&1 | tail -3

# ---- Flash Attention ----
echo "=== Installing Flash Attention ==="
pip3 install --break-system-packages -U flash-attn --no-build-isolation 2>&1 | tail -5 || {
  echo "FA pip failed, trying source build..."
  pip3 install --break-system-packages ninja packaging 2>&1 | tail -2
  cd /tmp && rm -rf flash-attention
  git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git 2>/dev/null
  cd flash-attention/hopper && python3 setup.py install 2>&1 | tail -5 || \
  { cd /tmp/flash-attention && pip3 install --break-system-packages . 2>&1 | tail -5; } || \
  echo "WARN: Flash Attention failed - using SDPA fallback"
}

# ---- Training data (skip if already on volume) ----
echo "=== Training data ==="
DATA="$VOL/upstream/data/datasets/fineweb10B_sp1024"
SHARDS=$(ls "$DATA"/fineweb_train_*.bin 2>/dev/null | wc -l)
if [ "$SHARDS" -ge 80 ]; then
  echo "Data already on volume ($SHARDS shards), skipping download"
else
  echo "Downloading training data (~16GB)..."
  mkdir -p "$VOL/upstream"
  cd "$VOL/upstream"
  [ -d ".git" ] || git clone --depth 1 https://github.com/openai/parameter-golf.git .
  python3 data/cached_challenge_fineweb.py 2>&1 | tail -10
fi

# ---- Our code ----
echo "=== Setting up workspace ==="
mkdir -p "$VOL/ours/logs"
# Private repo - will fail without auth, thats OK
git clone https://github.com/Tap-Mobile/parameter-golf.git "$VOL/ours" 2>/dev/null || \
  (cd "$VOL/ours" 2>/dev/null && git pull 2>/dev/null) || \
  echo "Private repo needs auth - SCP train_gpt.py to $VOL/ours/ after SSH-ing in"

# ---- Load data to RAM ----
echo "=== Loading data to RAM ==="
mkdir -p /dev/shm/data/datasets /dev/shm/data/tokenizers
cp -r "$VOL/upstream/data/datasets/fineweb10B_sp1024" /dev/shm/data/datasets/ 2>/dev/null
cp "$VOL/upstream/data/tokenizers/fineweb_1024_bpe.model" /dev/shm/data/tokenizers/ 2>/dev/null

# ---- Verify ----
echo ""
echo "=========================================="
python3 -c "
import torch; fa='SDPA'
try:
 from flash_attn_interface import flash_attn_func; fa='FA3'
except:
 try:
  from flash_attn import flash_attn_func; fa='FA2'
 except: pass
print(f'PyTorch:   {torch.__version__}')
print(f'CUDA:      {torch.version.cuda}')
print(f'GPU:       {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')
print(f'FlashAttn: {fa}')
" 2>&1 || echo "WARN: PyTorch verification failed"
echo "Volume:    $VOL"
echo "Data RAM:  $(du -sh /dev/shm/data 2>/dev/null | cut -f1 || echo 'not loaded')"
echo "Shards:    $(ls /dev/shm/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)/80"
echo "Code:      $VOL/ours/train_gpt.py"
echo "Log:       /var/log/pgolf-setup.log"
echo "=========================================="
echo "=== Setup complete at $(date) ==="
echo ""
echo "To train: cd $VOL/ours && CUDA_VISIBLE_DEVICES=0 DATA_PATH=/dev/shm/data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=/dev/shm/data/tokenizers/fineweb_1024_bpe.model TORCH_COMPILE=0 python3 train_gpt.py"

# ---- Write convenience aliases ----
cat >> /root/.bashrc << 'ALIASES'

# Parameter Golf shortcuts
export VOL="/mnt/vol"
export DATA_PATH="/dev/shm/data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="/dev/shm/data/tokenizers/fineweb_1024_bpe.model"
export TORCH_COMPILE=0
alias pgolf="cd $VOL/ours"
alias pgtrain="cd $VOL/ours && CUDA_VISIBLE_DEVICES=0 python3 train_gpt.py"
alias pglog="tail -50 $VOL/ours/logs/*.log 2>/dev/null || echo 'no logs yet'"
alias pgstatus="nvidia-smi; echo '---'; cat /var/log/pgolf-setup.log | tail -20"
ALIASES
