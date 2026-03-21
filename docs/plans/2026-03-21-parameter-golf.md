# Parameter Golf Challenge — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Achieve competitive BPB on the OpenAI Parameter Golf challenge — train the best language model that fits in a 16MB artifact in under 10 minutes on 8xH100.

**Architecture:** Start from the official baseline GPT (~1.2244 BPB), progressively apply high-impact optimizations in order of expected BPB improvement: sliding window eval, deeper model (10-12L), wider MLP (3x), int6/int5 QAT quantization, zstd-22 compression, SmearGate, BigramHash, SWA, EMA, and test-time training. Each optimization is validated independently before stacking.

**Tech Stack:** PyTorch, FlashAttention, SentencePiece, CUDA, torchrun (DDP), zstd compression

**Hardware:**
- **Development:** Single B200 GPU (gpu7, b200 server) — ~2x H100 compute, 112GB free VRAM
- **Submission validation:** 8xH100 via RunPod (challenge provides $1M credits pool)

**Current SOTA:** ~1.1428 BPB (merged), ~1.1248 BPB (open PRs)
**Baseline:** 1.2244 BPB

---

## Phase 0: Environment Setup

### Task 0.1: Clone challenge repo and download data on b200

**Files:**
- Create: `/storage/tap/parameter-golf/` (working dir on b200)

**Step 1: Set up workspace on b200**

```bash
ssh b200
mkdir -p /storage/tap/parameter-golf
cd /storage/tap/parameter-golf
git clone https://github.com/openai/parameter-golf.git upstream
cd upstream
```

**Step 2: Install dependencies**

```bash
# Use system Python or create a venv
pip install -r requirements.txt
# Ensure zstandard is installed for later compression work
pip install zstandard
```

**Step 3: Download tokenized training data**

```bash
python data/cached_challenge_fineweb.py
```

Expected: Downloads ~80 training shards + validation shards to `data/` directory.

**Step 4: Verify data**

```bash
ls -la data/fineweb_train_*.bin | wc -l  # expect 80
ls -la data/fineweb_val_*.bin | wc -l    # expect validation shards
```

**Step 5: Set up our working repo**

```bash
# Also clone our private repo for tracking
cd /storage/tap/parameter-golf
git clone git@github.com:Tap-Mobile/parameter-golf.git ours
cp upstream/train_gpt.py ours/train_gpt.py
cd ours
git add train_gpt.py && git commit -m "feat: copy baseline train_gpt.py from upstream"
```

**Step 6: Commit**

```bash
git push origin main
```

---

### Task 0.2: Run baseline on single GPU

**Files:**
- None (validation run)

**Step 1: Run baseline training on gpu7 only**

```bash
cd /storage/tap/parameter-golf/upstream
CUDA_VISIBLE_DEVICES=7 python train_gpt.py
```

Note: The baseline is designed for 8xH100 with torchrun. For single-GPU development on B200:
- B200 has ~2x H100 compute, so single-GPU runs will take ~4x longer than 8xH100 (16x ÷ 2x ≈ 8x, but batch parallelism helps)
- For iteration speed, reduce max_steps during development
- Record baseline val_bpb for our hardware as reference point

**Step 2: Record baseline numbers**

Save output to `ours/logs/baseline.log`. Note final val_bpb and training time.

**Step 3: Commit baseline log**

```bash
cd /storage/tap/parameter-golf/ours
mkdir -p logs
cp ../upstream/train*.log logs/baseline.log 2>/dev/null || echo "copy training output manually"
git add logs/ && git commit -m "docs: baseline training log on single B200"
git push
```

---

## Phase 1: Free Wins (No Model Changes)

### Task 1.1: Sliding Window Evaluation

**Impact:** ~-0.032 BPB (biggest single free improvement)

**Files:**
- Modify: `train_gpt.py` — evaluation function

**Step 1: Understand current eval**

Read the evaluation section of `train_gpt.py`. Current eval processes sequences independently with no overlap.

**Step 2: Implement sliding window eval**

Replace the evaluation loop with overlapping windows:
- Window size: 1024 tokens (full context)
- Stride: 64 tokens (score only last 64 tokens per window, but use full 960-token context)
- This means each token gets scored with ~960 tokens of context instead of variable (0 to 1023)

Key implementation:
```python
# In evaluation, instead of processing each 1024-token chunk independently:
# - Slide window by stride=64
# - Only count loss on the last `stride` tokens of each window
# - First window scores all 1024 tokens normally
```

**Step 3: Run eval-only with sliding window**

```bash
CUDA_VISIBLE_DEVICES=7 python train_gpt.py --eval_only
```

Verify: val_bpb should drop by ~0.03 from baseline. Eval time should be ~60-70s (well under 10 min).

**Step 4: Commit**

```bash
git add train_gpt.py && git commit -m "feat: sliding window evaluation (stride=64)"
git push
```

---

### Task 1.2: Document-Isolated Evaluation

**Impact:** ~-0.011 BPB

**Files:**
- Modify: `train_gpt.py` — evaluation function

**Step 1: Implement document boundary isolation**

Don't let context leak across document boundaries in the validation set. When a new document starts, reset the sliding window.

**Step 2: Verify improvement**

Run eval-only, compare to Task 1.1 result.

**Step 3: Commit**

```bash
git add train_gpt.py && git commit -m "feat: document-isolated evaluation"
git push
```

---

## Phase 2: Architecture Improvements

### Task 2.1: Deeper Model (10 Layers)

**Impact:** ~0.01-0.02 BPB per extra layer

**Files:**
- Modify: `train_gpt.py` — model config

**Step 1: Increase depth to 10 layers**

Change `n_layer` from 9 to 10. This adds parameters but we'll recover the byte budget via better quantization in Phase 3.

Calculate parameter budget:
- Baseline 9L: ~X params → Y bytes at int8+zlib
- 10L: ~X+ΔX params → need to verify fits in 16MB with planned int6+zstd

**Step 2: Adjust U-Net skip connections**

With 10 layers, the U-Net skip pattern needs updating (5+5 instead of 4+5 or similar).

**Step 3: Train and evaluate**

```bash
CUDA_VISIBLE_DEVICES=7 python train_gpt.py
```

Record val_bpb improvement.

**Step 4: Commit**

```bash
git add train_gpt.py && git commit -m "feat: 10-layer model with updated skip connections"
git push
```

---

### Task 2.2: 3x MLP Expansion

**Impact:** ~-0.02+ BPB (single largest architecture change)

**Files:**
- Modify: `train_gpt.py` — MLP hidden dim

**Step 1: Change MLP hidden dim from 1024 (2x) to 1536 (3x)**

This is the single most impactful architecture change across all leaderboard entries. MLP capacity is the bottleneck for a small model.

**Step 2: Verify parameter count still fits budget**

With int6 quantization + zstd (Phase 3), 10L + MLP3x should fit in 16MB.

**Step 3: Train and evaluate**

```bash
CUDA_VISIBLE_DEVICES=7 python train_gpt.py
```

**Step 4: Commit**

```bash
git add train_gpt.py && git commit -m "feat: 3x MLP expansion (hidden=1536)"
git push
```

---

### Task 2.3: SmearGate

**Impact:** ~-0.003 BPB for only ~512 extra params

**Files:**
- Modify: `train_gpt.py` — add SmearGate module

**Step 1: Implement SmearGate**

A learnable per-dimension gate that blends each token's embedding with the previous token's embedding:
```python
class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        g = torch.sigmoid(self.gate)
        x_prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        return g * x + (1 - g) * x_prev
```

Apply after embeddings, before first transformer layer.

**Step 2: Train and evaluate**

**Step 3: Commit**

```bash
git add train_gpt.py && git commit -m "feat: SmearGate for bigram-level context"
git push
```

---

### Task 2.4: BigramHash Embedding

**Impact:** ~-0.005 BPB

**Files:**
- Modify: `train_gpt.py` — add BigramHash module

**Step 1: Implement BigramHash**

Hash table that captures token pair patterns:
```python
class BigramHash(nn.Module):
    def __init__(self, num_buckets=4096, dim=128):
        super().__init__()
        self.embed = nn.Embedding(num_buckets, dim)
        self.num_buckets = num_buckets

    def forward(self, token_ids):
        # Hash consecutive token pairs
        prev_ids = torch.cat([torch.zeros_like(token_ids[:, :1]), token_ids[:, :-1]], dim=1)
        pair_hash = (token_ids * 31 + prev_ids) % self.num_buckets
        return self.embed(pair_hash)
```

Add BigramHash output (projected to model dim) to token embeddings.

**Step 2: Train and evaluate**

**Step 3: Commit**

```bash
git add train_gpt.py && git commit -m "feat: BigramHash embedding for token pair patterns"
git push
```

---

## Phase 3: Quantization & Compression (Critical for Byte Budget)

### Task 3.1: Int6 Quantization-Aware Training (QAT)

**Impact:** Saves ~25% bytes vs int8, enabling more parameters. QAT eliminates the ~0.016 BPB quantization gap.

**Files:**
- Modify: `train_gpt.py` — quantization functions + training loop

**Step 1: Implement int6 quantization/dequantization**

```python
def quantize_int6(weight):
    """Quantize to [-32, 31] range with per-row scaling"""
    scale = weight.abs().amax(dim=-1, keepdim=True) / 31.0
    weight_q = (weight / scale).round().clamp(-32, 31).to(torch.int8)
    return weight_q, scale

def dequantize_int6(weight_q, scale):
    return weight_q.float() * scale
```

**Step 2: Implement STE (Straight-Through Estimator) fake quantization for training**

```python
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        q, scale = quantize_int6(weight)
        return dequantize_int6(q, scale)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Straight-through
```

Apply fake quantization during forward pass for all linear layers starting at ~60% of training.

**Step 3: Implement int5 for MLP weights**

MLP weights are more robust to aggressive quantization:
```python
def quantize_int5(weight):
    scale = weight.abs().amax(dim=-1, keepdim=True) / 15.0
    weight_q = (weight / scale).round().clamp(-16, 15).to(torch.int8)
    return weight_q, scale
```

**Step 4: Train with QAT and evaluate**

Verify near-zero quantization gap (compare pre-quant vs post-quant val_bpb).

**Step 5: Commit**

```bash
git add train_gpt.py && git commit -m "feat: int6/int5 QAT with STE fake quantization"
git push
```

---

### Task 3.2: Zstd-22 Compression

**Impact:** ~5% better compression than zlib-9, saves ~1.5MB

**Files:**
- Modify: `train_gpt.py` — export/compression functions

**Step 1: Replace zlib with zstd level 22**

```python
import zstandard as zstd

def compress_model(state_dict):
    # Quantize all weights
    compressed_data = {}
    for name, param in state_dict.items():
        if is_mlp_weight(name):
            q, scale = quantize_int5(param)
        else:
            q, scale = quantize_int6(param)
        compressed_data[name] = (q, scale)

    # Serialize and compress with zstd-22
    raw_bytes = serialize(compressed_data)
    compressor = zstd.ZstdCompressor(level=22)
    return compressor.compress(raw_bytes)
```

**Step 2: Verify artifact size < 16MB**

```python
total_size = len(compressed_model_bytes) + len(open('train_gpt.py', 'rb').read())
assert total_size < 16_000_000, f"Artifact too large: {total_size}"
```

**Step 3: Verify roundtrip**

Decompress → dequantize → evaluate. Val_bpb should match pre-compression.

**Step 4: Commit**

```bash
git add train_gpt.py && git commit -m "feat: zstd-22 compression replacing zlib-9"
git push
```

---

### Task 3.3: Mixed Precision Export

**Impact:** Better quality for critical layers within same byte budget

**Files:**
- Modify: `train_gpt.py` — export function

**Step 1: Use FP16 for embeddings and key projection in last layer**

These are most sensitive to quantization. Everything else gets int6/int5.

**Step 2: Rebalance parameter budget**

With FP16 embeddings costing more bytes, verify total artifact still fits 16MB. May need to adjust BigramHash bucket count.

**Step 3: Evaluate and commit**

```bash
git add train_gpt.py && git commit -m "feat: mixed precision export (FP16 embeds, int5/6 blocks)"
git push
```

---

## Phase 4: Training Optimization

### Task 4.1: Muon Weight Decay + Orthogonal Init

**Impact:** ~-0.003-0.005 BPB, also makes quantization more effective

**Files:**
- Modify: `train_gpt.py` — optimizer config + initialization

**Step 1: Set Muon weight decay to 0.04**

Decoupled weight decay keeps weight magnitudes small and well-distributed, which directly improves quantization quality.

**Step 2: Orthogonal initialization for all weight matrices**

```python
for name, param in model.named_parameters():
    if param.dim() == 2:
        nn.init.orthogonal_(param, gain=1.0)
```

**Step 3: Train and evaluate**

**Step 4: Commit**

```bash
git add train_gpt.py && git commit -m "feat: Muon WD=0.04 + orthogonal initialization"
git push
```

---

### Task 4.2: SWA (Stochastic Weight Averaging)

**Impact:** ~-0.001-0.003 BPB, smoother weights quantize better

**Files:**
- Modify: `train_gpt.py` — training loop

**Step 1: Implement SWA during warmdown phase**

Average model weights over the last N checkpoints during the learning rate warmdown:
```python
# During warmdown (last 3000 iterations):
swa_model = copy.deepcopy(model)
swa_count = 0

# Every K steps during warmdown:
for name, param in model.named_parameters():
    swa_model.state_dict()[name].mul_(swa_count / (swa_count + 1))
    swa_model.state_dict()[name].add_(param.data / (swa_count + 1))
swa_count += 1
```

**Step 2: Use SWA model for final export and evaluation**

**Step 3: Evaluate and commit**

```bash
git add train_gpt.py && git commit -m "feat: SWA during warmdown for smoother quantization"
git push
```

---

### Task 4.3: EMA (Exponential Moving Average)

**Impact:** ~-0.002 BPB on top of SWA

**Files:**
- Modify: `train_gpt.py` — training loop

**Step 1: Maintain EMA of model weights throughout training**

```python
ema_decay = 0.999
ema_model = copy.deepcopy(model)

# After each optimizer step:
with torch.no_grad():
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
```

**Step 2: Evaluate with EMA model (may be better than SWA model)**

Compare SWA vs EMA vs SWA+EMA hybrid for best result.

**Step 3: Commit**

```bash
git add train_gpt.py && git commit -m "feat: EMA model tracking"
git push
```

---

### Task 4.4: Training Hyperparameter Tuning

**Impact:** Variable, ~-0.005 BPB cumulative

**Files:**
- Modify: `train_gpt.py` — hyperparameters

**Step 1: Apply known-good hyperparameters from leaderboard analysis**

- Sequence length: 2048 (up from 1024)
- Learning rate: matrix_lr=0.02-0.025 (lower for deeper model)
- Muon momentum: 0.99 with warmup from 0.85 over 1500 steps
- Warmdown: 3000 iterations
- Gradient clipping: 0.3

**Step 2: Run ablation on each change (time permitting)**

**Step 3: Commit best configuration**

```bash
git add train_gpt.py && git commit -m "feat: tuned hyperparameters (seq2048, lr0.025, warmdown3000)"
git push
```

---

## Phase 5: Advanced Techniques (Frontier Push)

### Task 5.1: Test-Time Training with LoRA

**Impact:** ~-0.003 BPB

**Files:**
- Modify: `train_gpt.py` — evaluation function

**Step 1: Implement LoRA TTT at eval time**

During evaluation, for each document:
1. Add small LoRA adapters (rank 2-4) to attention layers
2. Fine-tune on the document prefix using standard LM loss
3. Score remaining tokens with adapted model
4. Reset LoRA weights for next document

Must fit within the 10-minute eval budget.

**Step 2: Tune LoRA rank and learning rate**

**Step 3: Commit**

```bash
git add train_gpt.py && git commit -m "feat: LoRA test-time training at evaluation"
git push
```

---

### Task 5.2: Explore 11-12 Layer Models

**Impact:** Depends on byte budget remaining

**Files:**
- Modify: `train_gpt.py` — model config

**Step 1: With int5 MLP + int6 attention + zstd-22, calculate max layers that fit 16MB**

**Step 2: Try 11L and 12L, compare val_bpb**

More layers with aggressive quantization is the proven formula for leaderboard success.

**Step 3: Commit best configuration**

```bash
git add train_gpt.py && git commit -m "feat: optimal layer count for byte budget"
git push
```

---

### Task 5.3: Partial RoPE + LN Scale

**Impact:** Emerging technique in frontier PRs (~-0.002 BPB)

**Files:**
- Modify: `train_gpt.py` — attention implementation

**Step 1: Apply RoPE to only a fraction of attention heads**

Some heads benefit from absolute positioning. Partial RoPE lets the model learn which heads need rotary and which don't.

**Step 2: Add learnable LayerNorm scaling factors**

Small per-layer scale parameters that the model can tune.

**Step 3: Evaluate and commit**

```bash
git add train_gpt.py && git commit -m "feat: partial RoPE + learnable LN scale"
git push
```

---

## Phase 6: Submission

### Task 6.1: Multi-GPU Validation

**Files:**
- None (validation)

**Step 1: Test with torchrun on b200 (if multiple GPUs available temporarily)**

```bash
# If we can borrow GPUs briefly:
torchrun --nproc_per_node=8 train_gpt.py
```

**Step 2: Alternatively, use RunPod for 8xH100 validation**

The challenge provides compute credits. Spin up 8xH100 instance, run training + eval, record results.

**Step 3: Verify artifact size**

```bash
python -c "
import os
code_size = os.path.getsize('train_gpt.py')
model_size = os.path.getsize('model.zst')  # or whatever the compressed model file is
total = code_size + model_size
print(f'Code: {code_size:,} bytes')
print(f'Model: {model_size:,} bytes')
print(f'Total: {total:,} bytes')
print(f'Budget: 16,000,000 bytes')
print(f'Remaining: {16_000_000 - total:,} bytes')
assert total < 16_000_000
"
```

---

### Task 6.2: Prepare Submission PR

**Files:**
- Create: `records/track_10min_16mb/YYYY-MM-DD_OurSubmissionName/README.md`
- Create: `records/track_10min_16mb/YYYY-MM-DD_OurSubmissionName/submission.json`
- Create: `records/track_10min_16mb/YYYY-MM-DD_OurSubmissionName/train_gpt.py`
- Create: `records/track_10min_16mb/YYYY-MM-DD_OurSubmissionName/train.log`

**Step 1: Format submission per challenge requirements**

`submission.json`:
```json
{
  "val_bpb": <our_score>,
  "artifact_size": <total_bytes>,
  "training_time_seconds": <time>,
  "gpu_config": "8xH100_SXM",
  "techniques": ["list", "of", "techniques"]
}
```

**Step 2: Open PR on upstream repo**

```bash
cd /storage/tap/parameter-golf/upstream
git checkout -b tap-mobile-submission
# Copy our files into records/
git add records/
git commit -m "submission: Tap Mobile - <val_bpb> BPB"
gh pr create --repo openai/parameter-golf
```

---

## Development Strategy: Single-GPU Iteration

Since we're developing on a single B200:

1. **Quick iteration mode:** Set `max_steps=2000` for ~3-5 min runs during development
2. **Full training:** ~20-30 min on single B200 (equivalent to ~10 min on 8xH100)
3. **Track BPB improvements in a spreadsheet/log after each change**
4. **Stack optimizations incrementally** — never apply multiple untested changes at once
5. **Git tag each milestone** for easy comparison and rollback

## Expected BPB Progression

| Phase | Cumulative Techniques | Expected BPB | Delta |
|-------|----------------------|--------------|-------|
| Baseline | 9L, int8, zlib | ~1.224 | — |
| Phase 1 | + sliding window + doc isolation | ~1.181 | -0.043 |
| Phase 2 | + 10L, MLP3x, SmearGate, BigramHash | ~1.153 | -0.028 |
| Phase 3 | + int6/5 QAT, zstd-22, mixed precision | ~1.140 | -0.013 |
| Phase 4 | + WD, ortho init, SWA, EMA, tuned HP | ~1.130 | -0.010 |
| Phase 5 | + TTT, 11-12L, partial RoPE | ~1.120 | -0.010 |

**Target:** Sub-1.13 BPB would be competitive with current frontier.

## Risk Register

| Risk | Mitigation |
|------|-----------|
| Single B200 ≠ 8xH100 (different batch dynamics) | Validate on RunPod before submission |
| Artifact exceeds 16MB | Track byte budget continuously, have int5 fallback |
| Training exceeds 10 min on H100 | Profile early, have reduced-step fallback |
| GPU 7 needed for production | Coordinate with team, run during off-hours |
| Challenge ends April 30 | Aim for first submission by April 7, iterate after |
