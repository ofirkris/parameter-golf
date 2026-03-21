# Parameter Golf — Optimal Combo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Achieve sub-1.12 BPB by combining the best proven techniques from all top submissions and frontier PRs.

**Architecture:** 11-layer GPT with 3x MLP, XSA on last 4 layers, EMA, Partial RoPE, LN Scale, SmearGate, BigramHash(10240), Int5 MLP / Int6 attention QAT, zstd-22, and test-time training at eval.

**Tech Stack:** PyTorch, FlashAttention, SentencePiece, CUDA, torchrun (DDP), zstandard, Muon optimizer

**Target:** <1.12 BPB (would be SOTA if achieved)

---

## Technique Combo — Why These Specific Choices

Each technique is selected based on measured deltas from actual submissions:

| Technique | Source | Measured Impact | Rationale |
|-----------|--------|----------------|-----------|
| 11 layers | PR #315 | ~-0.015 vs 9L | More depth = better; 11L fits in 16MB with int5/6+zstd |
| 3x MLP (1536) | All top entries | ~-0.020 vs 2x | Single largest architecture win |
| XSA last 4 layers | PR #315 | ~-0.005 | Zero params, removes self-attention bias, forces cross-token info |
| EMA (0.997) | PR #315 | ~-0.002 vs SWA | Smoother than SWA, better quant compatibility |
| Partial RoPE (16/64 dims) | PR #315 | ~-0.001 | Position-free dims improve generalization |
| LN Scale (1/√(i+1)) | PR #315 | ~-0.001 | Stabilizes deep models, zero params |
| Late QAT (last 4%) | PR #315 | ~-0.001 | Cuts quant gap 3x vs post-training only |
| SmearGate | thwu1/raahil | ~-0.003 | 512 params, bigram-level context |
| BigramHash(10240, 128) | thwu1 | ~-0.005 | Token pair patterns, thwu1 showed 10K > 4K buckets |
| Int5 MLP + Int6 attn | thwu1 | saves ~1.8MB | Funds 11th layer; MLP tolerates int5 |
| Magnitude pruning 3% | thwu1 | saves ~0.3MB | Improves zstd compression |
| zstd-22 | All top entries | saves ~1.5MB vs zlib | Best compression for quantized weights |
| FP16 embeddings | All top entries | ~-0.002 | Embeddings most sensitive to quantization |
| Orthogonal init | thwu1/raahil | ~-0.002 | Faster convergence |
| Muon WD=0.04 | All top entries | ~-0.002 | Tighter weights → better quantization |
| Sliding window (stride=64) | All entries | ~-0.032 | Free eval improvement |
| TTT (3 epochs SGD) | PR #338 | ~-0.002 | Uses eval time budget to recover quant loss |

**Expected cumulative:** ~1.115–1.120 BPB

---

## Architecture Spec

```
Model:
  layers: 11
  model_dim: 512
  num_heads: 8 (query)
  kv_heads: 4 (GQA)
  head_dim: 64
  mlp_hidden: 1536 (3x)
  mlp_activation: relu²
  vocab: 1024 (SentencePiece BPE)
  seq_len: 2048 (train + eval)
  tied_embeddings: yes

  Extras:
    smeargate: dim=512, init gate=3.0 (sigmoid → ~0.95 pass-through)
    bigram_hash: buckets=10240, hash_dim=128, proj→512
    xsa: last 4 layers, GQA-aware efficient impl
    partial_rope: 16 of 64 dims get RoPE
    ln_scale: 1/sqrt(layer_idx + 1)
    u_net_skips: encoder=5, decoder=6
    logit_softcap: 30.0

Optimizer:
  matrix: Muon, lr=0.025, momentum=0.99 (warmup 0.92→0.99/1500 steps), WD=0.04
  embedding: AdamW, lr=0.035, WD=0.04
  scalar: AdamW, lr=0.025, WD=0.04
  grad_clip: 0.3
  batch_tokens: 786432

Schedule:
  warmup: 20 steps (compile warmup)
  warmdown: 3000 iters
  max_wallclock: 600s
  late_qat: enable when lr_scale < 0.1 (~last 4%)

Post-training:
  ema_decay: 0.997 (maintained throughout training, used for export)
  quantization: int5 MLP, int6 attention, int8 embeddings, fp16 last-layer c_k, fp32 scalars
  pruning: 3% magnitude on quantized weights
  compression: zstd level 22

Evaluation:
  sliding_window: stride=64, seq_len=2048
  ttt: 3 epochs SGD, lr=0.002, momentum=0.9, freeze first 2 blocks
  document_isolated: yes
```

---

## Task 0: Environment Setup on b200

**Step 1: SSH in and create workspace**

```bash
ssh b200
mkdir -p /storage/tap/parameter-golf && cd /storage/tap/parameter-golf
git clone https://github.com/openai/parameter-golf.git upstream
git clone git@github.com:Tap-Mobile/parameter-golf.git ours
```

**Step 2: Set up Python env**

```bash
cd /storage/tap/parameter-golf/upstream
pip install -r requirements.txt
pip install zstandard
```

**Step 3: Download data**

```bash
python data/cached_challenge_fineweb.py
ls data/fineweb_train_*.bin | wc -l  # expect 80
ls data/fineweb_val_*.bin | wc -l    # expect val shards
```

**Step 4: Verify single-GPU baseline runs**

```bash
CUDA_VISIBLE_DEVICES=7 python train_gpt.py 2>&1 | head -50
```

Just verify it starts, then Ctrl+C. We don't need a full baseline run.

**Step 5: Commit setup confirmation**

---

## Task 1: Build the Optimal train_gpt.py

This is the core task. We build a single train_gpt.py that incorporates ALL techniques from the combo.

**Starting point:** PR #315's train_gpt.py (best open submission at 1.1248 BPB). We enhance it with thwu1's innovations.

**Files:**
- Create: `ours/train_gpt.py` — the combo script

### Step 1: Start from PR #315 base

Fetch PR #315's train_gpt.py as our starting point. It already has:
- 11 layers, 512d, 3x MLP, GQA
- XSA on last 4 layers
- EMA (0.997)
- Partial RoPE (16 dims)
- LN Scale
- Late QAT (int6)
- SmearGate + BigramHash
- Orthogonal init
- Muon WD=0.04
- Sliding window eval
- zstd-22

### Step 2: Add Int5 MLP quantization (from thwu1)

Replace uniform int6 with mixed quantization:
- MLP weights (c_fc, c_proj): int5 [-16, 15]
- Attention weights (c_q, c_k, c_v, c_o): int6 [-32, 31]
- BigramHash: int6
- Embeddings: FP16 passthrough
- Last-layer c_k: FP16 passthrough
- Scalars: FP32 passthrough

Key change in `_classify_param`:
```python
def _classify_param(name):
    if "c_fc" in name or "c_proj" in name:
        return "mlp"  # → int5
    elif any(k in name for k in ["c_q", "c_k", "c_v", "c_o"]):
        return "attn"  # → int6
    elif "bigram" in name and "table" in name:
        return "bigram"  # → int6
    else:
        return "other"
```

In quantization:
```python
clip = 15 if cat == "mlp" else 31  # int5 for MLP, int6 for rest
```

### Step 3: Add magnitude pruning (from thwu1)

Before quantization, zero out the smallest 3% of weights:
```python
def magnitude_prune(tensor, fraction=0.03):
    threshold = torch.quantile(tensor.abs(), fraction)
    tensor[tensor.abs() < threshold] = 0
    return tensor
```

### Step 4: Increase BigramHash buckets to 10240 (from thwu1)

thwu1 showed 10240 > 4096 buckets. Change:
```python
BIGRAM_HASH_BUCKETS = 10240
```

### Step 5: Update Late QAT to handle int5

The STE fake-quantize during training should match export precision:
```python
def fake_quantize(w, name):
    cat = _classify_param(name)
    clip = 15 if cat == "mlp" else 31
    scale = w.abs().amax(dim=-1, keepdim=True) / clip
    w_q = (w / scale).round().clamp(-clip, clip) * scale
    return w + (w_q - w).detach()  # STE
```

### Step 6: Add TTT (from PR #338)

After training + quantization + decompression, before final eval:
```python
def ttt_adapt(model, val_tokens, num_epochs=3, lr=0.002, momentum=0.9, freeze_blocks=2):
    # Freeze first 2 blocks
    for i in range(freeze_blocks):
        for p in model.blocks[i].parameters():
            p.requires_grad_(False)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=momentum
    )

    for epoch in range(num_epochs):
        # Standard LM loss on validation tokens
        for batch in val_batches:
            loss = model(batch_x, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    # Unfreeze all
    for p in model.parameters():
        p.requires_grad_(True)
```

### Step 7: Verify artifact fits 16MB

```python
total = code_bytes + compressed_model_bytes
assert total < 16_000_000
print(f"Artifact: {total:,} / 16,000,000 bytes ({total/16_000_000*100:.1f}%)")
```

### Step 8: Commit

```bash
cd ours && git add train_gpt.py && git commit -m "feat: optimal combo train_gpt.py"
git push
```

---

## Task 2: Single-GPU Development Run

**Step 1: Quick validation run (reduced steps)**

```bash
cd /storage/tap/parameter-golf/upstream
CUDA_VISIBLE_DEVICES=7 MAX_STEPS=2000 python ../ours/train_gpt.py
```

Verify: model trains without errors, loss decreases, eval runs, artifact exports.

**Step 2: Check artifact size**

Verify compressed model + code < 16MB.

**Step 3: Full single-GPU training run**

```bash
CUDA_VISIBLE_DEVICES=7 python ../ours/train_gpt.py 2>&1 | tee ../ours/logs/combo_v1.log
```

Expected: ~25-35 min on single B200 (equivalent to ~10 min on 8xH100).

**Step 4: Record results**

Save val_bpb, training time, artifact size to `ours/logs/results.md`.

**Step 5: Commit**

```bash
cd ours && git add logs/ && git commit -m "results: combo v1 single-GPU run"
git push
```

---

## Task 3: Ablation & Tuning

Run ablations to verify each technique contributes positively. Disable one at a time:

| Ablation | Expected impact if removed |
|----------|---------------------------|
| XSA off | +0.005 BPB |
| EMA off (use SWA) | +0.002 BPB |
| Partial RoPE off (full RoPE) | +0.001 BPB |
| LN Scale off | +0.001 BPB |
| Int5→Int6 MLP | artifact grows, lose capacity |
| BigramHash 10240→4096 | +0.001 BPB |
| TTT off | +0.002 BPB |
| Late QAT off | +0.001 BPB |

If any technique hurts: remove it and recover the BPB.

Tuning passes:
1. Learning rate sweep: 0.020, 0.025, 0.030
2. EMA decay sweep: 0.995, 0.997, 0.999
3. BigramHash buckets: 8192, 10240, 12288
4. XSA layers: last 3 vs last 4 vs last 5
5. Warmdown iters: 2500, 3000, 3500

---

## Task 4: Multi-GPU Validation & Submission

**Step 1: Validate on 8 GPUs (RunPod or borrow b200 GPUs briefly)**

```bash
torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee train.log
```

Must complete in <600s and produce artifact <16MB.

**Step 2: Run 3 seeds for statistical significance**

```bash
for seed in 42 1337 2024; do
    SEED=$seed torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed${seed}.log
done
```

Need p < 0.01 and beat SOTA by ≥0.005 nats.

**Step 3: Prepare submission**

```bash
mkdir -p records/track_10min_16mb/2026-03-XX_TapMobile_OptimalCombo/
# Copy: train_gpt.py, README.md, submission.json, train.log
```

**Step 4: Open PR on openai/parameter-golf**

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Artifact > 16MB | Int5 all weights as fallback; reduce BigramHash buckets; drop to 10 layers |
| Training > 10 min on H100 | Reduce batch to 524K (more steps but smaller); reduce warmdown |
| Techniques don't stack | Ablation in Task 3 catches negative interactions early |
| Single-GPU perf differs from 8-GPU | Validate on RunPod before submission |
| GPU 7 occupied by boost-test | boost-test uses ~71GB but 0% compute; our training fits in remaining 112GB |
| XSA paper too new (Mar 2026) | PR #315 already validates it works; we just adapt |

---

## Byte Budget Estimate

Based on thwu1 (10L, int5 MLP, 15.9MB) and PR #315 (11L, int6 all, 15.6MB):

| Component | Params | Bits | Raw Bytes | zstd Est |
|-----------|--------|------|-----------|----------|
| 11 transformer blocks (MLP int5) | ~18M MLP | 5 | 11.25 MB | ~6.0 MB |
| 11 transformer blocks (attn int6) | ~4M attn | 6 | 3.0 MB | ~2.0 MB |
| BigramHash(10240, 128) int6 | ~1.3M | 6 | 0.98 MB | ~0.65 MB |
| Tied embeddings FP16 | 0.5M | 16 | 1.0 MB | ~0.8 MB |
| Scales, gates, scalars FP32 | ~10K | 32 | 0.04 MB | ~0.03 MB |
| BigramHash proj int6 | 65K | 6 | 0.049 MB | ~0.03 MB |
| Skip weights, SmearGate FP32 | ~6K | 32 | 0.024 MB | ~0.02 MB |
| **Compressed model total** | | | | **~9.5 MB** |
| Code (train_gpt.py) | | | ~55 KB | **~55 KB** |
| Serialization overhead | | | ~50 KB | **~50 KB** |
| **Total estimate** | | | | **~9.6 MB** |
| **16MB budget remaining** | | | | **~6.4 MB** |

We have significant headroom. This means we could potentially:
- Try 12 layers (PR #332 approach)
- Increase BigramHash to 16384 buckets
- Use int6 for MLP instead of int5 (if quality matters more)
