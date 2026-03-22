# Parameter Golf — Iteration Plan

> **For Claude on mac-mini:** Follow this plan step by step. Each task is one change → test → record. Never apply multiple untested changes at once.

## Connection
```bash
ssh -i ~/.ssh/smedia4 root@86.38.238.153   # Verda H100 (IP may change on new spot instance)
```
If instance is dead, wait for Ofir to spin up a new one and update the IP.

## Setup (run on every new spot instance)
```bash
# Clone our repo (private — needs SSH key or PAT)
git clone git@github.com:Tap-Mobile/parameter-golf.git /mnt/vol/ours 2>/dev/null || (cd /mnt/vol/ours && git pull)

# Or if data already on volume, just install deps:
apt-get update -qq && apt-get install -y -qq python3-pip git 2>&1 | tail -3
pip3 install --break-system-packages --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cu130 2>&1 | tail -3 || \
pip3 install --break-system-packages --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cu128 2>&1 | tail -3
pip3 install --break-system-packages zstandard sentencepiece huggingface-hub datasets numpy tqdm 2>&1 | tail -3
pip3 install --break-system-packages --upgrade flash-attn --no-build-isolation 2>&1 | tail -5 || echo "FA failed, using SDPA"
```

## How to Run a Test
```bash
# 1. Copy data to RAM
mkdir -p /dev/shm/data/datasets /dev/shm/data/tokenizers
cp -r /mnt/vol/upstream/data/datasets/fineweb10B_sp1024 /dev/shm/data/datasets/
cp /mnt/vol/upstream/data/tokenizers/fineweb_1024_bpe.model /dev/shm/data/tokenizers/

# 2. Run quick test (500 steps, 5 min cap)
cd /mnt/vol/ours
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=/dev/shm/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/dev/shm/data/tokenizers/fineweb_1024_bpe.model \
TORCH_COMPILE=0 \
MAX_WALLCLOCK_SECONDS=300 \
ITERATIONS=500 \
VAL_LOSS_EVERY=100 \
TRAIN_LOG_EVERY=25 \
EVAL_STRIDE=64 \
python3 train_gpt.py 2>&1 | tee logs/run_NAME.log
```

## How to Record Results
After each run, append to `/mnt/vol/ours/logs/results.md`:
```
## Run N — NAME
- Changes: what you changed
- Steps: X/500
- val_bpb: X.XXXX
- Artifact: X bytes (OVER/UNDER 16MB)
- Verdict: KEEP / REVERT
```

## Rules
- **One change per run.** Test it. If worse, revert.
- **Save versions:** `cp train_gpt.py train_gpt_vN.py` before each edit
- **All files on /mnt/vol** — spot instance can die anytime
- **Artifact MUST be < 16,000,000 bytes** (code + compressed model)
- **git push** after each successful improvement
- **val_bpb** at 500 steps is relative — use it to compare runs, not as absolute score
- **Check artifact size** — this is the binding constraint right now

## Current Status (after v5 autonomous runs)
- **10 layers** (11 was 17.2MB, over limit)
- val_bpb ~3.2 at 130 steps (too few steps to be meaningful, just sanity check)
- Artifact: ~17MB with current config (STILL OVER — needs fixing first)
- Value Residual + Gated Attention added
- GPTQ-lite was broken, reverted

## Critical Problem: Artifact Size
The artifact is OVER 16MB even at 10 layers. Before adding features, **fix the byte budget first.**

---

## Phase 1: Fix Artifact Size (MUST DO FIRST)

### Task 1.1: Replace BigramHash with Smear
**Why:** BigramHash(10240, dim=128) uses ~1.3M params + a projection layer. Karpathy tested both in nanochat — BigramHash was **reverted at scale**, Smear survived. Smear provides similar bigram info with ~500 params total.

**Implementation:**
Replace the `BigramHashEmbedding` class with a simple Smear:
```python
class Smear(nn.Module):
    def __init__(self, dim, gate_channels=24):
        super().__init__()
        self.gate_proj = nn.Linear(gate_channels, 1, bias=True)
        self.lam = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 3.0)  # sigmoid(3) ≈ 0.95 pass-through
        self.gate_channels = gate_channels

    def forward(self, x):
        gate = self.lam * torch.sigmoid(self.gate_proj(x[..., :self.gate_channels]))
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return x + gate * x_prev
```

Remove BigramHash from GPT.__init__, forward, forward_logits, optimizer setup, and quantization categories.

**Expected impact:** Free ~1-1.5MB in artifact. May enable 11 layers again.

**Test:** Run 500 steps, compare val_bpb and artifact size vs baseline.

### Task 1.2: Try 11 Layers Again (after Smear saves bytes)
**Why:** If Smear saves enough bytes, 11L should fit under 16MB.

**Test:** Set NUM_LAYERS=11, run, check artifact size.

### Task 1.3: Tune Pruning Percentage
**Why:** More pruning = smaller artifact but worse quality. Find the sweet spot.

**Test:** Try 5%, 8%, 10%, 15% magnitude pruning. Record artifact size and val_bpb for each.

### Task 1.4: Verify Artifact < 16MB
**Goal:** Get artifact confidently under 16,000,000 bytes with room to spare.

---

## Phase 2: Optimizer Upgrades (from nanochat)

### Task 2.1: Polar Express (replace Newton-Schulz in Muon)
**Why:** Better orthogonalization convergence, same cost. Drop-in replacement.

**Implementation:** Replace `zeropower_via_newtonschulz5` with Polar Express using precomputed coefficients from arxiv:2505.16932:
```python
def polar_express(G, steps=5, eps=1e-7):
    # Precomputed minimax-optimal coefficients
    coeffs = [(1.7321, 0.7321), (1.9319, 0.9319), (1.9829, 0.9829),
              (1.9957, 0.9957), (1.9989, 0.9989)]
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for a, b in coeffs[:steps]:
        X = a * X - b * (X @ X.T) @ X
    return X.T if transposed else X
```
Note: verify exact coefficients from the paper or nanochat source code.

**Test:** Run 500 steps, compare val_bpb vs Newton-Schulz.

### Task 2.2: Cautious Weight Decay
**Why:** Only apply WD where gradient and parameter agree in sign. Smarter than uniform WD.

**Implementation:** In Muon.step(), change:
```python
# Before:
p.data.mul_(1.0 - lr * wd)
# After:
if wd > 0.0:
    mask = (p.grad * p.data) >= 0
    p.data[mask] *= (1.0 - lr * wd)
```

**Test:** Run 500 steps, compare val_bpb.

### Task 2.3: Cosine Weight Decay Schedule
**Why:** Decay WD to zero by end of training. Prevents WD from fighting late convergence.

**Implementation:** Scale WD by `0.5 * (1 + cos(pi * step / total_steps))`.

**Test:** Run 500 steps, compare.

---

## Phase 3: Architecture Improvements

### Task 3.1: Backout (subtract mid-layer residual before logit head)
**Why:** Removes low-level features that hurt final prediction. 1 scalar param, nearly free.

**Implementation:**
```python
# In GPT.__init__:
self.backout_lambda = nn.Parameter(torch.tensor(0.2))

# In forward/forward_logits, after encoder-decoder loop but before final_norm:
# Cache x at mid-layer:  x_mid = x  (at layer num_layers//2)
# Before final norm:     x = x - self.backout_lambda * x_mid
```

**Test:** Run 500 steps, compare val_bpb.

### Task 3.2: Non-uniform Per-layer Init
**Why:** Earlier layers should get stronger residual scaling, later layers less.

**Implementation:**
```python
# In GPT.__init__, after creating blocks:
with torch.no_grad():
    for i, block in enumerate(self.blocks):
        frac = i / max(len(self.blocks) - 1, 1)
        # resid_mix[0] (residual weight): 1.15 -> 1.05
        block.resid_mix.data[0].fill_(1.15 - 0.10 * frac)
        # resid_mix[1] (x0 blend): 0.20 -> 0.05
        block.resid_mix.data[1].fill_(0.20 - 0.15 * frac)
```

**Test:** Run 500 steps, compare.

### Task 3.3: Logit Softcap 15 (from 30)
**Why:** Karpathy tuned to 15. We use 30. Easy test.

**Test:** Set LOGIT_SOFTCAP=15, run 500 steps, compare.

### Task 3.4: Muon Momentum Warmdown
**Why:** nanochat warms down momentum 0.97→0.90 during LR warmdown. We only warm up.

**Implementation:** During the LR warmdown phase, linearly decay Muon momentum from 0.99 to 0.90.

**Test:** Run 500 steps, compare.

---

## Phase 4: Validation & Submission

### Task 4.1: Full 1xH100 Run (600s)
After all improvements are stacked, run a full 600s single-GPU training.

### Task 4.2: Get 8xH100 Access
Need 8xH100 SXM for official submission timing. Options:
- Verda 8xH100 instance
- RunPod 8xH100 (max 6 available — may need cluster)
- OpenAI compute credits via the challenge

### Task 4.3: 3-Seed Validation
Run with seeds 42, 1337, 2024. Need p < 0.01 and ≥0.005 nat improvement.

### Task 4.4: Submit PR to openai/parameter-golf

---

## Changelog
Track all changes here as they're made.
