# Phase 3: Performance & Remaining Techniques

## 1. Data Loader Bottleneck (PRIORITY)

**Problem:** GPU utilization drops below 98% because:
- `load_data_shard()` reads from disk synchronously when switching shards
- `.to(device)` transfer blocks GPU — no async/non_blocking pipeline
- No prefetching — GPU waits while CPU prepares next batch
- `torch.cat()` on every multi-chunk take() adds CPU overhead

**Fix: Prefetch + Pinned Memory**
```python
class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        # Pre-load ALL shards into one contiguous CPU tensor (they fit in RAM)
        all_tokens = torch.cat([load_data_shard(f) for f in sorted(glob.glob(pattern))])
        self.tokens = all_tokens.pin_memory()  # pinned for fast H2D transfer
        self.pos = 0
        self.device = device
        self.rank = rank
        self.world_size = world_size
        # Prefetch buffer
        self._next_batch = None

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        if self._next_batch is not None:
            result = self._next_batch
            self._next_batch = None
            self._prefetch(global_tokens, seq_len, grad_accum_steps)
            return result
        return self._prepare_batch(global_tokens, seq_len, grad_accum_steps)

    def _prepare_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        span = local_tokens + 1
        # Wrap around if needed
        end = self.pos + span * self.world_size
        if end > self.tokens.numel():
            self.pos = 0
            end = span * self.world_size
        chunk = self.tokens[self.pos : end]
        self.pos = end
        start = self.rank * span
        local = chunk[start : start + span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
```

**Impact:** Eliminates per-shard I/O stalls. With 8GB data and 100GB+ RAM, all shards fit in memory.

## 2. Missing nanochat Techniques

### 2a. Polar Express (replaces Newton-Schulz in Muon)
- Better orthogonalization convergence
- Drop-in replacement, same number of iterations
- Source: nanochat, arxiv:2505.16932

### 2b. Cosine Weight Decay Schedule
- `wd * 0.5 * (1 + cos(pi * step / total_steps))`
- Full WD at start, zero at end
- Prevents WD from fighting late convergence

### 2c. Muon Momentum Warmdown
- During LR warmdown: decay momentum 0.99 -> 0.90
- nanochat does 0.97 -> 0.90

### 2d. Non-uniform Per-layer Init
- `resid_mix[0][i] = 1.15 - 0.10 * i/(n_layer-1)` (early layers stronger)
- `resid_mix[1][i] = 0.20 - 0.15 * i/(n_layer-1)` (early layers more x0)

### 2e. Logit Softcap 15 (currently 30)
- Karpathy tuned to 15, we use 30

## 3. Missing Competition Techniques

### 3a. Smear replacing BigramHash
- Saves ~1.3M params / ~1MB artifact
- Could enable 11 layers again
- nanochat proved Smear works as well at scale

### 3b. Value Residual (PR #413)
- Cache V from layer 0, mix into all subsequent layers
- -0.015 BPB in ablation
- 18 learnable params

### 3c. Dynamic Evaluation (PR #397)
- SGD gradient steps during sliding window scoring
- -0.024 BPB improvement
- Complementary to TTT
- Zero artifact cost

### 3d. Aggressive TTT (PR #398 approach)
- 20 epochs instead of 3
- All blocks unfrozen (freeze=0)
- lr=0.008 instead of 0.002
- This is what got 1.1213 BPB

## 4. FP8 + QAT for Larger Model (Future)

**Idea:** Train in FP8 with quantization-aware training targeting INT4 export.
- FP8 training saves memory -> can fit 13-14 layers
- QAT throughout training makes model robust to INT4
- INT4 + zstd -> fits more capacity in 16MB
- nanochat has a minimal 150-line FP8 implementation

**Risk:** High complexity, needs careful testing. PR #375 showed naive INT4 adds +0.06 BPB. But proper GPTQ/AWQ with QAT could be much better.

## Priority Order

1. **Fix data loader** (prefetch all to RAM + pin memory) — immediate throughput gain
2. **Aggressive TTT** (TTT_EPOCHS=10, TTT_FREEZE_BLOCKS=0, TTT_LR=0.005) — eval-time, no retrain
3. **Smear + 11 layers** — needs retrain but biggest architecture win
4. **Polar Express + Cosine WD + Momentum warmdown** — optimizer upgrades, needs retrain
5. **Value Residual + Dynamic Eval** — needs retrain + eval changes
6. **FP8 + INT4 QAT** — future phase, high risk/reward
