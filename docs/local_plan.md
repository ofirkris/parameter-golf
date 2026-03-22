# Parameter Golf Local Optimization Plan (Updated 2026-03-22)

## Competitive Landscape
| Submission | BPB | Key Techniques | TTT? |
|-----------|------|---------------|------|
| PR #398 | 1.1213 | 11L, EMA, aggressive TTT(20ep,freeze=0), no QAT | YES |
| PR #417 | 1.1227 | 11L, XSA4, SWA, Two-Phase TTT | YES |
| PR #401 | 1.1243 | 11L, EMA+SWA, QAT@0.15, VE128, no TTT | NO |
| Merged Record | 1.1428 | 10L, SWA, int5 MLP/int6 attn | NO |

## Our Advantages
1. **TTT** (two-phase + aggressive unfreeze option)
2. **Value Residual** (layer 0 V mixing across all layers)
3. **Gated Attention** (per-head sigmoid gate)
4. **QAT fix** (v7+: int5 clip for MLP, matching export)
5. **LeakyReLU²** (v8: from PR #434)

## Prepared Versions

### v6: Global Pruning Infrastructure
- Configurable PRUNE_FRACTION (default 3%), PRUNE_GLOBAL (default 1)
- Global pruning allocates sparsity smarter across layers
- Minimal risk, good infrastructure

### v7: v6 + QAT Bug Fix (HIGH PRIORITY)
- CastedLinear gets per-instance `qat_clip_range`
- MLP layers: int5 QAT (clip=15), Attn: int6 QAT (clip=31)
- Fixes mismatch where QAT simulated int6 but export uses int5 for MLP
- Expected: 0.001-0.01 BPB improvement in quant gap

### v8: v7 + Competition-Optimized Settings (BEST SHOT)
- **LeakyReLU(0.5)²** instead of ReLU² (PR #434)
- **Late QAT DISABLED** (counterproductive with aggressive TTT, PR #398)
- **warmdown=3000** (matching PR #398, was 3500)
- **TTT Phase 2**: All blocks unfrozen (freeze=0), lr=0.008, 20 epochs (PR #398)
- **TTT Phase 1**: Unchanged (50 epochs norm-only, lr=0.01)
- **EMA only** (no SWA, matching PR #398's winning approach)
- Keeps VR+GA (our unique advantages)

## Testing Strategy
1. **Quick test**: Deploy v8 directly (has all improvements, best chance to win)
2. **If artifact over 16MB**: Increase PRUNE_FRACTION to 0.05-0.08
3. **If val_bpb worse than expected**: Test components individually via env vars:
   - `LATE_QAT=1` to re-enable QAT
   - `TTT_PHASE2_UNFREEZE_BLOCKS=3` to revert to conservative TTT
   - `WARMDOWN_ITERS=3500` to revert warmdown

## Key Insights from Competition
- "Late QAT counterproductive with aggressive TTT" (PR #398)
- "Freezing early blocks during aggressive TTT creates internal inconsistency" (PR #398)
- "TTT paradoxically benefits from less-trained models" (PR #399)
- "Only PRs using simple EMA benefit from faster training" (PR #399)
- LeakyReLU(0.5)² gives better gradient flow than ReLU² (PR #434)

## Instance Status
- 86.38.238.153: Key rejected (port 22 reachable but Permission denied)
- Waiting for Ofir to update with new IP/credentials
- All code prepared locally, ready to deploy immediately

## Deployment
```bash
IP=<new_ip>
scp -i ~/.ssh/smedia4 -o StrictHostKeyChecking=no \
  /Users/newmacmini/pgolf/train_gpt_v8.py root@$IP:/mnt/vol/ours/train_gpt.py
ssh -i ~/.ssh/smedia4 -o StrictHostKeyChecking=no root@$IP \
  "cp /mnt/vol/ours/train_gpt.py /mnt/vol/ours/train_gpt_v8.py && \
   mkdir -p /dev/shm/data && cp -rn /mnt/vol/upstream/data/* /dev/shm/data/ 2>/dev/null && \
   cd /mnt/vol/ours && \
   DATA_PATH=/dev/shm/data/datasets/fineweb10B_sp1024 \
   TOKENIZER_PATH=/dev/shm/data/tokenizers/fineweb_1024_bpe.model \
   TORCH_COMPILE=0 \
   torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/run_v8.log"
```
