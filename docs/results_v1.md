# Parameter Golf Results Log

## Run 1 — Baseline v1 (1 GPU, 300s)
- **Code**: train_gpt_v1.py
- **Config**: 1xH100, 500 iter, 300s wallclock, TORCH_COMPILE=0
- **Steps completed**: 138/500 (wallclock limited)
- **Training val_bpb**: 3.2274 (step 138)
- **Int6 roundtrip val_bpb**: 3.3170 (quantization degradation: +0.090)
- **Artifact size**: 17,200,176 bytes (OVER 16M LIMIT)
  - Model int6+zstd: 17,126,884 bytes
  - Code: 73,292 bytes
- **Quantization**: int5 MLP, int6 attn, 3% magnitude pruning
- **Notes**: Baseline without Value Residual, Gated Attention

## Run 2 — v2 (VR + GA + GPTQ-lite, 1 GPU, 300s)
- **Changes**: Value Residual, Gated Attention, GPTQ-lite, warmdown 3500, QAT@0.15, 5% prune
- **Steps completed**: 130/500
- **Training val_bpb**: 3.2445 (step 130)
- **Int6 roundtrip val_bpb**: 3.5906 (MUCH WORSE — degradation: +0.346)
- **Artifact size**: 17,003,644 bytes (OVER 16M)
- **Problem**: GPTQ-lite clip search likely broken — caused 4x worse quantization degradation
- **Conclusion**: GPTQ-lite implementation is harmful, revert it

## Run 3 — v3 (VR + GA, int5 all, 8% prune, int5 QAT, 1 GPU, 300s)
- **Changes**: Reverted GPTQ-lite, int5 for ALL weights, 8% prune, int5 QAT
- **Steps completed**: 129/500
- **Training val_bpb**: 3.2520 (step 129)
- **Int5 roundtrip val_bpb**: 3.4473 (degradation: +0.195)
- **Artifact size**: 16,491,530 bytes (still OVER by 492K)
- **Conclusion**: Int5 for attention causes more quant degradation than int6

## Run 4 — v4 (VR + GA, int5 MLP/int6 attn, 15% prune, 1 GPU, 300s)
- **Changes**: Revert to int5 MLP/int6 attn, 15% magnitude pruning
- **Status**: INTERRUPTED — spot instance reclaimed at step ~100

## Key Findings
1. Value Residual + Gated Attention add ~7% per-step overhead
2. GPTQ-lite implementation was broken — avoid
3. Int5 for all weights saves ~500K but causes +0.1 BPB degradation
4. 15% magnitude pruning was untested (server died)
5. Artifact needs to be under 16M — 11 layers with int6 attn produces 17.1M
6. Options: reduce to 10 layers, or aggressive pruning (15-20%)
