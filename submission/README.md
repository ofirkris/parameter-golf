# Tap Mobile — Parameter Golf Submission

## Techniques
- 10 transformer layers, 512d, 8 heads / 4 KV heads (GQA), 3x MLP (1536 hidden)
- Exclusive Self-Attention (XSA) on last 4 layers (arXiv:2603.09078)
- EMA (decay=0.997) replacing SWA
- Partial RoPE (16/64 dims) — position-free dims for generalization
- LN Scale (1/sqrt(layer_idx+1)) — depth-dependent normalization
- SmearGate + BigramHash(10240, dim=128)
- U-Net skip connections
- Orthogonal initialization, logit softcap=30
- Muon optimizer (WD=0.04, momentum 0.99)
- Int5 MLP / Int6 attention mixed quantization + zstd-22
- FP16 embedding passthrough, 3% magnitude pruning
- Sliding window evaluation (stride=64)

## Base
Built on thwu1's merged #1 submission (1.1428 BPB) with zero-parameter improvements from PR #315.

## Training
- 8xH100 SXM, ~600s wallclock
- PyTorch 2.7.0 + FlashAttention 2.8.3
- Batch tokens: 786,432
- Sequence length: 2048
