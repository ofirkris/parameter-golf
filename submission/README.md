# 10L XSA-all + EMA + LeakyReLU^2 + Legal TTT

## Results
Pending 8xH100 validation run.

## Architecture
- 10 layers, 512d, 8/4 GQA, 3x MLP (1536), LeakyReLU(0.5)^2
- XSA on all 10 layers, EMA 0.997, Partial RoPE 16/64, LN Scale
- SmearGate + BigramHash(10240, 128), U-Net skips, softcap=30

## Training
- BF16, Muon (WD=0.04, momentum=0.99, LR=0.02) + AdamW (LR=0.03)
- 786K batch tokens, seq 2048, warmdown 3000
- Late QAT @0.15

## Quantization
- GPTQ-lite (multi-percentile clip search)
- Int5 MLP / Int6 attention, FP16 embeds
- Binary search pruning (exact 16MB fit), zstd-22

## Eval
- 30 epoch legal sequential score-then-train TTT
- AdamW lr=0.0005, cosine LR, per-layer scaling (mlp.proj 3x, mlp.fc 0.5x)
- 5-expert Hedge Mixer: neural + unigram + bigram + trigram + entropy
- Sliding window stride=64 (non-TTT fallback)

## Environment
- PyTorch 2.7.0+cu128, FlashAttention 2.8.3, 8xH100 SXM
