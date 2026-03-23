# 10L XSA + EMA + VR + VE128 + TTT

## Results
Pending 8xH100 validation run.

## Architecture
- 10 layers, 512d, 8/4 GQA, 3x MLP (1536), ReLU^2
- XSA last 4 layers, EMA 0.997, Partial RoPE 16/64, LN Scale
- Value Residual (layer-0 V blended into all layers)
- Value Embeddings VE128 at layers 8,9
- SmearGate + BigramHash(10240, 128), U-Net skips, softcap=30

## Training
- BF16, Muon (WD=0.04, momentum=0.99) + AdamW
- 786K batch tokens, seq 2048, warmdown 3000
- Late QAT @0.15

## Quantization
- GPTQ-lite (multi-percentile clip search)
- Int5 MLP / Int6 attention, FP16 embeds
- Binary search pruning (exact 16MB fit), zstd-22

## Eval
- 150 epoch sequential score-then-train TTT
- AdamW lr=0.0005, cosine LR, per-layer scaling (mlp.proj 3x, mlp.fc 0.5x)
- Sliding window stride=64 (non-TTT fallback)

## Environment
- PyTorch 2.7.0+cu128, FlashAttention 2.8.3, 8xH100 SXM
