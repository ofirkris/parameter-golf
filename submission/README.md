# XSA + EMA + Partial RoPE + LN Scale

## Results
- **val_bpb: 1.1365** (seed 42, 8xH100 SXM, 600s)
- Artifact: 15,759,319 bytes (under 16MB)
- 6491 steps, 92ms/step

## Architecture
- 10 layers, 512d, 8/4 GQA, 3x MLP (1536), ReLU^2
- XSA on last 4 layers, EMA 0.997, Partial RoPE 16/64, LN Scale
- SmearGate + BigramHash(10240, 128), U-Net skips, softcap=30

## Training
- BF16 mixed precision on 8xH100 SXM
- Muon (WD=0.04, momentum=0.99) + AdamW
- 786K batch tokens, seq 2048, warmdown 3000
- Optional FP8 via FP8_ENABLED=1 (experimental, ~13% faster but uses torch._scaled_mm)

## Quantization
- Int5 MLP / Int6 attention per-row, FP16 embeds, 3.2% pruning, zstd-22

## Eval
- Sliding window stride=64, scores all tokens including final partial window

## Environment
- PyTorch 2.7.0+cu128, FlashAttention 2.8.3, 8xH100 SXM 80GB
