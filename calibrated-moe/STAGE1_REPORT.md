# Stage 1 verification report

Verified locally on 2026-09-19. This is a baseline engineering run, not a
calibrated-router comparison or a publishable benchmark.

## Machine

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- VRAM: 6,144 MiB total; 5,870 MiB free at the initial probe
- NVIDIA driver: 581.95 (reports CUDA 13.0 support)
- Installed CUDA toolkit: 12.9
- PyTorch: 2.7.1+cu128; CUDA runtime 12.8; CUDA available
- Mixed precision: fp16 used (the device also reports bf16 support)

The exact probe output is saved locally in `results/hardware.json`.

## Architecture exercised

- 6,413,312 trainable parameters
- byte vocabulary: 256 UTF-8 byte values
- context: 128 bytes
- four pre-norm causal Transformer blocks, width 256, four attention heads
- dense 1,024-wide feed-forward networks in blocks 0 and 2
- four 1,024-wide feed-forward experts in blocks 1 and 3
- standard softmax linear router with token-level Top-2 selection
- selected-expert outputs combined by normalized router probabilities
- load-balancing coefficient 0.01; router z-loss coefficient 0.001
- micro-batch 4, accumulation 8, fp16 autocast, AdamW

## Verified run

Command:

```powershell
python training/train.py --config configs/baseline_small.yaml
```

Run: `results/baseline-top2-small-20260919-151643`

- optimizer steps: 1,000
- training bytes processed: 4,096,000
- final-step language-model loss: 2.0917
- validation loss: 2.0254
- validation perplexity: 7.5792
- throughput: 26,188 tokens/second
- peak PyTorch allocation: 155.6 MiB
- peak PyTorch reserved memory: 164.0 MiB
- mean routing entropy: 0.9736 nats (maximum for four experts: 1.3863)

Per-layer validation assignment fractions:

| MoE block | Expert 1 | Expert 2 | Expert 3 | Expert 4 |
|---|---:|---:|---:|---:|
| 1 | 25.45% | 26.80% | 25.65% | 22.09% |
| 3 | 25.76% | 25.67% | 25.16% | 23.41% |

The validation values cover the configured first 50 batches (25,600 target
bytes). Since tokenization is byte-level, perplexity is per byte and must not
be compared directly with subword-token perplexities.

## Verification and issues found

Four tests pass: router normalization/assignment count, CPU forward/backward,
CUDA fp16 forward/backward, and next-token dataset alignment. The saved
checkpoint was reloaded successfully and contains 79 model-state tensors at
step 1,000.

Two implementation problems were caught during smoke testing and fixed:

1. Windows kept a NumPy memory-map handle open during cleanup. The deliberately
   small token arrays now load directly into RAM.
2. CUDA autocast produced fp16 expert outputs while the residual accumulator
   was fp32, which made sparse `index_add_` fail. Expert outputs are now cast to
   the accumulator dtype, with a CUDA regression test covering the path.

The router was strongly imbalanced during the five-step smoke run. After the
full run, utilization is close to uniform in both layers, showing that the
auxiliary balancing objective is functioning. This does not yet demonstrate
useful expert specialization.

