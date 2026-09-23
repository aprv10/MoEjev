# Stage 6 larger-MoE replication report

## Result

**The learned-gate advantage survived only partially at larger scale.** Three independently initialized 35.62M-parameter MoEs were
evaluated with the unchanged Stage 4/5 compute-gate method. Negative deltas mean
the learned MLP gate has lower validation loss than router margin at the nearest
measured expert-compute point.

## Frozen protocol

All base models use six Transformer blocks, width 384, six attention heads,
three dense FFNs, three 8-expert MoE FFNs, context 128, and Top-2 routing. Only
base seeds 601/602/603 differ. Each frozen model uses the same 25,153-parameter
hidden-plus-router MLP, unchanged four router scalars, one-hot layer identity,
Huber regression target, 20 epochs, and gate seeds 11/22/33. Blocks 0–799 are
gate training data, block 800 is unused, blocks 801–1200 are gate calibration,
and validation is used only for final evaluation. No method tuning used
validation feedback.

## Base-model metrics

| Seed | Parameters | Validation loss | Perplexity | Router entropy | Utilization std |
|---:|---:|---:|---:|---:|---:|
| 601 | 35.62M | 1.7639 | 5.84 | 1.4730 | 0.0230 |
| 602 | 35.62M | 1.7624 | 5.83 | 1.4613 | 0.0251 |
| 603 | 35.62M | 1.7716 | 5.88 | 1.4991 | 0.0225 |

Full per-layer eight-expert utilization vectors are in each model's
`baseline_metrics.json` and `baseline_evaluation.json`.

## Per-model mean MLP − margin loss

| MoE seed | 1.10 | 1.20 | 1.30 | 1.40 | 1.50 |
|---:|---:|---:|---:|---:|---:|
| 601 | -0.0030 | -0.0017 | -0.0013 | -0.0006 | -0.0003 |
| 602 | -0.0019 | +0.0023 | +0.0036 | +0.0034 | +0.0024 |
| 603 | -0.0018 | -0.0007 | -0.0006 | -0.0004 | -0.0007 |

## Aggregate matched-compute result

- 1.10: mean -0.0022, 3/3 models won, 95% t-CI [-0.0039, -0.0005]
- 1.20: mean -0.0000, 2/3 models won, 95% t-CI [-0.0052, +0.0051]
- 1.30: mean +0.0006, 2/3 models won, 95% t-CI [-0.0061, +0.0072]
- 1.40: mean +0.0008, 2/3 models won, 95% t-CI [-0.0048, +0.0064]
- 1.50: mean +0.0005, 2/3 models won, 95% t-CI [-0.0036, +0.0046]

These are nearest measured operating points rather than interpolated values.
The maximum absolute target mismatch is 0.0119
experts/token for the MLP and 0.0050 for margin.

## Gate prediction and selected-policy benchmarks

| MoE | Gate seed | Pearson | Experts/token | Validation loss | Fraction K=2 | Tokens/s | Peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 601 | 11 | +0.1009 | 1.100 | 1.7803 | 0.100 | 10527 | 163.0 |
| 601 | 22 | +0.1054 | 1.102 | 1.7794 | 0.102 | 10690 | 163.0 |
| 601 | 33 | +0.1021 | 1.100 | 1.7809 | 0.100 | 10647 | 163.0 |
| 602 | 11 | +0.1100 | 1.107 | 1.7783 | 0.107 | 10704 | 163.0 |
| 602 | 22 | +0.1110 | 1.106 | 1.7773 | 0.106 | 10912 | 163.0 |
| 602 | 33 | +0.1043 | 1.103 | 1.7795 | 0.103 | 10725 | 163.0 |
| 603 | 11 | +0.1029 | 1.209 | 1.7829 | 0.209 | 7796 | 163.0 |
| 603 | 22 | +0.0960 | 1.200 | 1.7871 | 0.200 | 10243 | 163.0 |
| 603 | 33 | +0.1015 | 1.204 | 1.7832 | 0.204 | 9709 | 163.0 |

These throughput measurements use the calibration-selected operating point;
fewer expert executions are not treated as proof of a speedup.

## Answers to the research questions

1. **Did the advantage survive scale?** The predeclared verdict is **partial**.
2. **At which budgets?** Aggregate delta is negative at 1.10, 1.20 experts/token.
3. **Consistent across all three models?** All three larger MoEs win at 1.10 experts/token.
4. **Effect versus Stage 5:** The mean absolute 1.10–1.30 effect is **smaller** (0.56× Stage 5). Exact comparisons are below.
5. **Oracle headroom:** All models retain positive headroom: True; mean Top-2 minus oracle loss is 0.0907.
6. **Runtime:** MLP gating averages 10217 tokens/s versus 10964 for fixed Top-2: no runtime improvement (0.93× fixed-Top-2 throughput). Peak selected-gate memory is 163.0 MiB allocated and 1328.0 MiB reserved.
7. **Adaptive depth next?** No—not from Stage 6 alone; the larger-model compute-gating result is not sufficiently consistent.

## Comparison with Stage 5

- 1.10: Stage 6 -0.0022 versus Stage 5 -0.0027
- 1.20: Stage 6 -0.0000 versus Stage 5 -0.0017
- 1.30: Stage 6 +0.0006 versus Stage 5 -0.0006
- 1.40: Stage 6 +0.0008 versus Stage 5 +0.0001
- 1.50: Stage 6 +0.0005 versus Stage 5 +0.0006

## Limitations

- Only three checkpoints of one larger architecture were evaluated.
- Confidence intervals are wide and descriptive at N=3.
- Gate seeds are nested within checkpoints and are not independent base-model replications.
- Nearest measured points are approximate compute matches, not interpolation.
- Oracle decisions use validation targets and are non-deployable.
- Runtime reflects an unoptimized Python/PyTorch dynamic-dispatch implementation.

This project is inspired by calibrated decision models, but it does not
reproduce Jev or RLCD. Stage 6 changes model scale and expert count while
keeping the learned compute-gate method fixed.
