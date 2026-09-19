# Stage 5 cross-checkpoint replication report

## Result

**Strong directional replication under the predeclared framework, with statistical caution.** Five independently initialized and trained MoE checkpoints
were evaluated. The original Stage 4 checkpoint is not included in the
replication average. Negative deltas below mean the MLP gate has lower loss than
the margin heuristic at approximately matched expert compute.

## Frozen protocol

Every base model uses the exact Stage 1 architecture and 1,000-step training
configuration; only seeds 101, 202, 303, 404, and 505 differ. Each frozen model
uses the unchanged Stage 4 16,897-parameter MLP, hidden-plus-router features,
Huber regression target, 20 epochs, gate seeds 11/22/33, blocks 0–799 for gate
training, block 800 unused, blocks 801–1200 for gate calibration, and the first
50 validation batches for final evaluation. Thresholds are derived and selected
without validation feedback. No checkpoint or gate seed was selected or removed.

## Per-checkpoint mean Δloss

| MoE seed | 1.10 | 1.20 | 1.30 | 1.40 | 1.50 |
|---:|---:|---:|---:|---:|---:|
| 101 | -0.0052 | -0.0027 | +0.0001 | +0.0018 | +0.0021 |
| 202 | -0.0031 | -0.0018 | -0.0003 | +0.0004 | +0.0007 |
| 303 | -0.0030 | -0.0022 | -0.0027 | -0.0029 | -0.0023 |
| 404 | -0.0033 | -0.0024 | -0.0003 | +0.0006 | +0.0007 |
| 505 | +0.0011 | +0.0008 | +0.0000 | +0.0008 | +0.0020 |

## Aggregate matched-compute result

- 1.10: mean -0.0027, 4/5 checkpoints won, 95% t-CI [-0.0056, +0.0001]
- 1.20: mean -0.0017, 4/5 checkpoints won, 95% t-CI [-0.0034, +0.0001]
- 1.30: mean -0.0006, 3/5 checkpoints won, 95% t-CI [-0.0021, +0.0008]
- 1.40: mean +0.0001, 1/5 checkpoints won, 95% t-CI [-0.0021, +0.0023]
- 1.50: mean +0.0006, 1/5 checkpoints won, 95% t-CI [-0.0015, +0.0028]

## Answers to the research questions

1. **Replication:** The predeclared verdict is **strong**.
2. **Most consistent budgets:** The effect is most consistent at 1.10 and 1.20 experts/token (4/5 checkpoints each), remains directionally favorable at 1.30 (3 wins, 2 numerical ties), and does not persist at 1.40–1.50.
3. **Magnitude:** Fresh-checkpoint mean deltas are -0.0027, -0.0017, -0.0006, +0.0001, and +0.0006 from 1.10 through 1.50. Full median, SD, range, confidence intervals, and seed-level counts are in `aggregate/aggregate_metrics.json`.
4. **Base-checkpoint variation:** At 1.10, 1.20, and 1.30, between-checkpoint SD is 0.0023, 0.0014, and 0.0012 versus mean within-checkpoint gate-seed SD of 0.0006, 0.0010, and 0.0010. Base-model variation is therefore at least as important and is clearest at 1.10.
5. **Gate initialization:** Three seeds were averaged within every checkpoint; no best seed was selected.
6. **Hidden-state gating:** It is useful on most, but not all, checkpoints at aggressive budgets: seed 505 disagrees at 1.10–1.20. Stage 5 deliberately does not repeat the Stage 4 feature ablation, so this is a method-level replication against margin rather than a new causal attribution to hidden features alone.
7. **Prediction versus routing:** Mean held-out Pearson is 0.1307; its run-level correlation with mean routing delta is -0.5293. Because negative routing delta is better, the negative association suggests better prediction tends to accompany better routing, but the 15 runs are nested and the N=5 checkpoint correlation is exploratory.
8. **Oracle headroom:** All checkpoints have positive oracle headroom: True; mean Top-2 minus oracle loss is 0.0767.
9. **Runtime:** MLP gating averages 43747 tokens/s versus 49053 for fixed Top-2, so there is no wall-clock speedup. Selected-gate benchmarks peak at 123.1 MiB allocated and 660.0 MiB reserved.
10. **Scaling:** A larger-model experiment is justified only as a cautious follow-up if the aggregate direction and checkpoint consistency are favorable; N=5 is not definitive evidence of generality.

## Exploratory router-behavior relationships

At the 1.20-expert budget, checkpoint-level correlations between MLP-minus-margin
delta and base/router diagnostics are: top2 loss +0.611, router entropy +0.444, expert utilization std -0.192, top1 oracle accuracy -0.171, top2 oracle coverage -0.224, mean forced top1 regret +0.581. These are
descriptive only (N=5), were computed after the frozen experiment, and did not
influence the gate or checkpoint inclusion.

## Comparison with the original Stage 4 checkpoint

The fresh-checkpoint effect has the same favorable direction at 1.10–1.30 but
is weaker than Stage 4: -0.0027 versus -0.0054 at 1.10, -0.0017 versus -0.0035
at 1.20, and -0.0006 versus -0.0019 at 1.30. At 1.40 the replication mean is
essentially zero and slightly favors margin; at 1.50 it matches Stage 4's
small margin-favoring result. All N=5 confidence intervals include zero, so the
`strong` label describes consistency under the predeclared directional
framework—not definitive statistical significance.

## Limitations

- Only five fresh checkpoints of one tiny architecture were evaluated.
- Confidence intervals are wide and descriptive at N=5.
- Gate seeds are nested within checkpoints and are not independent base-model replications.
- Nearest measured points are approximate compute matches, not interpolation.
- Oracle decisions use validation targets and are non-deployable.
- Runtime reflects an unoptimized Python/PyTorch dynamic-dispatch implementation.

This project is inspired by calibrated decision models, but it does not
reproduce Jev or RLCD. Stage 5 establishes only replication behavior for one
tiny architecture, dataset, training recipe, and frozen routing intervention.
