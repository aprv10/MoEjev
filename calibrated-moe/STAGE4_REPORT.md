# Stage 4 learned compute-gate report

## Scope and question

This experiment asks whether a tiny model can predict the isolated value of the
second routed expert and thereby improve the validation quality–expert-compute
frontier over simple router-confidence heuristics. It uses one frozen tiny MoE
checkpoint; it does not reproduce Jev or RLCD.

## Protocol and leakage controls

- Gate training: blocks 0–799 (102400 input bytes).
- Gate calibration: blocks 801–1200 (51200 input bytes).
- One complete block is unused between these windows, preventing their shifted language-model targets from sharing a boundary byte.
- Final validation is the unchanged first 50 validation batches (25600 bytes).
- `delta_loss = isolated Top-1 loss - normal Top-2 loss`; oracle losses are labels and diagnostics only.
- Features are computed before the current MoE expert output: pre-MoE normalized hidden state, Top-1 and Top-2 router probabilities, margin, entropy, and one-hot layer identity.
- No target, expert loss, future hidden state, or downstream activation enters the gate. Normalization statistics come only from gate training. Thresholds and primary operating points come only from gate calibration. Validation frontiers are explicitly exploratory.
- Transformer, router, and expert parameters remained unchanged: **True**.

## Primary results (three gate initialization seeds)

| Gate | Parameters | Validation Pearson | Selected experts/token | Validation loss | Online tokens/s |
|---|---:|---:|---:|---:|---:|
| linear router only | 7 | 0.0828 ± 0.0153 | 1.2304 ± 0.0449 | 2.0469 ± 0.0009 | 43748 ± 5194 |
| linear hidden router | 263 | 0.0928 ± 0.0059 | 1.1996 ± 0.0050 | 2.0478 ± 0.0008 | 46832 ± 1175 |
| mlp hidden router | 16897 | 0.1321 ± 0.0047 | 1.2055 ± 0.0072 | 2.0445 ± 0.0005 | 48222 ± 744 |

Top-1 is 2.0682 at 1.000 experts/token; Top-2 is
2.0254 at 2.000; the non-deployable oracle is
1.9395 at 1.526.

## Answers to the predeclared questions

1. **Can delta loss be predicted?** Only weakly: the best model's held-out Pearson correlation is 0.1321. The decile and scatter plots show useful ranking structure, but substantial irreducible or unmodeled variation remains.
2. **Do hidden states help?** A linear hidden-state gate does not: at the independently calibration-selected point, adding hidden features changes mean validation loss by +0.0009 versus the router-only linear gate (positive is worse). The nonlinear hidden-state MLP does improve matched-compute loss, so the benefit is architecture-dependent rather than evidence that raw hidden features alone suffice.
3. **Does a learned gate beat simple heuristics?** The MLP beat margin in 12/15 seed-specific nearest-point comparisons. Its mean loss differences versus margin by target experts/token were 1.1: -0.0054, 1.2: -0.0035, 1.3: -0.0019, 1.4: -0.0004, 1.5: +0.0006; negative values favor the MLP.
4. **Matched compute:** `matched_compute.json` uses the nearest actually measured point at 1.10–1.50 experts/token; no interpolation or smoothing is used.
5. **Frontier shift:** The MLP produces a modest frontier improvement from roughly 1.10 through 1.40 experts/token and is effectively tied near 1.50. The linear gates do not consistently shift the frontier. The validation sweep was not used to change architecture, features, epochs, or thresholds.
6. **Gap to oracle:** The best preselected learned variant is mlp hidden router at mean loss 2.0445; the oracle is 1.9395. The oracle is a target-informed upper bound, not deployable.
7. **Runtime:** The MLP averages 48222 tokens/s versus 56220 for Stage 3 Top-2, so this implementation is slower despite executing fewer experts. Expert count is algorithmic compute; tokens/s includes gate and dynamic-dispatch overhead.
8. **Seed stability:** Each gate was trained from seeds [11, 22, 33]; mean and population SD are reported rather than selecting a favorable seed.
9. **Scaling criterion:** Scaling is justified only if learned gates repeatedly lower loss versus margin at tightly matched expert counts, with stable seeds and a meaningful remaining oracle gap. A prediction correlation alone is insufficient.

## Interpretation

The strongest preselected learned result is **mlp hidden router**.
The MLP beat margin in 12/15 seed-specific nearest-point comparisons. Its mean loss differences versus margin by target experts/token were 1.1: -0.0054, 1.2: -0.0035, 1.3: -0.0019, 1.4: -0.0004, 1.5: +0.0006; negative values favor the MLP. This is evidence about one checkpoint and one validation
region only; it does not establish general superiority of learned adaptive MoE
routing. See `matched_compute.json`, per-seed threshold sweeps, conditional
analyses, and worst-false-K1 tables for the complete result rather than relying
on one operating point.
