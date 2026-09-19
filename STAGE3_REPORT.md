# Stage 3 calibrated adaptive-routing report

## 1. Research question

Can post-hoc calibrated uncertainty reduce MoE expert compute while maintaining
the quality of the fixed Stage 1 checkpoint? This is an independently designed
temperature-scaling experiment; it does not reproduce Jev or RLCD.

## 2. Experimental setup

The 6,413,312-parameter Stage 1 checkpoint, Transformer weights, experts, and
router weights are frozen. Calibration uses the final
25,600
bytes from the training-token array. The untouched validation prefix contains
25,600
bytes. No validation targets influenced temperatures, thresholds, metric
choice, or final-policy selection.

Fixed Top-2, fixed Top-1, calibrated Top-1, three adaptive signals, and a
target-informed oracle reference are evaluated on identical batches. Timing
uses preloaded CUDA tensors, 50 warm-up
batches, and 20 complete timed repeats.

## 3. Calibration method

One scalar temperature is fitted per MoE layer using bounded one-dimensional
optimization. Hard calibration minimizes cross-entropy to the oracle-best
single expert. Soft calibration minimizes cross-entropy to
`softmax(-expert_loss / 1.0)`.

- Hard Oracle: layer 1 T=1.6230, layer 3 T=2.3800
- Soft Oracle: layer 1 T=3.2705, layer 3 T=7.2264

Temperature scaling does not change expert ordering. Consequently, both
calibrated Top-1 variants have exactly the same model outputs as Standard
Top-1; only their confidence metrics differ.

## 4. Adaptive-K policies

Maximum probability, entropy, and Top-1/Top-2 margin each choose between K=1
and K=2. K=2 uses the calibrated Top-2 probabilities renormalized over the two
selected experts. Threshold candidates are calibration-distribution quantiles.

The predeclared selection rule chooses the minimum experts/token among points
whose calibration loss is no more than
0.020 above Standard Top-2,
breaking ties by loss. The final policy was selected entirely on calibration:
**margin with hard_oracle temperature scaling at threshold 0.125602**. All other validation sweeps are labeled
predeclared comparisons and were not used to change this choice.

## 5. Results

| Model | Val loss | PPL | Experts/token | Compute reduction | Δ loss vs Top-2 | ECE | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard Top-2 | 2.0254 | 7.579 | 2.000 | 0.0% | +0.0000 | 0.1751 | 56220 |
| Standard Top-1 | 2.0682 | 7.910 | 1.000 | 50.0% | +0.0428 | 0.1751 | 70957 |
| Calibrated Top-1 Hard | 2.0682 | 7.910 | 1.000 | 50.0% | +0.0428 | 0.0303 | 70406 |
| Calibrated Top-1 Soft | 2.0682 | 7.910 | 1.000 | 50.0% | +0.0428 | 0.1241 | 73390 |
| Adaptive K - max_probability | 2.0431 | 7.714 | 1.302 | 34.9% | +0.0177 | 0.0303 | 54459 |
| Adaptive K - max_probability (uncalibrated control) | 2.0420 | 7.706 | 1.296 | 35.2% | +0.0166 | 0.1751 | 54852 |
| Adaptive K - entropy | 2.0457 | 7.735 | 1.411 | 29.4% | +0.0203 | 0.0303 | 52328 |
| Adaptive K - entropy (uncalibrated control) | 2.0440 | 7.721 | 1.299 | 35.1% | +0.0186 | 0.1751 | 50673 |
| Adaptive K - margin | 2.0411 | 7.699 | 1.297 | 35.1% | +0.0157 | 0.0303 | 46545 |
| Adaptive K - margin (uncalibrated control) | 2.0407 | 7.696 | 1.299 | 35.1% | +0.0153 | 0.1751 | 48349 |
| Oracle Adaptive-K | 1.9395 | 6.956 | 1.526 | 23.7% | -0.0859 | — | 52827 |

## 6. Calibration

Aggregate hard-label ECE changes from 0.1751 to 0.0303
for the calibration method used by the selected policy. Brier score changes
from 0.7373 to 0.6841; NLL changes from
1.3818 to 1.2667. Confidence/correctness correlation
changes from 0.091 to
0.110. Since temperature is a
monotone transform, ranking accuracy remains unchanged.

The objectives behave differently rather than one universally winning. Hard
calibration gives the best hard-label ECE/NLL. Soft calibration gives the best
soft-oracle squared distance (0.0667 versus 0.0914 for hard calibration), as
expected from its training target.

## 7. Quality vs compute

The calibration-selected policy uses 1.297
experts/token, a 35.1% reduction from Top-2, with validation
loss change +0.0157. Its measured throughput is
46545 tokens/s versus 56220
for the fixed Top-2 control. GPU kernels and Python dispatch overhead mean
expert-count reductions need not translate proportionally to wall-clock speed.
Here throughput changes by -17.2%: the selected adaptive implementation is
slower despite executing fewer experts, so this run shows an expert-compute
reduction but not an end-to-end latency improvement.

The matched uncalibrated margin control uses
1.299 experts/token at loss
2.0407. The calibrated policy changes
those values by -0.0012
experts/token and
+0.0004 loss.
This control is essential: better ECE alone does not establish a better
adaptive-compute decision rule.

Fixed Top-1 gives the endpoint: 1.0 expert/token at
loss 2.0682. The plotted validation curves contain every
predeclared threshold from the calibration sweep without smoothing.

## 8. Oracle upper bound

The oracle reference compares the routed Top-1 expert with the existing Top-2
mixture using the observed target loss for an isolated intervention at each
layer, choosing K=1 on ties. Applying those decisions simultaneously yields
loss 1.9395 at 1.526
experts/token.

This is target-informed and non-deployable. Because early-layer token
interventions can interact through later causal attention, it is a greedy
counterfactual oracle reference, not a proof of the globally optimal joint
K-mask. Per-layer isolated diagnostics are saved in `oracle_adaptive.json`.

## 9. Findings

1. **Does calibration improve confidence?** ECE/Brier/NLL above answer this;
   temperature scaling changes confidence but not Top-1 correctness.
2. **Can K=1 cases be identified?** The selected policy's validation point and
   conditional regret table quantify whether its K=1 subset is safer than
   fixed Top-1.
3. **Compute saved:** 35.1% by expert count.
4. **Quality cost:** +0.0157 validation loss.
5. **Versus fixed Top-1:** selected adaptive loss 2.0411
   versus 2.0682, at 1.297
   versus 1.000 experts/token.
6. **Oracle gap:** the target-informed reference reaches
   1.9395 at 1.526; the
   remaining distance indicates how much the simple confidence rule leaves
   unrecovered.
7. **Does calibration itself improve adaptive routing?** The matched
   uncalibrated control has loss 2.0407
   at 1.299 experts/token. Therefore,
   confidence calibration and adaptive-routing utility must be treated as
   separate findings.

Regret comparisons use forced-Top-1 regret only on decisions actually sent to
K=1. They do not call Top-2 mixture minus best-single loss “regret.”

## 10. Conclusion

**Partial success with a negative calibration-control result.** Adaptive K reduces compute with a small quality cost, and temperature scaling improves confidence metrics, but the matched uncalibrated adaptive control is at least as good. The compute benefit therefore cannot be attributed to calibration. This conclusion applies to one frozen checkpoint,
one training-tail calibration split, and one validation prefix; repeated seeds
would be required before making a general claim.
