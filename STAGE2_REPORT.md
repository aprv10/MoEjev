# Stage 2 oracle-routing report

This is an observational analysis of the unchanged Stage 1 checkpoint. It does
not implement or train a calibrated router.

## Method

For each routed token and target MoE layer, every candidate expert is evaluated
by an isolated single-token intervention. All other tokens retain their
baseline Top-2 MoE outputs at that layer; the selected token receives one
expert at unit weight; the unchanged remaining blocks produce the next-byte LM
loss at that position. This is an exact counterfactual within the forced-single-
expert intervention family. The actual Top-2 mixture is separately retained.

The forced Top-1 regret is nonnegative and compares like with like. The
`top2_mixture_minus_best_single` quantity is signed: negative values mean the
trained Top-2 mixture beat every forced single expert, so it is not labeled
regret.

## Run scope

- Source checkpoint: Stage 1 run `baseline-top2-small-20260919-151643` (`last.pt` is excluded from Git).
- Unique validation target bytes: 25,600
- Token-layer decisions: 51,200
- Oracle temperature: 1
- Wall time: 11.82 seconds
- Throughput: 4333.3 token-layer decisions/s
- Peak allocated/reserved CUDA memory: 203.8 / 278.0 MiB

## A. Do experts meaningfully differ?

The median best-vs-second loss gap is 0.299316; P90 is
0.988281, P95 is 1.296875, and P99 is 2.201172.
3.5% of choices are below 0.01,
while 77.4% exceed 0.1.

The strong-win rule is `Tukey upper fence (Q3 + 1.5 * IQR)` at a gap of
1.345184; it identifies 4.4% of
token-layer decisions.

**Finding:** yes, within this intervention definition. Only
3.5% are near-ties below 0.01,
and the long gap tail is large enough that expert identity is a meaningful
selection variable rather than a cosmetic distinction.

## B. Does the router select the best expert?

- Top-1 oracle accuracy: 45.40%
- Top-2 oracle coverage: 69.31%
- Most common oracle expert: Expert 3 (27.36%)
- Strong-win Top-1 accuracy: 68.54%
- Strong-win Top-2 coverage: 76.10%

The router is substantially better than uniform random selection (25% Top-1,
50% Top-2), especially on strong wins, but it still misses the best forced
single expert on 54.6% of all
decisions and outside its Top-2 on
30.7%.

## C. How expensive are routing mistakes?

Forced Top-1 mean regret is 0.340932; median 0.050812;
P95 1.404297; P99 2.291016; maximum
5.711182. Strong-win mean regret is
0.674955.

The actual Top-2 mixture minus best forced-single mean is
0.321287. This signed result
must not be interpreted as a pure router regret because it compares a mixture
against a single expert.
The Top-2 mixture beats every forced single expert on
16.0% of decisions, which
is direct evidence that Top-2 cannot simply be treated as a worse K=1 policy.

## D. Is router confidence meaningful?

Mean router entropy is 0.9736 nats versus
1.2269 nats for the temperature-scaled soft
oracle. The exploratory binned confidence error is
0.1751. This is exploratory evidence,
not a definitive calibration measurement.

Router confidence has only a
0.091 correlation with oracle
correctness and a -0.053 correlation
with regret. The reliability curve is notably non-monotonic in layer 3, so
confidence contains some signal but is not a dependable decision rule as-is.

Distribution alignment: KL(q || p)
0.4311, cross-entropy
1.6579, squared
probability distance 0.2131,
and cosine similarity 0.7685.

## E. Are experts showing specialization?

- Layer 1 expert 1: non_ascii_utf8 is the largest supported enrichment in input_category (2.96x; 0.95% within this expert's wins).
- Layer 1 expert 2: newline is the largest supported enrichment in target_category (3.47x; 1.49% within this expert's wins).
- Layer 1 expert 3: punctuation is the largest supported enrichment in target_category (2.28x; 7.87% within this expert's wins).
- Layer 1 expert 4: digit is the largest supported enrichment in input_category (1.34x; 3.38% within this expert's wins).
- Layer 3 expert 1: punctuation is the largest supported enrichment in target_category (1.95x; 6.71% within this expert's wins).
- Layer 3 expert 2: alphabetic is the largest supported enrichment in target_category (1.21x; 89.46% within this expert's wins).
- Layer 3 expert 3: newline is the largest supported enrichment in input_category (3.29x; 1.42% within this expert's wins).
- Layer 3 expert 4: punctuation is the largest supported enrichment in input_category (3.36x; 11.58% within this expert's wins).

These are byte/context enrichments, not semantic expert labels. Representative
contexts are saved separately and should be read before assigning meaning.
The analyzed validation prefix is dominated by a small number of WikiText
articles, so these patterns are preliminary and corpus-local.

## F. Is Stage 3 justified?

**Proceed to Stage 3 as a controlled, falsifiable experiment.** The reason is
not that calibration has been proven useful: it is that expert choices have
material loss differences, the existing router leaves measurable selection
headroom, and its confidence is sharper than the temperature-1 oracle evidence
and only weakly aligned with correctness. Those are the prerequisites for a
calibration experiment.

There are important cautions. The forced-single oracle uses the observed next
byte and is therefore an unattainable ceiling, not a deployable router. Soft-
oracle entropy depends on temperature. The actual Top-2 mixture beats every
single expert for a meaningful minority of decisions. Stage 3 should therefore
learn/evaluate uncertainty on held-out data and test K=1 versus K=2 without
assuming that the best single-expert label fully describes mixture quality.
The quality-versus-compute benchmark can still reject the hypothesis.

## Data products

The result directory contains the token-level Parquet file, JSON summaries,
case studies, and publication-style static plots. Raw expert losses are saved,
so alternative soft-oracle temperatures do not require counterfactual reruns.
