# Learning when MoEs need more expert compute

This repository explores whether a small learned gate can predict when a token actually benefits from additional expert compute in a Mixture-of-Experts model. The goal is to improve the quality-vs-compute tradeoff, not model quality itself.

The project started from an interest in calibrated decision models such as Jev, but the experiments here are independently designed and do not reproduce Jev or RLCD.

## Current scope: Stages 1–5

Stage 1 implements a controlled baseline:

- a decoder-only, pre-layer-normalized causal Transformer;
- byte-level WikiText-2 input with no pretrained model or tokenizer;
- dense feed-forward layers alternating with four-expert MoE layers;
- a standard learned softmax router and token-level Top-2 dispatch;
- Switch-style auxiliary load balancing and router z-loss;
- fp16 CUDA training, gradient accumulation, gradient clipping, and local
  checkpoints/JSON metrics;
- validation loss, perplexity, expert utilization, routing entropy,
  throughput, peak allocated GPU memory, and peak allocator-reserved memory.

Stage 2 adds evaluation-only, isolated single-token expert interventions and
routing analysis. Stage 3 fits per-layer post-hoc temperatures on a distinct
training-tail calibration split and evaluates fixed Top-1/Top-2, three
adaptive-K signals, matched uncalibrated controls, and a target-informed oracle
reference. It does not add a neural router or retrain model weights. The public
dashboard remains unimplemented.

Stage 4 keeps that Transformer, router, and all experts frozen, then trains a
separate tiny compute-value gate to predict whether the second routed expert is
useful. It compares linear router-only, linear hidden-plus-router, and tiny MLP
gates over three initialization seeds. Thresholds are selected on a disjoint
training-derived gate-calibration window; the original validation region is
used only for final measurement. On this checkpoint, the MLP modestly improves
the quality–expert-compute frontier over the margin heuristic at roughly
1.10–1.40 experts/token, but its current dynamic implementation is slower in
wall-clock throughput than fixed Top-2.

Stage 5 repeats the frozen Stage 4 MLP method across five independently trained
Stage 1 MoE checkpoints and three gate initializations per checkpoint. The
learned gate's advantage replicates directionally at aggressive budgets
(1.10–1.30 experts/token), but with smaller effects than the original Stage 4
checkpoint, one disagreeing base seed at 1.10–1.20, and N=5 confidence
intervals that include zero. It does not produce a wall-clock speedup.

## Hardware sizing

Capture the machine state before training:

```powershell
python scripts/check_hardware.py --output results/hardware.json
```

`configs/baseline_small.yaml` is designed for a 6 GB RTX 3060 Laptop GPU. It
uses four layers at width 256, two MoE layers, 128-token sequences, micro-batch
four, and eight-step gradient accumulation. The smaller `smoke.yaml` confirms
the complete pipeline in a few optimizer steps.

## Run

From this directory:

```powershell
python -m unittest discover -s tests -v
python training/train.py --config configs/smoke.yaml
python training/train.py --config configs/baseline_small.yaml
python evaluation/run_oracle_analysis.py --config configs/oracle_analysis.yaml
python evaluation/stage3_benchmark.py --config configs/stage3_calibrated_routing.yaml
python evaluation/stage4_compute_gate.py --config configs/stage4_learned_compute_gate.yaml
python evaluation/stage5_replication.py --config configs/stage5_cross_checkpoint_replication.yaml
```

For a short check of the full default architecture without changing its saved
configuration:

```powershell
python training/train.py --config configs/baseline_small.yaml --max-steps 10
```

Each run creates an isolated directory under `results/` containing:

- `config.resolved.json`: exact experiment configuration;
- `metrics.jsonl`: append-only training metrics;
- `summary.json`: final validation, routing, compute, and memory metrics;
- `last.pt`: model/optimizer checkpoint and configuration.

Raw run directories, checkpoints, cached features, and token-level tables are
ignored by Git. A curated snapshot of lightweight JSON/CSV summaries and plots
from the canonical Stage 1–5 runs is versioned under `results/` so the reported
findings can be inspected without downloading model weights or regenerating
large intermediate data.

The metrics from a five- or ten-step smoke run verify correctness and memory
fit; they are not research results. Meaningful comparisons require fully
trained, seed-controlled runs with the same non-router architecture and data.

See [STAGE1_REPORT.md](STAGE1_REPORT.md) for the verified local baseline run,
hardware measurements, known caveats, and fixes made during smoke testing.
See [STAGE2_REPORT.md](STAGE2_REPORT.md) for the oracle-routing evidence and
the critical decision about whether a calibration experiment is warranted.
See [STAGE3_REPORT.md](STAGE3_REPORT.md) for the post-hoc calibration and
adaptive-compute result, including matched uncalibrated controls.
See [STAGE4_REPORT.md](STAGE4_REPORT.md) for the learned compute-value gates,
three-seed results, matched-compute comparisons, and runtime caveat.
See [STAGE5_REPORT.md](STAGE5_REPORT.md) for the five-checkpoint replication,
nested gate-seed analysis, oracle headroom, and aggregate frontier.

## Layout

```text
configs/       reproducible experiment definitions
data/          WikiText-2 preparation and local cache
moe/           Transformer, experts, and baseline router
training/      training loop and CLI
evaluation/    reserved for Stage 2 onward
results/       curated summaries/plots; heavy local artifacts remain ignored
scripts/       hardware and experiment utilities
tests/         focused Stage 1 correctness tests
```
