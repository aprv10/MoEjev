from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import ByteTokenizer, TokenBlockDataset, prepare_wikitext2  # noqa: E402
from evaluation.oracle import (  # noqa: E402
    checkpoint_parameter_fingerprint,
    evaluate_layer_oracle,
)
from evaluation.routing_analysis import (  # noqa: E402
    compute_metrics,
    create_plots,
    select_case_studies,
    write_json,
)
from evaluation.specialization import (  # noqa: E402
    analyze_specialization,
    representative_examples,
    specialization_observations,
)
from training.trainer import build_model, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 all-expert counterfactual routing analysis"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--validation-batches",
        type=int,
        help="Optional scope override for a smaller correctness/performance run",
    )
    return parser.parse_args()


def _decode_context(tokens: list[int], position: int, radius: int) -> str:
    left = bytes(tokens[max(0, position - radius) : position]).decode(
        "utf-8", errors="replace"
    )
    current = bytes([tokens[position]]).decode("utf-8", errors="replace")
    right = bytes(tokens[position + 1 : position + radius + 1]).decode(
        "utf-8", errors="replace"
    )
    return f"{left}⟦{current}⟧{right}"


def _rows_from_batch(
    oracle: Any,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    layer_index: int,
    first_sequence_id: int,
    temperature: float,
    context_radius: int,
) -> list[dict[str, Any]]:
    probabilities = oracle.router_probabilities.cpu()
    top2 = oracle.router_top2.cpu()
    losses = oracle.expert_losses.cpu()
    baseline_losses = oracle.baseline_top2_losses.cpu()
    soft_oracle = torch.softmax(-losses / temperature, dim=-1)
    sorted_losses, sorted_experts = torch.sort(losses, dim=-1)
    oracle_experts = sorted_experts[:, 0]
    gaps = sorted_losses[:, 1] - sorted_losses[:, 0]
    top1 = probabilities.argmax(dim=-1)
    row_indices = torch.arange(losses.size(0))
    top1_losses = losses[row_indices, top1]
    top1_regret = (top1_losses - sorted_losses[:, 0]).clamp_min(0.0)
    mixture_delta = baseline_losses - sorted_losses[:, 0]
    router_entropy = -torch.sum(
        probabilities * torch.log(probabilities.clamp_min(1e-12)), dim=-1
    )
    oracle_entropy = -torch.sum(
        soft_oracle * torch.log(soft_oracle.clamp_min(1e-12)), dim=-1
    )

    input_cpu = input_ids.cpu().tolist()
    target_cpu = targets.cpu().tolist()
    sequence_length = input_ids.size(1)
    rows: list[dict[str, Any]] = []
    for flat_index in range(losses.size(0)):
        sequence_offset = flat_index // sequence_length
        position = flat_index % sequence_length
        sequence_tokens = input_cpu[sequence_offset]
        rows.append(
            {
                "layer": layer_index,
                "sequence_id": first_sequence_id + sequence_offset,
                "position": position,
                "input_byte": sequence_tokens[position],
                "target_byte": target_cpu[sequence_offset][position],
                "preceding_byte": (
                    sequence_tokens[position - 1] if position > 0 else -1
                ),
                "following_byte": target_cpu[sequence_offset][position],
                "context": _decode_context(
                    sequence_tokens, position, context_radius
                ),
                "router_probabilities": probabilities[flat_index].tolist(),
                "router_top1": int(top1[flat_index]),
                "router_top2": top2[flat_index].tolist(),
                "expert_losses": losses[flat_index].tolist(),
                "oracle_expert": int(oracle_experts[flat_index]),
                "soft_oracle_distribution": soft_oracle[flat_index].tolist(),
                "loss_gap": float(gaps[flat_index]),
                "router_confidence": float(probabilities[flat_index, top1[flat_index]]),
                "router_entropy": float(router_entropy[flat_index]),
                "oracle_entropy": float(oracle_entropy[flat_index]),
                "top1_regret": float(top1_regret[flat_index]),
                "baseline_top2_loss": float(baseline_losses[flat_index]),
                "top2_mixture_delta": float(mixture_delta[flat_index]),
            }
        )
    return rows


def _report_markdown(
    summary: dict[str, Any],
    per_layer: dict[str, Any],
    strong_wins: dict[str, Any],
    observations: list[str],
) -> str:
    aggregate = per_layer["aggregate"]
    gap = aggregate["best_vs_second_gap"]
    regret = aggregate["top1_regret"]
    utilization = summary["oracle_expert_utilization"]
    most_common = int(np.argmax(utilization)) + 1
    observation_lines = (
        "\n".join(f"- {item}" for item in observations)
        if observations
        else "- No byte-category enrichment exceeded the conservative reporting rule."
    )
    return f"""# Stage 2 oracle-routing report

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

- Checkpoint: `{summary['checkpoint']}`
- Unique validation target bytes: {summary['evaluated_tokens']:,}
- Token-layer decisions: {summary['evaluated_token_layer_decisions']:,}
- Oracle temperature: {summary['oracle_temperature']:.3g}
- Wall time: {summary['oracle_evaluation_seconds']:.2f} seconds
- Throughput: {summary['token_layer_decisions_per_second']:.1f} token-layer decisions/s
- Peak allocated/reserved CUDA memory: {summary['peak_gpu_memory_mb']:.1f} / {summary['peak_gpu_memory_reserved_mb']:.1f} MiB

## A. Do experts meaningfully differ?

The median best-vs-second loss gap is {gap['median']:.6f}; P90 is
{gap['p90']:.6f}, P95 is {gap['p95']:.6f}, and P99 is {gap['p99']:.6f}.
{gap['threshold_fractions']['less_than_0.01']:.1%} of choices are below 0.01,
while {gap['threshold_fractions']['greater_than_0.1']:.1%} exceed 0.1.

The strong-win rule is `{strong_wins['threshold_method']}` at a gap of
{strong_wins['threshold']:.6f}; it identifies {strong_wins['fraction']:.1%} of
token-layer decisions.

**Finding:** yes, within this intervention definition. Only
{gap['threshold_fractions']['less_than_0.01']:.1%} are near-ties below 0.01,
and the long gap tail is large enough that expert identity is a meaningful
selection variable rather than a cosmetic distinction.

## B. Does the router select the best expert?

- Top-1 oracle accuracy: {aggregate['router_top1_oracle_accuracy']:.2%}
- Top-2 oracle coverage: {aggregate['router_top2_oracle_coverage']:.2%}
- Most common oracle expert: Expert {most_common} ({utilization[most_common - 1]:.2%})
- Strong-win Top-1 accuracy: {strong_wins['router_top1_oracle_accuracy']:.2%}
- Strong-win Top-2 coverage: {strong_wins['router_top2_oracle_coverage']:.2%}

The router is substantially better than uniform random selection (25% Top-1,
50% Top-2), especially on strong wins, but it still misses the best forced
single expert on {1 - aggregate['router_top1_oracle_accuracy']:.1%} of all
decisions and outside its Top-2 on
{1 - aggregate['router_top2_oracle_coverage']:.1%}.

## C. How expensive are routing mistakes?

Forced Top-1 mean regret is {regret['mean']:.6f}; median {regret['median']:.6f};
P95 {regret['p95']:.6f}; P99 {regret['p99']:.6f}; maximum
{regret['maximum']:.6f}. Strong-win mean regret is
{strong_wins['mean_top1_regret']:.6f}.

The actual Top-2 mixture minus best forced-single mean is
{aggregate['top2_mixture_minus_best_single']['mean']:.6f}. This signed result
must not be interpreted as a pure router regret because it compares a mixture
against a single expert.
The Top-2 mixture beats every forced single expert on
{aggregate['fraction_top2_mixture_beats_best_single']:.1%} of decisions, which
is direct evidence that Top-2 cannot simply be treated as a worse K=1 policy.

## D. Is router confidence meaningful?

Mean router entropy is {aggregate['mean_router_entropy']:.4f} nats versus
{aggregate['mean_oracle_entropy']:.4f} nats for the temperature-scaled soft
oracle. The exploratory binned confidence error is
{aggregate['exploratory_confidence_ece']:.4f}. This is exploratory evidence,
not a definitive calibration measurement.

Router confidence has only a
{aggregate['confidence_correctness_correlation']:.3f} correlation with oracle
correctness and a {aggregate['confidence_regret_correlation']:.3f} correlation
with regret. The reliability curve is notably non-monotonic in layer 3, so
confidence contains some signal but is not a dependable decision rule as-is.

Distribution alignment: KL(q || p)
{aggregate['distribution_alignment']['mean_kl_q_to_p']:.4f}, cross-entropy
{aggregate['distribution_alignment']['mean_cross_entropy']:.4f}, squared
probability distance {aggregate['distribution_alignment']['mean_brier_squared_distance']:.4f},
and cosine similarity {aggregate['distribution_alignment']['mean_cosine_similarity']:.4f}.

## E. Are experts showing specialization?

{observation_lines}

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
"""


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.validation_batches is not None:
        config["validation_batches"] = args.validation_batches
    set_seed(int(config["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = PROJECT_ROOT / config["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_config = checkpoint["config"]
    model = build_model(training_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    before_fingerprint = checkpoint_parameter_fingerprint(model)

    data_config = training_config["data"]
    paths = prepare_wikitext2(
        PROJECT_ROOT / data_config["cache_dir"],
        data_config.get("max_train_tokens"),
        data_config.get("max_validation_tokens"),
    )
    validation_dataset = TokenBlockDataset(
        paths["validation"], training_config["model"]["sequence_length"]
    )
    loader = DataLoader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    moe_layers = list(training_config["model"]["moe_layers"])
    temperature = float(config["oracle_temperature"])
    if temperature <= 0:
        raise ValueError("oracle_temperature must be positive")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = PROJECT_ROOT / config["experiment"]["output_dir"] / (
        f"{config['experiment']['name']}-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.resolved.json", config)

    use_amp = device.type == "cuda" and config["mixed_precision"] == "fp16"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    evaluated_sequences = 0
    max_batches = int(config["validation_batches"])
    for batch_index, (input_ids, targets) in enumerate(loader):
        if batch_index >= max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        for layer_index in moe_layers:
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                oracle = evaluate_layer_oracle(
                    model,
                    input_ids,
                    targets,
                    layer_index,
                    int(config["intervention_chunk_size"]),
                )
            rows.extend(
                _rows_from_batch(
                    oracle,
                    input_ids,
                    targets,
                    layer_index,
                    evaluated_sequences,
                    temperature,
                    int(config["context_radius"]),
                )
            )
        evaluated_sequences += input_ids.size(0)
        if (batch_index + 1) == 1 or (batch_index + 1) % 5 == 0:
            elapsed = time.perf_counter() - start_time
            print(
                json.dumps(
                    {
                        "validation_batches": batch_index + 1,
                        "unique_tokens": evaluated_sequences * input_ids.size(1),
                        "token_layer_decisions": len(rows),
                        "elapsed_seconds": elapsed,
                    }
                ),
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    after_fingerprint = checkpoint_parameter_fingerprint(model)
    if before_fingerprint != after_fingerprint:
        raise RuntimeError("Model parameters changed during observational analysis")

    frame = pd.DataFrame(rows)
    frame.to_parquet(output_dir / "token_results.parquet", index=False)
    num_experts = int(training_config["model"]["num_experts"])
    per_layer, confidence, utilization, strong_wins = compute_metrics(
        frame, list(config["confidence_bins"]), num_experts
    )
    cases = select_case_studies(frame)
    specialization = analyze_specialization(
        frame, num_experts, int(training_config["model"]["sequence_length"])
    )
    observations = specialization_observations(specialization)
    examples = representative_examples(
        frame, num_experts, int(config["examples_per_expert"])
    )
    aggregate_utilization = utilization["aggregate"]["oracle_best"]
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "model_parameters": model.parameter_count(),
        "evaluated_tokens": int(evaluated_sequences * training_config["model"]["sequence_length"]),
        "evaluated_token_layer_decisions": int(len(frame)),
        "validation_batches": int(config["validation_batches"]),
        "oracle_temperature": temperature,
        "oracle_evaluation_seconds": elapsed,
        "token_layer_decisions_per_second": len(frame) / elapsed,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
        ),
        "peak_gpu_memory_reserved_mb": (
            torch.cuda.max_memory_reserved() / (1024**2) if device.type == "cuda" else 0.0
        ),
        "router_top1_oracle_accuracy": per_layer["aggregate"]["router_top1_oracle_accuracy"],
        "router_top2_oracle_coverage": per_layer["aggregate"]["router_top2_oracle_coverage"],
        "mean_top1_regret": per_layer["aggregate"]["top1_regret"]["mean"],
        "p95_top1_regret": per_layer["aggregate"]["top1_regret"]["p95"],
        "median_best_vs_second_gap": per_layer["aggregate"]["best_vs_second_gap"]["median"],
        "p90_best_vs_second_gap": per_layer["aggregate"]["best_vs_second_gap"]["p90"],
        "mean_router_entropy": per_layer["aggregate"]["mean_router_entropy"],
        "mean_oracle_entropy": per_layer["aggregate"]["mean_oracle_entropy"],
        "confidence_correctness_correlation": per_layer["aggregate"]["confidence_correctness_correlation"],
        "confidence_regret_correlation": per_layer["aggregate"]["confidence_regret_correlation"],
        "fraction_top2_mixture_beats_best_single": per_layer["aggregate"]["fraction_top2_mixture_beats_best_single"],
        "oracle_expert_utilization": aggregate_utilization,
        "most_common_oracle_expert": int(np.argmax(aggregate_utilization)),
        "strong_win_threshold": strong_wins["threshold"],
        "strong_win_fraction": strong_wins["fraction"],
        "weights_unchanged": True,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "per_layer_metrics.json", per_layer)
    write_json(output_dir / "confidence_bins.json", confidence)
    write_json(output_dir / "expert_utilization.json", utilization)
    write_json(output_dir / "strong_wins.json", strong_wins)
    write_json(output_dir / "confident_mistakes.json", cases["confident_mistakes"])
    write_json(output_dir / "uncertain_but_correct.json", cases["uncertain_but_correct"])
    write_json(output_dir / "case_selection.json", cases["selection"])
    write_json(output_dir / "representative_examples.json", examples)
    write_json(output_dir / "specialization.json", specialization)
    create_plots(frame, output_dir, confidence, utilization, num_experts)
    report = _report_markdown(summary, per_layer, strong_wins, observations)
    (output_dir / "STAGE2_REPORT.md").write_text(report, encoding="utf-8")
    (PROJECT_ROOT / "STAGE2_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_directory": str(output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
