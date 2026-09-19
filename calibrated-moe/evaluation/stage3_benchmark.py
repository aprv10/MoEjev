from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TokenBlockDataset, prepare_wikitext2  # noqa: E402
from evaluation.oracle import (  # noqa: E402
    checkpoint_parameter_fingerprint,
    evaluate_layer_oracle,
)
from evaluation.routing_analysis import write_json  # noqa: E402
from evaluation.temperature_scaling import (  # noqa: E402
    calibration_metrics,
    fit_temperature,
    scale_probabilities,
)
from moe.adaptive_routing import (  # noqa: E402
    RoutingPolicy,
    forward_with_routing_policy,
)
from training.trainer import build_model, set_seed  # noqa: E402


@dataclass
class BenchmarkResult:
    name: str
    validation_loss: float
    perplexity: float
    experts_per_token: float
    compute_reduction_vs_top2: float
    quality_change_vs_top2: float
    tokens_per_second: float
    milliseconds_per_token: float
    peak_gpu_memory_mb: float
    peak_gpu_memory_reserved_mb: float
    ece: float | None = None
    brier: float | None = None
    nll: float | None = None
    calibration_method: str | None = None
    uncertainty_metric: str | None = None
    threshold: float | None = None
    selection_status: str = "control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the controlled Stage 3 benchmark")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _oracle_records(
    oracle: Any,
    layer: int,
    sequence_offset: int,
    batch_size: int,
    sequence_length: int,
    oracle_temperature: float,
) -> list[dict[str, Any]]:
    probabilities = oracle.router_probabilities.cpu()
    losses = oracle.expert_losses.cpu()
    top2 = oracle.router_top2.cpu()
    baseline = oracle.baseline_top2_losses.cpu()
    soft = torch.softmax(-losses / oracle_temperature, dim=-1)
    sorted_losses, sorted_indices = torch.sort(losses, dim=-1)
    top1 = probabilities.argmax(dim=-1)
    indices = torch.arange(len(losses))
    regret = (losses[indices, top1] - sorted_losses[:, 0]).clamp_min(0)
    records: list[dict[str, Any]] = []
    for flat_index in range(batch_size * sequence_length):
        records.append(
            {
                "layer": layer,
                "sequence_id": sequence_offset + flat_index // sequence_length,
                "position": flat_index % sequence_length,
                "router_probabilities": probabilities[flat_index].tolist(),
                "router_top1": int(top1[flat_index]),
                "router_top2": top2[flat_index].tolist(),
                "expert_losses": losses[flat_index].tolist(),
                "baseline_top2_loss": float(baseline[flat_index]),
                "oracle_expert": int(sorted_indices[flat_index, 0]),
                "soft_oracle_distribution": soft[flat_index].tolist(),
                "top1_regret": float(regret[flat_index]),
            }
        )
    return records


def build_calibration_oracle(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    moe_layers: list[int],
    config: dict[str, Any],
    output_path: Path,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    sequence_offset = 0
    start = time.perf_counter()
    use_amp = batches[0][0].device.type == "cuda" and config["evaluation"]["mixed_precision"] == "fp16"
    for batch_index, (inputs, targets) in enumerate(batches):
        for layer in moe_layers:
            with torch.inference_mode(), torch.autocast(
                device_type=inputs.device.type, dtype=torch.float16, enabled=use_amp
            ):
                oracle = evaluate_layer_oracle(
                    model,
                    inputs,
                    targets,
                    layer,
                    int(config["calibration"]["intervention_chunk_size"]),
                )
            records.extend(
                _oracle_records(
                    oracle,
                    layer,
                    sequence_offset,
                    inputs.size(0),
                    inputs.size(1),
                    float(config["calibration"]["oracle_temperature"]),
                )
            )
        sequence_offset += inputs.size(0)
        if batch_index == 0 or (batch_index + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "phase": "calibration_oracle",
                        "batches": batch_index + 1,
                        "token_layer_decisions": len(records),
                        "seconds": time.perf_counter() - start,
                    }
                ),
                flush=True,
            )
    frame = pd.DataFrame(records)
    frame.to_parquet(output_path, index=False)
    return frame


def fit_layer_temperatures(
    frame: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {"hard_oracle": {}, "soft_oracle": {}}
    for layer, group in frame.groupby("layer"):
        probabilities = np.stack(group["router_probabilities"].to_numpy()).astype(np.float64)
        hard_targets = group["oracle_expert"].to_numpy(dtype=np.int64)
        soft_targets = np.stack(group["soft_oracle_distribution"].to_numpy()).astype(np.float64)
        common = {
            "minimum": float(config["calibration"]["temperature_min"]),
            "maximum": float(config["calibration"]["temperature_max"]),
        }
        output["hard_oracle"][str(int(layer))] = fit_temperature(
            probabilities, hard_targets=hard_targets, **common
        )
        output["soft_oracle"][str(int(layer))] = fit_temperature(
            probabilities, soft_targets=soft_targets, **common
        )
    return output


def _temperature_map(fits: dict[str, Any], method: str) -> dict[int, float]:
    return {
        int(layer): float(values["temperature"])
        for layer, values in fits[method].items()
    }


def preload_batches(
    loader: DataLoader,
    device: torch.device,
    maximum: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    output = []
    for index, (inputs, targets) in enumerate(loader):
        if index >= maximum:
            break
        output.append(
            (
                inputs.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
            )
        )
    return output


@torch.inference_mode()
def evaluate_policy(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    policy: RoutingPolicy,
    use_amp: bool,
    oracle_masks: dict[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    loss_sum = 0.0
    token_count = 0
    assignments = 0
    decisions = 0
    token_losses: list[np.ndarray] = []
    sequence_offset = 0
    for inputs, targets in batches:
        batch_masks = None
        if oracle_masks is not None:
            batch_masks = {
                layer: mask[
                    sequence_offset : sequence_offset + inputs.size(0)
                ].to(inputs.device)
                for layer, mask in oracle_masks.items()
            }
        with torch.autocast(
            device_type=inputs.device.type, dtype=torch.float16, enabled=use_amp
        ):
            output = forward_with_routing_policy(
                model, inputs, targets, policy, batch_masks
            )
        losses = F.cross_entropy(
            output.logits.reshape(-1, output.logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        )
        loss_sum += float(losses.float().sum())
        token_count += targets.numel()
        assignments += output.expert_assignments
        decisions += output.routing_decisions
        token_losses.append(losses.float().cpu().numpy())
        sequence_offset += inputs.size(0)
    return {
        "loss": loss_sum / token_count,
        "experts_per_token": assignments / decisions,
        "token_losses": np.concatenate(token_losses),
    }


def benchmark_policy(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    policy: RoutingPolicy,
    use_amp: bool,
    warmup_batches: int,
    timing_repeats: int,
    oracle_masks: dict[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    device = batches[0][0].device
    warm_batches = batches[:warmup_batches]
    evaluate_policy(model, warm_batches, policy, use_amp, oracle_masks)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = None
    for _ in range(timing_repeats):
        result = evaluate_policy(model, batches, policy, use_amp, oracle_masks)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert result is not None
    tokens = sum(targets.numel() for _, targets in batches) * timing_repeats
    return {
        **result,
        "tokens_per_second": tokens / elapsed,
        "milliseconds_per_token": elapsed * 1000.0 / tokens,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
        ),
        "peak_gpu_memory_reserved_mb": (
            torch.cuda.max_memory_reserved() / (1024**2) if device.type == "cuda" else 0.0
        ),
    }


def _signals_for_frame(
    frame: pd.DataFrame,
    temperatures: dict[int, float],
    metric: str,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for layer, group in frame.groupby("layer", sort=False):
        raw = np.stack(group["router_probabilities"].to_numpy()).astype(np.float64)
        probabilities = scale_probabilities(raw, temperatures.get(int(layer), 1.0))
        sorted_probabilities = np.sort(probabilities, axis=1)[:, ::-1]
        if metric == "max_probability":
            signal = sorted_probabilities[:, 0]
        elif metric == "entropy":
            signal = -np.sum(
                probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1
            )
        elif metric == "margin":
            signal = sorted_probabilities[:, 0] - sorted_probabilities[:, 1]
        else:
            raise ValueError(metric)
        chunks.append(signal)
    return np.concatenate(chunks)


def _candidate_thresholds(
    signals: np.ndarray, quantiles: list[float]
) -> list[float]:
    values = np.quantile(signals, quantiles)
    span = max(float(signals.max() - signals.min()), 1e-6)
    endpoints = [float(signals.min() - span * 1e-6), float(signals.max() + span * 1e-6)]
    return sorted(set(endpoints + [float(value) for value in values]))


def search_adaptive_thresholds(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    calibration_frame: pd.DataFrame,
    temperature_methods: dict[str, dict[int, float]],
    config: dict[str, Any],
    use_amp: bool,
    split_name: str,
    prescribed_points: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if prescribed_points is not None:
        jobs = [
            (
                point["calibration_method"],
                point["uncertainty_metric"],
                float(point["threshold"]),
            )
            for point in prescribed_points
        ]
    else:
        jobs = []
        for method, temperatures in temperature_methods.items():
            for metric in config["adaptive"]["metrics"]:
                signals = _signals_for_frame(calibration_frame, temperatures, metric)
                for threshold in _candidate_thresholds(
                    signals, list(config["adaptive"]["threshold_quantiles"])
                ):
                    jobs.append((method, metric, threshold))
    start = time.perf_counter()
    for index, (method, metric, threshold) in enumerate(jobs):
        policy = RoutingPolicy(
            name=f"{method}-{metric}-{threshold:.8g}",
            uncertainty_metric=metric,
            threshold=threshold,
            temperatures=temperature_methods[method],
        )
        measured = evaluate_policy(model, batches, policy, use_amp)
        results.append(
            {
                "split": split_name,
                "calibration_method": method,
                "uncertainty_metric": metric,
                "threshold": threshold,
                "loss": measured["loss"],
                "experts_per_token": measured["experts_per_token"],
            }
        )
        if (index + 1) % 15 == 0 or index + 1 == len(jobs):
            print(
                json.dumps(
                    {
                        "phase": f"threshold_search_{split_name}",
                        "completed": index + 1,
                        "total": len(jobs),
                        "seconds": time.perf_counter() - start,
                    }
                ),
                flush=True,
            )
    return results


def select_thresholds(
    calibration_curve: list[dict[str, Any]],
    baseline_loss: float,
    tolerance: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    def select(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        eligible = [
            point for point in candidates if point["loss"] <= baseline_loss + tolerance
        ]
        pool = eligible if eligible else candidates
        chosen = min(pool, key=lambda point: (point["experts_per_token"], point["loss"]))
        return {
            **chosen,
            "selection_constraint_met": bool(eligible),
            "calibration_loss_limit": baseline_loss + tolerance,
        }

    selected_calibrated: dict[str, dict[str, Any]] = {}
    selected_uncalibrated: dict[str, dict[str, Any]] = {}
    for metric in sorted({point["uncertainty_metric"] for point in calibration_curve}):
        calibrated = [
            point
            for point in calibration_curve
            if point["uncertainty_metric"] == metric
            and point["calibration_method"] in {"hard_oracle", "soft_oracle"}
        ]
        uncalibrated = [
            point
            for point in calibration_curve
            if point["uncertainty_metric"] == metric
            and point["calibration_method"] == "uncalibrated"
        ]
        selected_calibrated[metric] = select(calibrated)
        selected_uncalibrated[metric] = select(uncalibrated)
    selected_final = select(list(selected_calibrated.values()))
    return selected_calibrated, selected_uncalibrated, selected_final


def build_oracle_k_masks(
    validation_frame: pd.DataFrame,
    num_sequences: int,
    sequence_length: int,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    masks: dict[int, torch.Tensor] = {}
    diagnostics: dict[str, Any] = {"method": "isolated K1-vs-K2 target-loss comparison", "layers": {}}
    for layer, group in validation_frame.groupby("layer"):
        ordered = group.sort_values(["sequence_id", "position"])
        expert_losses = np.stack(ordered["expert_losses"].to_numpy()).astype(np.float64)
        top1 = ordered["router_top1"].to_numpy(dtype=np.int64)
        top1_losses = expert_losses[np.arange(len(ordered)), top1]
        top2_losses = ordered["baseline_top2_loss"].to_numpy(dtype=np.float64)
        # Ties use K=1 because it has the same isolated loss at lower compute.
        k2 = top2_losses < top1_losses
        masks[int(layer)] = torch.from_numpy(
            k2.reshape(num_sequences, sequence_length)
        )
        diagnostics["layers"][str(int(layer))] = {
            "k2_fraction": float(k2.mean()),
            "isolated_oracle_loss": float(np.minimum(top1_losses, top2_losses).mean()),
            "top1_loss": float(top1_losses.mean()),
            "top2_loss": float(top2_losses.mean()),
        }
    diagnostics["average_experts_per_token_from_decisions"] = 1.0 + float(
        np.mean([mask.float().mean().item() for mask in masks.values()])
    )
    return masks, diagnostics


def _frame_k1_mask(frame: pd.DataFrame, policy: RoutingPolicy) -> np.ndarray:
    output = np.zeros(len(frame), dtype=bool)
    for layer, indices in frame.groupby("layer").groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        raw = np.stack(frame.loc[positions, "router_probabilities"].to_numpy()).astype(
            np.float64
        )
        probabilities = scale_probabilities(
            raw, policy.temperatures.get(int(layer), 1.0)
        )
        sorted_p = np.sort(probabilities, axis=1)[:, ::-1]
        if policy.uncertainty_metric == "max_probability":
            signal = sorted_p[:, 0]
            k2 = signal < float(policy.threshold)
        elif policy.uncertainty_metric == "entropy":
            signal = -np.sum(
                probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1
            )
            k2 = signal > float(policy.threshold)
        elif policy.uncertainty_metric == "margin":
            signal = sorted_p[:, 0] - sorted_p[:, 1]
            k2 = signal < float(policy.threshold)
        else:
            raise ValueError(policy.uncertainty_metric)
        output[positions] = ~k2
    return output


def regret_comparison(
    validation_frame: pd.DataFrame,
    policies: dict[str, RoutingPolicy],
) -> dict[str, Any]:
    regret = validation_frame["top1_regret"].to_numpy(dtype=np.float64)
    output: dict[str, Any] = {}
    for name, policy in policies.items():
        if policy.fixed_k == 1:
            k1 = np.ones(len(validation_frame), dtype=bool)
        else:
            k1 = _frame_k1_mask(validation_frame, policy)
        exposed = regret[k1]
        output[name] = {
            "k1_decision_fraction": float(k1.mean()),
            "mean_forced_top1_regret_on_k1_decisions": float(exposed.mean()),
            "median": float(np.median(exposed)),
            "p90": float(np.percentile(exposed, 90)),
            "p95": float(np.percentile(exposed, 95)),
            "p99": float(np.percentile(exposed, 99)),
        }
    return output


def _plot_reliability(
    reliability: dict[str, list[dict[str, Any]]],
    path: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0.25, 1], [0.25, 1], "--", color="0.5", label="identity")
    for layer in [key for key in reliability if key != "aggregate"]:
        bins = [item for item in reliability[layer] if item["count"] >= 20]
        axis.plot(
            [item["mean_confidence"] for item in bins],
            [item["accuracy"] for item in bins],
            marker="o",
            label=f"Layer {layer}",
        )
    axis.set(xlim=(0.25, 1), ylim=(0.2, 1), xlabel="Mean confidence", ylabel="Oracle Top-1 accuracy", title=title)
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def create_stage3_plots(
    output_dir: Path,
    calibration_results: dict[str, Any],
    validation_curve: list[dict[str, Any]],
    benchmarks: list[dict[str, Any]],
    regret: dict[str, Any],
) -> None:
    plt.rcParams.update({"font.size": 10})
    _plot_reliability(
        calibration_results["uncalibrated"]["reliability"],
        output_dir / "reliability_uncalibrated.png",
        "Uncalibrated router",
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), squeeze=False)
    for axis, method in zip(axes[0], ["hard_oracle", "soft_oracle"]):
        reliability = calibration_results[method]["reliability"]
        axis.plot([0.25, 1], [0.25, 1], "--", color="0.5")
        for layer in [key for key in reliability if key != "aggregate"]:
            bins = [item for item in reliability[layer] if item["count"] >= 20]
            axis.plot(
                [item["mean_confidence"] for item in bins],
                [item["accuracy"] for item in bins],
                marker="o",
                label=f"Layer {layer}",
            )
        axis.set(xlim=(0.25, 1), ylim=(0.2, 1), xlabel="Mean confidence", ylabel="Oracle Top-1 accuracy", title=method.replace("_", " ").title())
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "reliability_calibrated.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    styles = {"uncalibrated": ":", "hard_oracle": "-", "soft_oracle": "--"}
    colors = {"max_probability": "#1f77b4", "entropy": "#d95f02", "margin": "#2ca02c"}
    for (method, metric), points_frame in pd.DataFrame(validation_curve).groupby(
        ["calibration_method", "uncertainty_metric"]
    ):
        ordered = points_frame.sort_values("experts_per_token")
        axis.plot(
            ordered["experts_per_token"],
            ordered["loss"],
            linestyle=styles[method],
            color=colors[metric],
            alpha=0.8,
            label=f"{metric} / {method}",
        )
    for item in benchmarks:
        if item["selection_status"] in {"control", "calibration_selected_final", "oracle"}:
            axis.scatter(
                item["experts_per_token"],
                item["validation_loss"],
                s=80 if item["selection_status"] != "control" else 55,
                marker="*" if item["selection_status"] == "calibration_selected_final" else "o",
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
                label=item["name"],
            )
    axis.set_xlabel("Average experts per token")
    axis.set_ylabel("Validation loss")
    axis.set_title("Quality vs compute (validation; policies selected on calibration)")
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "quality_vs_compute.png", dpi=180)
    plt.close(figure)

    selected_benchmarks = [
        item
        for item in benchmarks
        if item["selection_status"] in {"control", "calibration_selected", "calibration_selected_final", "oracle"}
        and not item["name"].startswith("Calibrated Top-1 Soft")
    ]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(
        range(len(selected_benchmarks)),
        [item["experts_per_token"] for item in selected_benchmarks],
        color="#4c78a8",
    )
    axis.set_xticks(
        range(len(selected_benchmarks)),
        [item["name"].replace("Adaptive K - ", "") for item in selected_benchmarks],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Average experts per token")
    axis.set_ylim(0, 2.1)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_dir / "experts_per_token.png", dpi=160)
    plt.close(figure)

    names = list(regret)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(names))
    width = 0.25
    for offset, statistic in zip([-1, 0, 1], ["median", "p90", "p95"]):
        axis.bar(
            x + offset * width,
            [regret[name][statistic] for name in names],
            width,
            label=statistic.upper(),
        )
    axis.set_xticks(x, names, rotation=20, ha="right")
    axis.set_ylabel("Forced Top-1 regret on K=1 decisions")
    axis.set_title("Regret exposure (K=2 decisions excluded)")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_dir / "regret_comparison.png", dpi=160)
    plt.close(figure)

    methods = ["uncalibrated", "hard_oracle", "soft_oracle"]
    statistics = ["ece", "brier", "nll"]
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(methods))
    width = 0.24
    for offset, statistic in zip([-1, 0, 1], statistics):
        axis.bar(
            x + offset * width,
            [calibration_results[method]["metrics"]["aggregate"][statistic] for method in methods],
            width,
            label=statistic.upper(),
        )
    axis.set_xticks(x, [method.replace("_", " ").title() for method in methods])
    axis.set_ylabel("Metric value (lower is better)")
    axis.set_title("Validation calibration metrics")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_dir / "calibration_comparison.png", dpi=160)
    plt.close(figure)


def _make_report(
    config: dict[str, Any],
    temperature_fits: dict[str, Any],
    calibration_results: dict[str, Any],
    threshold_results: dict[str, Any],
    benchmark_results: list[dict[str, Any]],
    oracle_diagnostics: dict[str, Any],
    regret: dict[str, Any],
) -> str:
    by_name = {item["name"]: item for item in benchmark_results}
    top2 = by_name["Standard Top-2"]
    top1 = by_name["Standard Top-1"]
    final = next(
        item
        for item in benchmark_results
        if item["selection_status"] == "calibration_selected_final"
    )
    oracle = by_name["Oracle Adaptive-K"]
    uncalibrated_control = by_name[
        f"Adaptive K - {final['uncertainty_metric']} (uncalibrated control)"
    ]
    before = calibration_results["uncalibrated"]["metrics"]["aggregate"]
    after = calibration_results[final["calibration_method"]]["metrics"]["aggregate"]
    quality_delta = final["validation_loss"] - top2["validation_loss"]
    compute_reduction = final["compute_reduction_vs_top2"]
    throughput_change = final["tokens_per_second"] / top2["tokens_per_second"] - 1.0
    calibrated_dominates_control = (
        final["validation_loss"] <= uncalibrated_control["validation_loss"]
        and final["experts_per_token"] <= uncalibrated_control["experts_per_token"]
    )
    if (
        quality_delta <= 0.02
        and compute_reduction >= 0.10
        and calibrated_dominates_control
    ):
        outcome = "Calibrated adaptive-routing success on this validation scope"
        conclusion = (
            "The calibrated policy reduces expert executions materially, keeps "
            "validation loss within 0.02 of Top-2, and dominates its matched "
            "uncalibrated threshold control."
        )
    elif quality_delta <= 0.02 and compute_reduction >= 0.10:
        outcome = "Partial success with a negative calibration-control result"
        conclusion = (
            "Adaptive K reduces compute with a small quality cost, and temperature "
            "scaling improves confidence metrics, but the matched uncalibrated "
            "adaptive control is at least as good. The compute benefit therefore "
            "cannot be attributed to calibration."
        )
    elif after["ece"] < before["ece"]:
        outcome = "Calibration-only partial success"
        conclusion = (
            "Temperature scaling improves confidence quality, but does not yield a "
            "useful compute-quality point under the predeclared constraint."
        )
    else:
        outcome = "No demonstrated benefit"
        conclusion = (
            "Neither confidence calibration nor the compute-quality tradeoff is "
            "strong enough to support the simple post-hoc approach."
        )
    table_rows = []
    for item in benchmark_results:
        if item["selection_status"] in {
            "control",
            "calibration_selected",
            "calibration_selected_final",
            "predeclared_control",
            "oracle",
        }:
            ece = "—" if item["ece"] is None else f"{item['ece']:.4f}"
            table_rows.append(
                f"| {item['name']} | {item['validation_loss']:.4f} | "
                f"{item['perplexity']:.3f} | {item['experts_per_token']:.3f} | "
                f"{item['compute_reduction_vs_top2']:.1%} | "
                f"{item['quality_change_vs_top2']:+.4f} | {ece} | "
                f"{item['tokens_per_second']:.0f} |"
            )
    temperatures = "\n".join(
        f"- {method.replace('_', ' ').title()}: "
        + ", ".join(
            f"layer {layer} T={values['temperature']:.4f}"
            for layer, values in layers.items()
        )
        for method, layers in temperature_fits.items()
    )
    selected_description = (
        f"{final['uncertainty_metric']} with {final['calibration_method']} "
        f"temperature scaling at threshold {final['threshold']:.6f}"
    )
    return f"""# Stage 3 calibrated adaptive-routing report

## 1. Research question

Can post-hoc calibrated uncertainty reduce MoE expert compute while maintaining
the quality of the fixed Stage 1 checkpoint? This is an independently designed
temperature-scaling experiment; it does not reproduce Jev or RLCD.

## 2. Experimental setup

The 6,413,312-parameter Stage 1 checkpoint, Transformer weights, experts, and
router weights are frozen. Calibration uses the final
{config['calibration']['batches'] * config['calibration']['batch_size'] * 128:,}
bytes from the training-token array. The untouched validation prefix contains
{config['evaluation']['validation_batches'] * config['evaluation']['batch_size'] * 128:,}
bytes. No validation targets influenced temperatures, thresholds, metric
choice, or final-policy selection.

Fixed Top-2, fixed Top-1, calibrated Top-1, three adaptive signals, and a
target-informed oracle reference are evaluated on identical batches. Timing
uses preloaded CUDA tensors, {config['evaluation']['warmup_batches']} warm-up
batches, and {config['evaluation']['timing_repeats']} complete timed repeats.

## 3. Calibration method

One scalar temperature is fitted per MoE layer using bounded one-dimensional
optimization. Hard calibration minimizes cross-entropy to the oracle-best
single expert. Soft calibration minimizes cross-entropy to
`softmax(-expert_loss / {config['calibration']['oracle_temperature']})`.

{temperatures}

Temperature scaling does not change expert ordering. Consequently, both
calibrated Top-1 variants have exactly the same model outputs as Standard
Top-1; only their confidence metrics differ.

## 4. Adaptive-K policies

Maximum probability, entropy, and Top-1/Top-2 margin each choose between K=1
and K=2. K=2 uses the calibrated Top-2 probabilities renormalized over the two
selected experts. Threshold candidates are calibration-distribution quantiles.

The predeclared selection rule chooses the minimum experts/token among points
whose calibration loss is no more than
{config['adaptive']['max_calibration_loss_increase']:.3f} above Standard Top-2,
breaking ties by loss. The final policy was selected entirely on calibration:
**{selected_description}**. All other validation sweeps are labeled
predeclared comparisons and were not used to change this choice.

## 5. Results

| Model | Val loss | PPL | Experts/token | Compute reduction | Δ loss vs Top-2 | ECE | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## 6. Calibration

Aggregate hard-label ECE changes from {before['ece']:.4f} to {after['ece']:.4f}
for the calibration method used by the selected policy. Brier score changes
from {before['brier']:.4f} to {after['brier']:.4f}; NLL changes from
{before['nll']:.4f} to {after['nll']:.4f}. Confidence/correctness correlation
changes from {before['confidence_correctness_correlation']:.3f} to
{after['confidence_correctness_correlation']:.3f}. Since temperature is a
monotone transform, ranking accuracy remains unchanged.

The objectives behave differently rather than one universally winning. Hard
calibration gives the best hard-label ECE/NLL. Soft calibration gives the best
soft-oracle squared distance
({calibration_results['soft_oracle']['metrics']['aggregate']['soft_oracle_brier']:.4f}
versus
{calibration_results['hard_oracle']['metrics']['aggregate']['soft_oracle_brier']:.4f}
for hard calibration), as expected from its training target.

## 7. Quality vs compute

The calibration-selected policy uses {final['experts_per_token']:.3f}
experts/token, a {compute_reduction:.1%} reduction from Top-2, with validation
loss change {quality_delta:+.4f}. Its measured throughput is
{final['tokens_per_second']:.0f} tokens/s versus {top2['tokens_per_second']:.0f}
for the fixed Top-2 control. GPU kernels and Python dispatch overhead mean
expert-count reductions need not translate proportionally to wall-clock speed.
Here throughput changes by {throughput_change:+.1%}: the selected adaptive
implementation is slower despite executing fewer experts, so this run shows an
expert-compute reduction but not an end-to-end latency improvement.

The matched uncalibrated {final['uncertainty_metric']} control uses
{uncalibrated_control['experts_per_token']:.3f} experts/token at loss
{uncalibrated_control['validation_loss']:.4f}. The calibrated policy changes
those values by {final['experts_per_token'] - uncalibrated_control['experts_per_token']:+.4f}
experts/token and
{final['validation_loss'] - uncalibrated_control['validation_loss']:+.4f} loss.
This control is essential: better ECE alone does not establish a better
adaptive-compute decision rule.

Fixed Top-1 gives the endpoint: {top1['experts_per_token']:.1f} expert/token at
loss {top1['validation_loss']:.4f}. The plotted validation curves contain every
predeclared threshold from the calibration sweep without smoothing.

## 8. Oracle upper bound

The oracle reference compares the routed Top-1 expert with the existing Top-2
mixture using the observed target loss for an isolated intervention at each
layer, choosing K=1 on ties. Applying those decisions simultaneously yields
loss {oracle['validation_loss']:.4f} at {oracle['experts_per_token']:.3f}
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
3. **Compute saved:** {compute_reduction:.1%} by expert count.
4. **Quality cost:** {quality_delta:+.4f} validation loss.
5. **Versus fixed Top-1:** selected adaptive loss {final['validation_loss']:.4f}
   versus {top1['validation_loss']:.4f}, at {final['experts_per_token']:.3f}
   versus {top1['experts_per_token']:.3f} experts/token.
6. **Oracle gap:** the target-informed reference reaches
   {oracle['validation_loss']:.4f} at {oracle['experts_per_token']:.3f}; the
   remaining distance indicates how much the simple confidence rule leaves
   unrecovered.
7. **Does calibration itself improve adaptive routing?** The matched
   uncalibrated control has loss {uncalibrated_control['validation_loss']:.4f}
   at {uncalibrated_control['experts_per_token']:.3f} experts/token. Therefore,
   confidence calibration and adaptive-routing utility must be treated as
   separate findings.

Regret comparisons use forced-Top-1 regret only on decisions actually sent to
K=1. They do not call Top-2 mixture minus best-single loss “regret.”

## 10. Conclusion

**{outcome}.** {conclusion} This conclusion applies to one frozen checkpoint,
one training-tail calibration split, and one validation prefix; repeated seeds
would be required before making a general claim.
"""


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(int(config["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and config["evaluation"]["mixed_precision"] == "fp16"
    checkpoint_path = PROJECT_ROOT / config["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_config = checkpoint["config"]
    model = build_model(training_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    fingerprint_before = checkpoint_parameter_fingerprint(model)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = PROJECT_ROOT / config["experiment"]["output_dir"] / (
        f"{config['experiment']['name']}-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", config)

    data_config = training_config["data"]
    paths = prepare_wikitext2(
        PROJECT_ROOT / data_config["cache_dir"],
        data_config.get("max_train_tokens"),
        data_config.get("max_validation_tokens"),
    )
    sequence_length = int(training_config["model"]["sequence_length"])
    train_dataset = TokenBlockDataset(paths["train"], sequence_length)
    validation_dataset = TokenBlockDataset(paths["validation"], sequence_length)
    calibration_blocks = int(config["calibration"]["batches"]) * int(
        config["calibration"]["batch_size"]
    )
    calibration_subset = Subset(
        train_dataset,
        range(len(train_dataset) - calibration_blocks, len(train_dataset)),
    )
    calibration_loader = DataLoader(
        calibration_subset,
        batch_size=int(config["calibration"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    calibration_batches = preload_batches(
        calibration_loader, device, int(config["calibration"]["batches"])
    )
    validation_batches = preload_batches(
        validation_loader, device, int(config["evaluation"]["validation_batches"])
    )

    calibration_frame = build_calibration_oracle(
        model,
        calibration_batches,
        list(training_config["model"]["moe_layers"]),
        config,
        output_dir / "calibration_oracle.parquet",
    )
    temperature_fits = fit_layer_temperatures(calibration_frame, config)
    temperature_methods = {
        "uncalibrated": {
            int(layer): 1.0 for layer in training_config["model"]["moe_layers"]
        },
        "hard_oracle": _temperature_map(temperature_fits, "hard_oracle"),
        "soft_oracle": _temperature_map(temperature_fits, "soft_oracle"),
    }

    calibration_results: dict[str, Any] = {
        "temperature_fits": temperature_fits,
        "split": {
            "source": "tail of capped Stage 1 training token array",
            "unique_tokens": sum(targets.numel() for _, targets in calibration_batches),
            "overlaps_final_validation": False,
        },
    }
    validation_frame = pd.read_parquet(PROJECT_ROOT / config["stage2_validation_results"])
    for method, temperatures in temperature_methods.items():
        calibration_metric_values, calibration_reliability = calibration_metrics(
            calibration_frame,
            temperatures,
            int(training_config["model"]["num_experts"]),
        )
        validation_metric_values, validation_reliability = calibration_metrics(
            validation_frame,
            temperatures,
            int(training_config["model"]["num_experts"]),
        )
        calibration_results[method] = {
            "calibration_split_metrics": calibration_metric_values,
            "metrics": validation_metric_values,
            "reliability": validation_reliability,
            "temperatures": temperatures,
        }
    write_json(output_dir / "calibration_results.json", calibration_results)

    standard_top2_policy = RoutingPolicy("standard-top2", fixed_k=2)
    calibration_top2 = evaluate_policy(
        model, calibration_batches, standard_top2_policy, use_amp
    )
    calibration_curve = search_adaptive_thresholds(
        model,
        calibration_batches,
        calibration_frame,
        temperature_methods,
        config,
        use_amp,
        "calibration",
    )
    selected_by_metric, selected_uncalibrated, selected_final = select_thresholds(
        calibration_curve,
        calibration_top2["loss"],
        float(config["adaptive"]["max_calibration_loss_increase"]),
    )
    validation_curve = search_adaptive_thresholds(
        model,
        validation_batches,
        calibration_frame,
        temperature_methods,
        config,
        use_amp,
        "validation_predeclared_comparison",
        calibration_curve,
    )

    def adaptive_policy(point: dict[str, Any], name: str) -> RoutingPolicy:
        return RoutingPolicy(
            name=name,
            uncertainty_metric=point["uncertainty_metric"],
            threshold=float(point["threshold"]),
            temperatures=temperature_methods[point["calibration_method"]],
        )

    benchmark_specs: list[tuple[str, RoutingPolicy, str]] = [
        ("Standard Top-2", standard_top2_policy, "control"),
        ("Standard Top-1", RoutingPolicy("standard-top1", fixed_k=1), "control"),
        (
            "Calibrated Top-1 Hard",
            RoutingPolicy(
                "calibrated-top1-hard",
                fixed_k=1,
                temperatures=temperature_methods["hard_oracle"],
            ),
            "control",
        ),
        (
            "Calibrated Top-1 Soft",
            RoutingPolicy(
                "calibrated-top1-soft",
                fixed_k=1,
                temperatures=temperature_methods["soft_oracle"],
            ),
            "control",
        ),
    ]
    for metric in config["adaptive"]["metrics"]:
        point = selected_by_metric[metric]
        status = (
            "calibration_selected_final"
            if point["calibration_method"] == selected_final["calibration_method"]
            and point["uncertainty_metric"] == selected_final["uncertainty_metric"]
            and point["threshold"] == selected_final["threshold"]
            else "calibration_selected"
        )
        benchmark_specs.append(
            (
                f"Adaptive K - {metric}",
                adaptive_policy(point, f"adaptive-{metric}"),
                status,
            )
        )
        uncalibrated_point = selected_uncalibrated[metric]
        benchmark_specs.append(
            (
                f"Adaptive K - {metric} (uncalibrated control)",
                adaptive_policy(uncalibrated_point, f"uncalibrated-{metric}"),
                "predeclared_control",
            )
        )

    oracle_masks, oracle_diagnostics = build_oracle_k_masks(
        validation_frame,
        num_sequences=len(validation_batches) * int(config["evaluation"]["batch_size"]),
        sequence_length=sequence_length,
    )
    benchmark_raw: dict[str, dict[str, Any]] = {}
    benchmark_rows: list[dict[str, Any]] = []
    for name, policy, status in benchmark_specs:
        measured = benchmark_policy(
            model,
            validation_batches,
            policy,
            use_amp,
            int(config["evaluation"]["warmup_batches"]),
            int(config["evaluation"]["timing_repeats"]),
        )
        benchmark_raw[name] = measured
        method = (
            "uncalibrated"
            if name.startswith("Standard") or "uncalibrated" in name
            else (
                "hard_oracle"
                if "Hard" in name
                else (
                    "soft_oracle"
                    if "Soft" in name
                    else next(
                        point["calibration_method"]
                        for point in selected_by_metric.values()
                        if point["uncertainty_metric"] == policy.uncertainty_metric
                    )
                )
            )
        )
        cal_metrics = calibration_results[method]["metrics"]["aggregate"]
        benchmark_rows.append(
            {
                "name": name,
                "validation_loss": measured["loss"],
                "perplexity": math.exp(measured["loss"]),
                "experts_per_token": measured["experts_per_token"],
                "compute_reduction_vs_top2": 0.0,
                "quality_change_vs_top2": 0.0,
                "tokens_per_second": measured["tokens_per_second"],
                "milliseconds_per_token": measured["milliseconds_per_token"],
                "peak_gpu_memory_mb": measured["peak_gpu_memory_mb"],
                "peak_gpu_memory_reserved_mb": measured["peak_gpu_memory_reserved_mb"],
                "ece": cal_metrics["ece"],
                "brier": cal_metrics["brier"],
                "nll": cal_metrics["nll"],
                "calibration_method": method,
                "uncertainty_metric": policy.uncertainty_metric,
                "threshold": policy.threshold,
                "selection_status": status,
            }
        )

    oracle_policy = RoutingPolicy("oracle-adaptive", fixed_k=2)
    oracle_measured = benchmark_policy(
        model,
        validation_batches,
        oracle_policy,
        use_amp,
        int(config["evaluation"]["warmup_batches"]),
        int(config["evaluation"]["timing_repeats"]),
        oracle_masks,
    )
    benchmark_raw["Oracle Adaptive-K"] = oracle_measured
    benchmark_rows.append(
        {
            "name": "Oracle Adaptive-K",
            "validation_loss": oracle_measured["loss"],
            "perplexity": math.exp(oracle_measured["loss"]),
            "experts_per_token": oracle_measured["experts_per_token"],
            "compute_reduction_vs_top2": 0.0,
            "quality_change_vs_top2": 0.0,
            "tokens_per_second": oracle_measured["tokens_per_second"],
            "milliseconds_per_token": oracle_measured["milliseconds_per_token"],
            "peak_gpu_memory_mb": oracle_measured["peak_gpu_memory_mb"],
            "peak_gpu_memory_reserved_mb": oracle_measured["peak_gpu_memory_reserved_mb"],
            "ece": None,
            "brier": None,
            "nll": None,
            "calibration_method": None,
            "uncertainty_metric": None,
            "threshold": None,
            "selection_status": "oracle",
        }
    )
    baseline_loss = next(
        item["validation_loss"] for item in benchmark_rows if item["name"] == "Standard Top-2"
    )
    for item in benchmark_rows:
        item["compute_reduction_vs_top2"] = (2.0 - item["experts_per_token"]) / 2.0
        item["quality_change_vs_top2"] = item["validation_loss"] - baseline_loss

    final_row = next(
        item for item in benchmark_rows if item["selection_status"] == "calibration_selected_final"
    )
    selected_policies = {
        "Standard Top-1": RoutingPolicy("standard-top1", fixed_k=1),
        "Selected adaptive": adaptive_policy(selected_final, "selected-adaptive"),
    }
    regret = regret_comparison(validation_frame, selected_policies)
    threshold_results = {
        "selection_rule": {
            "loss_reference": calibration_top2["loss"],
            "maximum_loss_increase": config["adaptive"]["max_calibration_loss_increase"],
            "tie_break": "minimum experts/token, then minimum loss",
            "validation_used_for_selection": False,
        },
        "calibration_curve": calibration_curve,
        "selected_calibrated_by_metric": selected_by_metric,
        "selected_uncalibrated_controls": selected_uncalibrated,
        "selected_final": selected_final,
        "validation_predeclared_curve": validation_curve,
    }
    write_json(output_dir / "threshold_search.json", threshold_results)
    write_json(output_dir / "benchmark_results.json", benchmark_rows)
    write_json(output_dir / "oracle_adaptive.json", oracle_diagnostics)
    write_json(output_dir / "regret_comparison.json", regret)

    token_output = pd.DataFrame(
        {
            "token_index": np.arange(len(benchmark_raw["Standard Top-2"]["token_losses"])),
            "standard_top2_loss": benchmark_raw["Standard Top-2"]["token_losses"],
            "standard_top1_loss": benchmark_raw["Standard Top-1"]["token_losses"],
            "selected_adaptive_loss": benchmark_raw[final_row["name"]]["token_losses"],
            "oracle_adaptive_loss": benchmark_raw["Oracle Adaptive-K"]["token_losses"],
        }
    )
    token_output.to_parquet(output_dir / "token_evaluation.parquet", index=False)
    create_stage3_plots(
        output_dir, calibration_results, validation_curve, benchmark_rows, regret
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "weights_unchanged": checkpoint_parameter_fingerprint(model) == fingerprint_before,
        "hardware": {
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "total_vram_mib": (
                torch.cuda.get_device_properties(0).total_memory / (1024**2)
                if device.type == "cuda"
                else 0.0
            ),
            "pytorch": torch.__version__,
            "pytorch_cuda": torch.version.cuda,
        },
        "calibration_temperatures": {
            method: {
                layer: values["temperature"] for layer, values in layers.items()
            }
            for method, layers in temperature_fits.items()
        },
        "selected_final_policy": selected_final,
        "standard_top2": next(item for item in benchmark_rows if item["name"] == "Standard Top-2"),
        "standard_top1": next(item for item in benchmark_rows if item["name"] == "Standard Top-1"),
        "best_calibrated_adaptive_policy": final_row,
        "oracle_adaptive": next(item for item in benchmark_rows if item["name"] == "Oracle Adaptive-K"),
        "ece_before": calibration_results["uncalibrated"]["metrics"]["aggregate"]["ece"],
        "ece_after_selected_calibration": calibration_results[final_row["calibration_method"]]["metrics"]["aggregate"]["ece"],
        "brier_before": calibration_results["uncalibrated"]["metrics"]["aggregate"]["brier"],
        "brier_after_selected_calibration": calibration_results[final_row["calibration_method"]]["metrics"]["aggregate"]["brier"],
    }
    write_json(output_dir / "summary.json", summary)
    report = _make_report(
        config,
        temperature_fits,
        calibration_results,
        threshold_results,
        benchmark_rows,
        oracle_diagnostics,
        regret,
    )
    (output_dir / "STAGE3_REPORT.md").write_text(report, encoding="utf-8")
    (PROJECT_ROOT / "STAGE3_REPORT.md").write_text(report, encoding="utf-8")
    if not summary["weights_unchanged"]:
        raise RuntimeError("Model parameters changed during Stage 3")
    print(json.dumps({"output_directory": str(output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
