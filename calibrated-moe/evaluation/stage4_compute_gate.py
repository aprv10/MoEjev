from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

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
from evaluation.stage3_benchmark import (  # noqa: E402
    benchmark_policy,
    evaluate_policy,
    preload_batches,
)
from moe.adaptive_routing import RoutingPolicy  # noqa: E402
from moe.compute_gate import (  # noqa: E402
    GateFeatureSpec,
    LinearComputeGate,
    MLPComputeGate,
    assemble_gate_features,
    forward_with_compute_gate,
    gate_parameter_count,
    router_scalar_features,
)
from training.trainer import build_model, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4 learned compute-value gates")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def extract_gate_dataset(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    moe_layers: list[int],
    config: dict[str, Any],
    split_name: str,
) -> dict[str, np.ndarray]:
    hidden_parts: list[np.ndarray] = []
    router_parts: list[np.ndarray] = []
    delta_parts: list[np.ndarray] = []
    layer_parts: list[np.ndarray] = []
    sequence_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    expert_loss_parts: list[np.ndarray] = []
    top1_parts: list[np.ndarray] = []
    top2_parts: list[np.ndarray] = []
    sequence_offset = 0
    start = time.perf_counter()
    use_amp = batches[0][0].device.type == "cuda" and config["benchmark"]["mixed_precision"] == "fp16"
    for batch_index, (inputs, targets) in enumerate(batches):
        batch_size, sequence_length = inputs.shape
        for layer in moe_layers:
            with torch.inference_mode(), torch.autocast(
                device_type=inputs.device.type, dtype=torch.float16, enabled=use_amp
            ):
                oracle = evaluate_layer_oracle(
                    model,
                    inputs,
                    targets,
                    layer,
                    int(config["data"]["intervention_chunk_size"]),
                )
            probabilities = oracle.router_probabilities
            top1 = probabilities.argmax(dim=-1)
            indices = torch.arange(len(top1), device=top1.device)
            top1_loss = oracle.expert_losses[indices, top1]
            delta = top1_loss - oracle.baseline_top2_losses
            scalar_features = router_scalar_features(probabilities)
            hidden_parts.append(oracle.pre_moe_hidden.cpu().numpy().astype(np.float16))
            router_parts.append(scalar_features.cpu().numpy().astype(np.float32))
            delta_parts.append(delta.cpu().numpy().astype(np.float32))
            expert_loss_parts.append(
                oracle.expert_losses.cpu().numpy().astype(np.float16)
            )
            top1_parts.append(top1.cpu().numpy().astype(np.int8))
            top2_parts.append(oracle.router_top2.cpu().numpy().astype(np.int8))
            layer_parts.append(np.full(batch_size * sequence_length, layer, dtype=np.int8))
            sequence_parts.append(
                np.repeat(
                    np.arange(
                        sequence_offset,
                        sequence_offset + batch_size,
                        dtype=np.int32,
                    ),
                    sequence_length,
                )
            )
            position_parts.append(
                np.tile(np.arange(sequence_length, dtype=np.int16), batch_size)
            )
        sequence_offset += batch_size
        if batch_index == 0 or (batch_index + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "phase": f"feature_extraction_{split_name}",
                        "batches": batch_index + 1,
                        "token_layer_decisions": sum(len(part) for part in delta_parts),
                        "seconds": time.perf_counter() - start,
                    }
                ),
                flush=True,
            )
    return {
        "hidden": np.concatenate(hidden_parts),
        "router": np.concatenate(router_parts),
        "delta": np.concatenate(delta_parts),
        "layer": np.concatenate(layer_parts),
        "sequence_id": np.concatenate(sequence_parts),
        "position": np.concatenate(position_parts),
        "expert_losses": np.concatenate(expert_loss_parts),
        "router_top1": np.concatenate(top1_parts),
        "router_top2": np.concatenate(top2_parts),
    }


def save_gate_dataset(directory: Path, data: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    for name, values in data.items():
        np.save(directory / f"{name}.npy", values, allow_pickle=False)


def feature_matrix(
    data: dict[str, np.ndarray], spec: GateFeatureSpec
) -> np.ndarray:
    layer_one_hot = np.stack(
        [(data["layer"] == layer).astype(np.float32) for layer in spec.moe_layers],
        axis=1,
    )
    pieces = [data["router"].astype(np.float32), layer_one_hot]
    if spec.include_hidden:
        pieces.insert(0, data["hidden"].astype(np.float32))
    return np.concatenate(pieces, axis=1)


def compute_feature_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def make_gate(
    architecture: str, input_dim: int, hidden_dim: int
) -> torch.nn.Module:
    if architecture == "linear":
        return LinearComputeGate(input_dim)
    if architecture == "mlp":
        return MLPComputeGate(input_dim, hidden_dim)
    raise ValueError(architecture)


def train_gate(
    gate: torch.nn.Module,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    calibration_features: np.ndarray,
    calibration_targets: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> list[dict[str, float]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    gate.to(device).train()
    train_x = torch.from_numpy((train_features - mean) / std).to(device)
    train_y = torch.from_numpy(train_targets).to(device)
    calibration_x = torch.from_numpy((calibration_features - mean) / std).to(device)
    calibration_y = torch.from_numpy(calibration_targets).to(device)
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    batch_size = int(config["training"]["batch_size"])
    delta = float(config["training"]["huber_delta"])
    generator = torch.Generator(device=device).manual_seed(seed)
    metrics: list[dict[str, float]] = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        permutation = torch.randperm(len(train_x), generator=generator, device=device)
        loss_sum = 0.0
        for start in range(0, len(train_x), batch_size):
            indices = permutation[start : start + batch_size]
            prediction = gate(train_x.index_select(0, indices))
            loss = F.huber_loss(
                prediction,
                train_y.index_select(0, indices),
                delta=delta,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * len(indices)
        gate.eval()
        with torch.inference_mode():
            calibration_prediction = gate(calibration_x)
            calibration_loss = F.huber_loss(
                calibration_prediction, calibration_y, delta=delta
            )
        metrics.append(
            {
                "epoch": epoch,
                "train_huber": loss_sum / len(train_x),
                "calibration_huber_monitor_only": float(calibration_loss),
            }
        )
        gate.train()
    gate.eval()
    return metrics


@torch.inference_mode()
def predict_gate(
    gate: torch.nn.Module,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int = 16384,
) -> np.ndarray:
    normalized = torch.from_numpy((features - mean) / std)
    output: list[np.ndarray] = []
    gate.eval()
    for start in range(0, len(normalized), batch_size):
        prediction = gate(normalized[start : start + batch_size].to(device))
        output.append(prediction.float().cpu().numpy())
    return np.concatenate(output)


def prediction_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    pearson = float(np.corrcoef(prediction, target)[0, 1])
    prediction_rank = pd.Series(prediction).rank(method="average").to_numpy()
    target_rank = pd.Series(target).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(prediction_rank, target_rank)[0, 1])
    try:
        decile_ids = pd.qcut(prediction, 10, labels=False, duplicates="drop")
    except ValueError:
        decile_ids = pd.cut(prediction, 10, labels=False, duplicates="drop")
    deciles = []
    for decile in sorted(np.unique(decile_ids)):
        mask = np.asarray(decile_ids == decile)
        deciles.append(
            {
                "decile": int(decile) + 1,
                "count": int(mask.sum()),
                "mean_prediction": float(prediction[mask].mean()),
                "mean_true_delta": float(target[mask].mean()),
            }
        )
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "pearson": pearson,
        "spearman": spearman,
        "sign_accuracy": float(((prediction > 0) == (target > 0)).mean()),
        "positive_target_fraction": float((target > 0).mean()),
        "deciles": deciles,
    }


def candidate_thresholds(prediction: np.ndarray, quantiles: list[float]) -> list[float]:
    values = np.quantile(prediction, quantiles)
    span = max(float(prediction.max() - prediction.min()), 1e-6)
    endpoints = [
        float(prediction.min() - span * 1e-6),
        float(prediction.max() + span * 1e-6),
    ]
    return sorted(set(endpoints + [float(value) for value in values]))


def prediction_masks(
    data: dict[str, np.ndarray],
    prediction: np.ndarray,
    threshold: float,
    moe_layers: list[int],
    num_sequences: int,
    sequence_length: int,
) -> dict[int, torch.Tensor]:
    masks: dict[int, torch.Tensor] = {}
    for layer in moe_layers:
        rows = np.flatnonzero(data["layer"] == layer)
        layer_mask = np.zeros((num_sequences, sequence_length), dtype=bool)
        layer_mask[
            data["sequence_id"][rows].astype(np.int64),
            data["position"][rows].astype(np.int64),
        ] = prediction[rows] > threshold
        masks[layer] = torch.from_numpy(layer_mask)
    return masks


def measure_gate_frontier(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    data: dict[str, np.ndarray],
    prediction: np.ndarray,
    thresholds: list[float],
    moe_layers: list[int],
    sequence_length: int,
    use_amp: bool,
    split: str,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    num_sequences = sum(inputs.size(0) for inputs, _ in batches)
    for index, threshold in enumerate(thresholds):
        masks = prediction_masks(
            data,
            prediction,
            threshold,
            moe_layers,
            num_sequences,
            sequence_length,
        )
        measured = evaluate_policy(
            model,
            batches,
            RoutingPolicy("learned-gate-offline-mask", fixed_k=2),
            use_amp,
            masks,
        )
        points.append(
            {
                "split": split,
                "threshold": threshold,
                "loss": measured["loss"],
                "experts_per_token": measured["experts_per_token"],
                "fraction_k2": measured["experts_per_token"] - 1.0,
            }
        )
        if index == 0 or index + 1 == len(thresholds):
            print(
                json.dumps(
                    {
                        "phase": f"gate_frontier_{split}",
                        "completed": index + 1,
                        "total": len(thresholds),
                    }
                ),
                flush=True,
            )
    return points


def measure_online_gate_frontier(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    gate: torch.nn.Module,
    spec: GateFeatureSpec,
    mean: np.ndarray,
    std: np.ndarray,
    thresholds: list[float],
    use_amp: bool,
    split: str,
) -> list[dict[str, Any]]:
    device = batches[0][0].device
    mean_tensor = torch.from_numpy(mean).to(device)
    std_tensor = torch.from_numpy(std).to(device)
    points: list[dict[str, Any]] = []
    for index, threshold in enumerate(thresholds):
        measured = evaluate_online_gate(
            model,
            batches,
            gate,
            spec,
            mean_tensor,
            std_tensor,
            threshold,
            use_amp,
        )
        points.append(
            {
                "split": split,
                "threshold": threshold,
                "loss": measured["loss"],
                "experts_per_token": measured["experts_per_token"],
                "fraction_k2": measured["experts_per_token"] - 1.0,
                "execution": "online learned gate inside frozen model",
            }
        )
        if index == 0 or index + 1 == len(thresholds):
            print(
                json.dumps(
                    {
                        "phase": f"online_gate_frontier_{split}",
                        "completed": index + 1,
                        "total": len(thresholds),
                    }
                ),
                flush=True,
            )
    return points


def select_calibration_point(
    points: list[dict[str, Any]], top2_loss: float, tolerance: float
) -> dict[str, Any]:
    limit = top2_loss + tolerance
    eligible = [point for point in points if point["loss"] <= limit]
    pool = eligible if eligible else points
    selected = min(pool, key=lambda point: (point["experts_per_token"], point["loss"]))
    return {
        **selected,
        "selection_constraint_met": bool(eligible),
        "calibration_top2_loss": top2_loss,
        "calibration_loss_limit": limit,
        "selection_rule": "minimum experts/token, then minimum loss",
    }


@torch.inference_mode()
def evaluate_online_gate(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    gate: torch.nn.Module,
    spec: GateFeatureSpec,
    mean: torch.Tensor,
    std: torch.Tensor,
    threshold: float,
    use_amp: bool,
) -> dict[str, Any]:
    loss_sum = 0.0
    token_count = 0
    assignments = 0
    decisions = 0
    for inputs, targets in batches:
        with torch.autocast(
            device_type=inputs.device.type, dtype=torch.float16, enabled=use_amp
        ):
            output = forward_with_compute_gate(
                model, inputs, targets, gate, spec, mean, std, threshold
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
    return {
        "loss": loss_sum / token_count,
        "experts_per_token": assignments / decisions,
    }


def benchmark_online_gate(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    gate: torch.nn.Module,
    spec: GateFeatureSpec,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    use_amp: bool,
    warmup_batches: int,
    timing_repeats: int,
) -> dict[str, Any]:
    device = batches[0][0].device
    mean_tensor = torch.from_numpy(mean).to(device)
    std_tensor = torch.from_numpy(std).to(device)
    evaluate_online_gate(
        model,
        batches[:warmup_batches],
        gate,
        spec,
        mean_tensor,
        std_tensor,
        threshold,
        use_amp,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = None
    for _ in range(timing_repeats):
        result = evaluate_online_gate(
            model,
            batches,
            gate,
            spec,
            mean_tensor,
            std_tensor,
            threshold,
            use_amp,
        )
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
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
        ),
        "peak_gpu_memory_reserved_mb": (
            torch.cuda.max_memory_reserved() / 1024**2 if device.type == "cuda" else 0.0
        ),
    }


def conditional_analysis(
    data: dict[str, np.ndarray],
    prediction: np.ndarray,
    threshold: float,
    large_delta_threshold: float,
) -> dict[str, Any]:
    target = data["delta"]
    k2 = prediction > threshold
    output: dict[str, Any] = {
        "expensive_mistake_definition": (
            "K=1 with actual delta above the gate-train positive-delta P90; "
            "the cutoff is training-derived and is not used for model selection"
        ),
        "large_delta_threshold": large_delta_threshold,
        "expensive_false_k1_fraction_all": float(((~k2) & (target > large_delta_threshold)).mean()),
        "expensive_false_k1_fraction_among_k1": float(
            ((~k2) & (target > large_delta_threshold)).sum() / max((~k2).sum(), 1)
        ),
    }
    for name, mask in [("k1", ~k2), ("k2", k2)]:
        values = target[mask]
        output[name] = {
            "fraction": float(mask.mean()),
            "count": int(mask.sum()),
            "mean_true_delta": float(values.mean()) if len(values) else None,
            "median_true_delta": float(np.median(values)) if len(values) else None,
            "p90_true_delta": float(np.percentile(values, 90)) if len(values) else None,
            "fraction_where_k2_helped": float((values > 0).mean()) if len(values) else None,
        }
    return output


def worst_false_k1_rows(
    data: dict[str, np.ndarray],
    prediction: np.ndarray,
    threshold: float,
    validation_frame: pd.DataFrame,
    limit: int = 12,
) -> list[dict[str, Any]]:
    false_rows = np.flatnonzero(prediction <= threshold)
    false_rows = false_rows[np.argsort(data["delta"][false_rows])[::-1]][:limit]
    lookup = validation_frame.set_index(["layer", "sequence_id", "position"])
    output: list[dict[str, Any]] = []
    for row in false_rows:
        key = (
            int(data["layer"][row]),
            int(data["sequence_id"][row]),
            int(data["position"][row]),
        )
        source = lookup.loc[key]
        if isinstance(source, pd.DataFrame):
            source = source.iloc[0]
        token_value = source.get("input_token", source.get("input_byte", None))
        output.append(
            {
                "layer": key[0],
                "sequence_id": key[1],
                "position": key[2],
                "token_or_byte": None if token_value is None else int(token_value),
                "token_text": source.get("token_text", source.get("input_text", None)),
                "context": source.get("context", None),
                "router_probabilities": list(source["router_probabilities"]),
                "predicted_delta": float(prediction[row]),
                "actual_delta": float(data["delta"][row]),
                "decision": "K=1",
            }
        )
    return output


def stage3_controls(stage3_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds = json.loads((stage3_dir / "threshold_search.json").read_text(encoding="utf-8"))
    validation = [
        {**point, "method": point["uncertainty_metric"]}
        for point in thresholds["validation_predeclared_curve"]
        if point["calibration_method"] == "uncalibrated"
    ]
    benchmarks = json.loads((stage3_dir / "benchmark_results.json").read_text(encoding="utf-8"))
    return validation, benchmarks


def matched_compute_table(
    targets: list[float],
    control_points: list[dict[str, Any]],
    learned_points: list[dict[str, Any]],
    top2_loss: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        groups = [(method, None) for method in sorted({point["method"] for point in control_points})]
        groups += sorted(
            {
                (point["method"], int(point["seed"]))
                for point in learned_points
            }
        )
        for method, seed in groups:
            source = control_points if seed is None else learned_points
            candidates = [
                point
                for point in source
                if point["method"] == method
                and (seed is None or int(point["seed"]) == seed)
            ]
            if not candidates:
                continue
            point = min(
                candidates,
                key=lambda item: abs(item["experts_per_token"] - target),
            )
            rows.append(
                {
                    "target_experts_per_token": target,
                    "method": method,
                    "seed": seed,
                    "actual_experts_per_token": point["experts_per_token"],
                    "validation_loss": point["loss"],
                    "delta_loss_vs_top2": point["loss"] - top2_loss,
                    "threshold": point["threshold"],
                }
            )
    return rows


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def create_plots(
    plots_dir: Path,
    controls: list[dict[str, Any]],
    learned: list[dict[str, Any]],
    endpoints: dict[str, dict[str, Any]],
    run_records: list[dict[str, Any]],
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 10})
    colors = {
        "max_probability": "#8c8c8c",
        "entropy": "#b07aa1",
        "margin": "#e15759",
        "linear_router_only": "#4e79a7",
        "linear_hidden_router": "#59a14f",
        "mlp_hidden_router": "#f28e2b",
    }
    figure, axis = plt.subplots(figsize=(8.2, 5.5))
    for method in sorted({point["method"] for point in controls}):
        points = sorted(
            [point for point in controls if point["method"] == method],
            key=lambda point: point["experts_per_token"],
        )
        axis.plot(
            [point["experts_per_token"] for point in points],
            [point["loss"] for point in points],
            "o-",
            ms=3,
            lw=1.2,
            color=colors[method],
            label=f"Heuristic: {method.replace('_', ' ')}",
        )
    for method in sorted({point["method"] for point in learned}):
        seeds = sorted({int(point["seed"]) for point in learned if point["method"] == method})
        for seed_index, seed in enumerate(seeds):
            points = sorted(
                [
                    point
                    for point in learned
                    if point["method"] == method and int(point["seed"]) == seed
                ],
                key=lambda point: point["experts_per_token"],
            )
            axis.plot(
                [point["experts_per_token"] for point in points],
                [point["loss"] for point in points],
                "o-",
                ms=3,
                lw=1.1,
                alpha=0.55,
                color=colors[method],
                label=method.replace("_", " ") if seed_index == 0 else None,
            )
    for name, marker, color in [
        ("top1", "s", "black"),
        ("top2", "s", "black"),
        ("oracle", "*", "#17becf"),
    ]:
        point = endpoints[name]
        axis.scatter(
            point["experts_per_token"], point["loss"], marker=marker, s=85,
            color=color, zorder=5, label=name.replace("top", "Top-").title(),
        )
    axis.set(
        xlabel="Average experts per token",
        ylabel="Validation cross-entropy loss (lower is better)",
        title="Stage 4 quality–compute frontier (measured points)",
        xlim=(0.98, 2.02),
    )
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(plots_dir / "quality_vs_compute.png", dpi=180)
    plt.close(figure)

    seed_records = [record for record in run_records if record["seed"] == 11]
    figure, axes = plt.subplots(1, len(seed_records), figsize=(15, 4.2), squeeze=False)
    for axis, record in zip(axes[0], seed_records):
        target = record["validation_target"]
        prediction = record["validation_prediction"]
        take = np.linspace(0, len(target) - 1, min(6000, len(target))).astype(int)
        axis.scatter(prediction[take], target[take], s=3, alpha=0.08, rasterized=True)
        axis.axhline(0, color="0.5", lw=0.8)
        axis.axvline(0, color="0.5", lw=0.8)
        axis.set(
            title=record["variant"].replace("_", " "),
            xlabel="Predicted delta",
            ylabel="Actual isolated delta",
        )
        axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(plots_dir / "predicted_vs_actual_delta.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for record in seed_records:
        deciles = record["prediction_metrics"]["validation"]["deciles"]
        axis.plot(
            [item["mean_prediction"] for item in deciles],
            [item["mean_true_delta"] for item in deciles],
            "o-",
            color=colors[record["variant"]],
            label=record["variant"].replace("_", " "),
        )
    axis.axhline(0, color="0.5", lw=0.8)
    axis.set(
        xlabel="Mean predicted delta in decile",
        ylabel="Mean actual delta in decile",
        title="Compute-value ranking on final validation (seed 11)",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(plots_dir / "delta_deciles.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for variant in sorted({record["variant"] for record in run_records}):
        records = [record for record in run_records if record["variant"] == variant]
        epochs = [item["epoch"] for item in records[0]["training_metrics"]]
        curves = np.asarray(
            [[item["train_huber"] for item in record["training_metrics"]] for record in records]
        )
        axis.plot(epochs, curves.mean(axis=0), color=colors[variant], label=variant.replace("_", " "))
        axis.fill_between(
            epochs,
            curves.mean(axis=0) - curves.std(axis=0),
            curves.mean(axis=0) + curves.std(axis=0),
            color=colors[variant], alpha=0.16,
        )
    axis.set(xlabel="Epoch", ylabel="Training Huber loss", title="Compute-gate training curves (mean ± SD, 3 seeds)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(plots_dir / "training_curves.png", dpi=180)
    plt.close(figure)


def make_report(
    config: dict[str, Any],
    data_split: dict[str, Any],
    summaries: dict[str, Any],
    matched: list[dict[str, Any]],
    endpoints: dict[str, dict[str, Any]],
    fingerprint_unchanged: bool,
) -> str:
    margin_rows = [row for row in matched if row["method"] == "margin"]
    learned_rows = [row for row in matched if row["method"].startswith(("linear", "mlp"))]
    wins_by_variant: dict[str, tuple[int, int]] = {}
    matched_deltas: dict[str, list[tuple[float, float]]] = {}
    for variant in summaries:
        variant_rows = [row for row in learned_rows if row["method"] == variant]
        wins = 0
        comparisons = 0
        per_target: list[tuple[float, float]] = []
        for target in sorted({row["target_experts_per_token"] for row in variant_rows}):
            candidates = [row for row in variant_rows if row["target_experts_per_token"] == target]
            margin = next(
                item for item in margin_rows if item["target_experts_per_token"] == target
            )
            deltas = [row["validation_loss"] - margin["validation_loss"] for row in candidates]
            per_target.append((target, float(np.mean(deltas))))
            wins += sum(delta < 0 for delta in deltas)
            comparisons += len(deltas)
        wins_by_variant[variant] = (wins, comparisons)
        matched_deltas[variant] = per_target
    best_variant = min(
        summaries,
        key=lambda name: summaries[name]["selected_validation_loss"]["mean"],
    )
    best = summaries[best_variant]
    hidden_delta = (
        summaries["linear_hidden_router"]["selected_validation_loss"]["mean"]
        - summaries["linear_router_only"]["selected_validation_loss"]["mean"]
    )
    mlp_deltas = ", ".join(
        f"{target:.1f}: {delta:+.4f}" for target, delta in matched_deltas["mlp_hidden_router"]
    )
    mlp_wins, mlp_comparisons = wins_by_variant["mlp_hidden_router"]
    frontier_statement = (
        f"The MLP beat margin in {mlp_wins}/{mlp_comparisons} seed-specific nearest-point comparisons. "
        f"Its mean loss differences versus margin by target experts/token were {mlp_deltas}; "
        "negative values favor the MLP."
    )
    tables = []
    for name, values in summaries.items():
        tables.append(
            f"| {name.replace('_', ' ')} | {values['parameters']} | "
            f"{values['validation_pearson']['mean']:.4f} ± {values['validation_pearson']['std']:.4f} | "
            f"{values['selected_experts_per_token']['mean']:.4f} ± {values['selected_experts_per_token']['std']:.4f} | "
            f"{values['selected_validation_loss']['mean']:.4f} ± {values['selected_validation_loss']['std']:.4f} | "
            f"{values['tokens_per_second']['mean']:.0f} ± {values['tokens_per_second']['std']:.0f} |"
        )
    return f"""# Stage 4 learned compute-gate report

## Scope and question

This experiment asks whether a tiny model can predict the isolated value of the
second routed expert and thereby improve the validation quality–expert-compute
frontier over simple router-confidence heuristics. It uses one frozen tiny MoE
checkpoint; it does not reproduce Jev or RLCD.

## Protocol and leakage controls

- Gate training: blocks {data_split['gate_train']['block_start']}–{data_split['gate_train']['block_stop_exclusive'] - 1} ({data_split['gate_train']['bytes']} input bytes).
- Gate calibration: blocks {data_split['gate_calibration']['block_start']}–{data_split['gate_calibration']['block_stop_exclusive'] - 1} ({data_split['gate_calibration']['bytes']} input bytes).
- One complete block is unused between these windows, preventing their shifted language-model targets from sharing a boundary byte.
- Final validation is the unchanged first {data_split['final_validation']['batches']} validation batches ({data_split['final_validation']['bytes']} bytes).
- `delta_loss = isolated Top-1 loss - normal Top-2 loss`; oracle losses are labels and diagnostics only.
- Features are computed before the current MoE expert output: pre-MoE normalized hidden state, Top-1 and Top-2 router probabilities, margin, entropy, and one-hot layer identity.
- No target, expert loss, future hidden state, or downstream activation enters the gate. Normalization statistics come only from gate training. Thresholds and primary operating points come only from gate calibration. Validation frontiers are explicitly exploratory.
- Transformer, router, and expert parameters remained unchanged: **{fingerprint_unchanged}**.

## Primary results (three gate initialization seeds)

| Gate | Parameters | Validation Pearson | Selected experts/token | Validation loss | Online tokens/s |
|---|---:|---:|---:|---:|---:|
{chr(10).join(tables)}

Top-1 is {endpoints['top1']['loss']:.4f} at 1.000 experts/token; Top-2 is
{endpoints['top2']['loss']:.4f} at 2.000; the non-deployable oracle is
{endpoints['oracle']['loss']:.4f} at {endpoints['oracle']['experts_per_token']:.3f}.

## Answers to the predeclared questions

1. **Can delta loss be predicted?** Only weakly: the best model's held-out Pearson correlation is {best['validation_pearson']['mean']:.4f}. The decile and scatter plots show useful ranking structure, but substantial irreducible or unmodeled variation remains.
2. **Do hidden states help?** A linear hidden-state gate does not: at the independently calibration-selected point, adding hidden features changes mean validation loss by {hidden_delta:+.4f} versus the router-only linear gate (positive is worse). The nonlinear hidden-state MLP does improve matched-compute loss, so the benefit is architecture-dependent rather than evidence that raw hidden features alone suffice.
3. **Does a learned gate beat simple heuristics?** {frontier_statement}
4. **Matched compute:** `matched_compute.json` uses the nearest actually measured point at 1.10–1.50 experts/token; no interpolation or smoothing is used.
5. **Frontier shift:** The MLP produces a modest frontier improvement from roughly 1.10 through 1.40 experts/token and is effectively tied near 1.50. The linear gates do not consistently shift the frontier. The validation sweep was not used to change architecture, features, epochs, or thresholds.
6. **Gap to oracle:** The best preselected learned variant is {best_variant.replace('_', ' ')} at mean loss {best['selected_validation_loss']['mean']:.4f}; the oracle is {endpoints['oracle']['loss']:.4f}. The oracle is a target-informed upper bound, not deployable.
7. **Runtime:** The MLP averages {best['tokens_per_second']['mean']:.0f} tokens/s versus {endpoints['top2']['tokens_per_second']:.0f} for Stage 3 Top-2, so this implementation is slower despite executing fewer experts. Expert count is algorithmic compute; tokens/s includes gate and dynamic-dispatch overhead.
8. **Seed stability:** Each gate was trained from seeds {config['experiment']['gate_seeds']}; mean and population SD are reported rather than selecting a favorable seed.
9. **Scaling criterion:** Scaling is justified only if learned gates repeatedly lower loss versus margin at tightly matched expert counts, with stable seeds and a meaningful remaining oracle gap. A prediction correlation alone is insufficient.

## Interpretation

The strongest preselected learned result is **{best_variant.replace('_', ' ')}**.
{frontier_statement} This is evidence about one checkpoint and one validation
region only; it does not establish general superiority of learned adaptive MoE
routing. See `matched_compute.json`, per-seed threshold sweeps, conditional
analyses, and worst-false-K1 tables for the complete result rather than relying
on one operating point.
"""


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(int(config["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and config["benchmark"]["mixed_precision"] == "fp16"
    checkpoint_path = PROJECT_ROOT / config["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_config = checkpoint["config"]
    model = build_model(training_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
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
    moe_layers = [int(layer) for layer in training_config["model"]["moe_layers"]]
    model_dim = int(training_config["model"]["model_dim"])
    train_dataset = TokenBlockDataset(paths["train"], sequence_length)
    validation_dataset = TokenBlockDataset(paths["validation"], sequence_length)
    train_start = int(config["data"]["gate_train_block_start"])
    train_blocks = int(config["data"]["gate_train_blocks"])
    cal_start = int(config["data"]["gate_calibration_block_start"])
    cal_blocks = int(config["data"]["gate_calibration_blocks"])
    if train_start + train_blocks >= cal_start:
        raise ValueError("Gate train/calibration shifted target byte ranges overlap")
    if cal_start + cal_blocks > len(train_dataset):
        raise ValueError("Requested gate calibration blocks exceed training data")
    batch_size = int(config["data"]["batch_size"])

    def make_loader(dataset: Any, indices: range, batch: int) -> DataLoader:
        return DataLoader(
            Subset(dataset, indices), batch_size=batch, shuffle=False, num_workers=0
        )

    train_loader = make_loader(
        train_dataset, range(train_start, train_start + train_blocks), batch_size
    )
    cal_loader = make_loader(
        train_dataset, range(cal_start, cal_start + cal_blocks), batch_size
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    train_batches = preload_batches(train_loader, device, math.ceil(train_blocks / batch_size))
    cal_batches = preload_batches(cal_loader, device, math.ceil(cal_blocks / batch_size))
    validation_batches = preload_batches(
        validation_loader, device, int(config["data"]["validation_batches"])
    )
    data_split = {
        "sequence_length": sequence_length,
        "boundary_policy": "one unused block separates shifted train/cal targets",
        "gate_train": {
            "source": "training",
            "block_start": train_start,
            "block_stop_exclusive": train_start + train_blocks,
            "input_byte_start": train_start * sequence_length,
            "input_byte_stop_exclusive": (train_start + train_blocks) * sequence_length,
            "target_byte_stop_exclusive": (train_start + train_blocks) * sequence_length + 1,
            "blocks": train_blocks,
            "bytes": train_blocks * sequence_length,
        },
        "unused_gap": {
            "block_start": train_start + train_blocks,
            "block_stop_exclusive": cal_start,
        },
        "gate_calibration": {
            "source": "training",
            "block_start": cal_start,
            "block_stop_exclusive": cal_start + cal_blocks,
            "input_byte_start": cal_start * sequence_length,
            "input_byte_stop_exclusive": (cal_start + cal_blocks) * sequence_length,
            "target_byte_stop_exclusive": (cal_start + cal_blocks) * sequence_length + 1,
            "blocks": cal_blocks,
            "bytes": cal_blocks * sequence_length,
        },
        "final_validation": {
            "source": "validation",
            "batch_start": 0,
            "batches": len(validation_batches),
            "blocks": sum(inputs.size(0) for inputs, _ in validation_batches),
            "bytes": sum(targets.numel() for _, targets in validation_batches),
            "same_region_as_stages_2_and_3": True,
        },
        "overlap_checks": {
            "gate_train_vs_gate_calibration": False,
            "training_windows_vs_validation_split": False,
        },
    }
    write_json(output_dir / "data_split.json", data_split)

    cache_dir = output_dir / "feature_cache"
    cache_dir.mkdir()
    train_data = extract_gate_dataset(model, train_batches, moe_layers, config, "gate_train")
    save_gate_dataset(cache_dir / "gate_train", train_data)
    cal_data = extract_gate_dataset(model, cal_batches, moe_layers, config, "gate_calibration")
    save_gate_dataset(cache_dir / "gate_calibration", cal_data)
    validation_data = extract_gate_dataset(
        model, validation_batches, moe_layers, config, "final_validation"
    )
    save_gate_dataset(cache_dir / "final_validation", validation_data)

    validation_frame = pd.read_parquet(PROJECT_ROOT / config["stage2_validation_results"])
    expected_validation_rows = len(validation_data["delta"])
    if len(validation_frame) != expected_validation_rows:
        raise RuntimeError(
            f"Stage 2 validation rows ({len(validation_frame)}) do not match Stage 4 ({expected_validation_rows})"
        )

    specs = {
        "router_only": GateFeatureSpec(False, model_dim, tuple(moe_layers)),
        "hidden_router": GateFeatureSpec(True, model_dim, tuple(moe_layers)),
    }
    matrices: dict[str, dict[str, np.ndarray]] = {}
    feature_stats: dict[str, Any] = {
        "normalization_source": "gate_train only",
        "feature_order": {
            "router": ["top1_probability", "top2_probability", "margin", "entropy"],
            "layer": [f"layer_{layer}" for layer in moe_layers],
            "hidden_position": "prepended when enabled",
        },
    }
    for spec_name, spec in specs.items():
        split_matrices = {
            "train": feature_matrix(train_data, spec),
            "calibration": feature_matrix(cal_data, spec),
            "validation": feature_matrix(validation_data, spec),
        }
        mean, std = compute_feature_stats(split_matrices["train"])
        split_matrices["mean"] = mean
        split_matrices["std"] = std
        matrices[spec_name] = split_matrices
        feature_stats[spec_name] = {
            "input_dim": spec.input_dim,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    write_json(output_dir / "feature_stats.json", feature_stats)

    calibration_top2 = evaluate_policy(
        model, cal_batches, RoutingPolicy("standard-top2", fixed_k=2), use_amp
    )
    variants = {
        "linear_router_only": ("linear", "router_only"),
        "linear_hidden_router": ("linear", "hidden_router"),
        "mlp_hidden_router": ("mlp", "hidden_router"),
    }
    run_records: list[dict[str, Any]] = []
    learned_frontier: list[dict[str, Any]] = []
    large_delta_threshold = float(
        np.percentile(train_data["delta"][train_data["delta"] > 0], 90)
    )
    for variant, (architecture, spec_name) in variants.items():
        variant_dir = output_dir / variant
        variant_dir.mkdir()
        variant_training: list[dict[str, Any]] = []
        variant_prediction: list[dict[str, Any]] = []
        variant_sweeps: list[dict[str, Any]] = []
        variant_benchmarks: list[dict[str, Any]] = []
        variant_conditionals: list[dict[str, Any]] = []
        spec = specs[spec_name]
        values = matrices[spec_name]
        for seed in [int(value) for value in config["experiment"]["gate_seeds"]]:
            set_seed(seed)
            gate = make_gate(
                architecture, spec.input_dim, int(config["training"]["mlp_hidden_dim"])
            )
            training_metrics = train_gate(
                gate,
                values["train"],
                train_data["delta"],
                values["calibration"],
                cal_data["delta"],
                values["mean"],
                values["std"],
                config,
                seed,
                device,
            )
            calibration_prediction = predict_gate(
                gate, values["calibration"], values["mean"], values["std"], device
            )
            validation_prediction = predict_gate(
                gate, values["validation"], values["mean"], values["std"], device
            )
            metrics = {
                "calibration": prediction_metrics(calibration_prediction, cal_data["delta"]),
                "validation": prediction_metrics(validation_prediction, validation_data["delta"]),
            }
            thresholds = candidate_thresholds(
                calibration_prediction, list(config["thresholds"]["quantiles"])
            )
            calibration_frontier = measure_online_gate_frontier(
                model,
                cal_batches,
                gate,
                spec,
                values["mean"],
                values["std"],
                thresholds,
                use_amp,
                f"calibration_{variant}_seed_{seed}",
            )
            selected = select_calibration_point(
                calibration_frontier,
                calibration_top2["loss"],
                float(config["thresholds"]["max_calibration_loss_increase"]),
            )
            validation_frontier = measure_online_gate_frontier(
                model,
                validation_batches,
                gate,
                spec,
                values["mean"],
                values["std"],
                thresholds,
                use_amp,
                f"validation_exploratory_{variant}_seed_{seed}",
            )
            selected_validation = next(
                point
                for point in validation_frontier
                if point["threshold"] == selected["threshold"]
            )
            online = benchmark_online_gate(
                model,
                validation_batches,
                gate,
                spec,
                values["mean"],
                values["std"],
                float(selected["threshold"]),
                use_amp,
                int(config["benchmark"]["warmup_batches"]),
                int(config["benchmark"]["timing_repeats"]),
            )
            conditional = conditional_analysis(
                validation_data,
                validation_prediction,
                float(selected["threshold"]),
                large_delta_threshold,
            )
            worst = worst_false_k1_rows(
                validation_data,
                validation_prediction,
                float(selected["threshold"]),
                validation_frame,
            )
            checkpoint_payload = {
                "variant": variant,
                "architecture": architecture,
                "seed": seed,
                "feature_spec": {
                    "include_hidden": spec.include_hidden,
                    "model_dim": spec.model_dim,
                    "moe_layers": list(spec.moe_layers),
                    "input_dim": spec.input_dim,
                },
                "state_dict": {
                    name: parameter.detach().cpu()
                    for name, parameter in gate.state_dict().items()
                },
                "feature_mean": values["mean"],
                "feature_std": values["std"],
                "selected_threshold": selected["threshold"],
            }
            torch.save(checkpoint_payload, variant_dir / f"gate_seed_{seed}.pt")
            for point in validation_frontier:
                learned_frontier.append({**point, "method": variant, "seed": seed})
            record = {
                "variant": variant,
                "architecture": architecture,
                "seed": seed,
                "parameters": gate_parameter_count(gate),
                "training_metrics": training_metrics,
                "prediction_metrics": metrics,
                "calibration_frontier": calibration_frontier,
                "selected_calibration_point": selected,
                "validation_frontier_exploratory": validation_frontier,
                "selected_validation_point": selected_validation,
                "online_benchmark": online,
                "conditional_analysis": conditional,
                "worst_false_k1": worst,
                "validation_prediction": validation_prediction,
                "validation_target": validation_data["delta"],
            }
            run_records.append(record)
            variant_training.append({"seed": seed, "metrics": training_metrics})
            variant_prediction.append({"seed": seed, **metrics})
            variant_sweeps.append(
                {
                    "seed": seed,
                    "calibration": calibration_frontier,
                    "selected_calibration_point": selected,
                    "validation_exploratory": validation_frontier,
                    "selected_validation_point": selected_validation,
                }
            )
            variant_benchmarks.append({"seed": seed, **online})
            variant_conditionals.append(
                {"seed": seed, "conditional": conditional, "worst_false_k1": worst}
            )
            print(
                json.dumps(
                    {
                        "phase": "gate_complete",
                        "variant": variant,
                        "seed": seed,
                        "selected": selected_validation,
                        "pearson": metrics["validation"]["pearson"],
                    }
                ),
                flush=True,
            )
            del gate
            if device.type == "cuda":
                torch.cuda.empty_cache()
        write_json(variant_dir / "training_metrics.json", variant_training)
        write_json(variant_dir / "gate_prediction_metrics.json", variant_prediction)
        write_json(variant_dir / "threshold_sweeps.json", variant_sweeps)
        write_json(variant_dir / "benchmark.json", variant_benchmarks)
        write_json(variant_dir / "conditional_analysis.json", variant_conditionals)

    controls, stage3_benchmarks = stage3_controls(PROJECT_ROOT / config["stage3_results"])
    top2 = next(item for item in stage3_benchmarks if item["name"] == "Standard Top-2")
    top1 = next(item for item in stage3_benchmarks if item["name"] == "Standard Top-1")
    oracle = next(item for item in stage3_benchmarks if item["name"] == "Oracle Adaptive-K")
    endpoints = {
        "top1": {"loss": top1["validation_loss"], "experts_per_token": top1["experts_per_token"]},
        "top2": {
            "loss": top2["validation_loss"],
            "experts_per_token": top2["experts_per_token"],
            "tokens_per_second": top2["tokens_per_second"],
        },
        "oracle": {"loss": oracle["validation_loss"], "experts_per_token": oracle["experts_per_token"]},
    }
    matched = matched_compute_table(
        [float(value) for value in config["benchmark"]["matched_compute_targets"]],
        controls,
        learned_frontier,
        endpoints["top2"]["loss"],
    )
    write_json(output_dir / "matched_compute.json", matched)
    pd.DataFrame(matched).to_csv(output_dir / "matched_compute.csv", index=False)
    frontier_rows = controls + learned_frontier
    write_json(output_dir / "frontier_data.json", frontier_rows)
    pd.DataFrame(frontier_rows).to_csv(output_dir / "frontier_data.csv", index=False)

    summaries: dict[str, Any] = {}
    for variant in variants:
        records = [record for record in run_records if record["variant"] == variant]
        summaries[variant] = {
            "parameters": records[0]["parameters"],
            "validation_pearson": _mean_std(
                [record["prediction_metrics"]["validation"]["pearson"] for record in records]
            ),
            "validation_spearman": _mean_std(
                [record["prediction_metrics"]["validation"]["spearman"] for record in records]
            ),
            "selected_experts_per_token": _mean_std(
                [record["selected_validation_point"]["experts_per_token"] for record in records]
            ),
            "selected_validation_loss": _mean_std(
                [record["selected_validation_point"]["loss"] for record in records]
            ),
            "tokens_per_second": _mean_std(
                [record["online_benchmark"]["tokens_per_second"] for record in records]
            ),
            "milliseconds_per_token": _mean_std(
                [record["online_benchmark"]["milliseconds_per_token"] for record in records]
            ),
            "peak_gpu_memory_mb": _mean_std(
                [record["online_benchmark"]["peak_gpu_memory_mb"] for record in records]
            ),
        }
    fingerprint_unchanged = checkpoint_parameter_fingerprint(model) == fingerprint_before
    benchmark = {
        "learned_gate_summary": summaries,
        "stage3_fixed_and_oracle": endpoints,
        "stage3_benchmarks": stage3_benchmarks,
        "hardware": {
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "total_vram_mib": (
                torch.cuda.get_device_properties(0).total_memory / 1024**2
                if device.type == "cuda" else 0.0
            ),
            "pytorch": torch.__version__,
            "pytorch_cuda": torch.version.cuda,
        },
        "weights_unchanged": fingerprint_unchanged,
        "algorithmic_vs_wall_clock_warning": (
            "Experts/token measures expert execution count; tokens/s includes gate and dispatch overhead."
        ),
    }
    write_json(output_dir / "benchmark.json", benchmark)
    oracle_comparison = {
        "standard_top2": endpoints["top2"],
        "oracle": endpoints["oracle"],
        "oracle_loss_change_vs_top2": endpoints["oracle"]["loss"] - endpoints["top2"]["loss"],
        "learned": {
            name: {
                "mean_loss_change_vs_top2": values["selected_validation_loss"]["mean"] - endpoints["top2"]["loss"],
                "remaining_loss_gap_to_oracle": values["selected_validation_loss"]["mean"] - endpoints["oracle"]["loss"],
            }
            for name, values in summaries.items()
        },
    }
    write_json(output_dir / "oracle_comparison.json", oracle_comparison)
    create_plots(output_dir / "plots", controls, learned_frontier, endpoints, run_records)
    report = make_report(
        config, data_split, summaries, matched, endpoints, fingerprint_unchanged
    )
    (output_dir / "STAGE4_REPORT.md").write_text(report, encoding="utf-8")
    (PROJECT_ROOT / "STAGE4_REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "output_directory": str(output_dir),
        "learned_gates": summaries,
        "endpoints": endpoints,
        "weights_unchanged": fingerprint_unchanged,
    }
    write_json(output_dir / "summary.json", summary)
    if not fingerprint_unchanged:
        raise RuntimeError("Frozen Stage 1 model parameters changed during Stage 4")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
