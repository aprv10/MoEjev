from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TokenBlockDataset, prepare_wikitext2  # noqa: E402
from evaluation.oracle import checkpoint_parameter_fingerprint  # noqa: E402
from evaluation.routing_analysis import write_json  # noqa: E402
from evaluation.stage3_benchmark import (  # noqa: E402
    benchmark_policy,
    evaluate_policy,
    preload_batches,
)
from evaluation.stage4_compute_gate import (  # noqa: E402
    benchmark_online_gate,
    candidate_thresholds,
    compute_feature_stats,
    extract_gate_dataset,
    feature_matrix,
    make_gate,
    measure_online_gate_frontier,
    predict_gate,
    prediction_masks,
    prediction_metrics,
    save_gate_dataset,
    select_calibration_point,
    train_gate,
)
from moe.adaptive_routing import RoutingPolicy  # noqa: E402
from moe.compute_gate import GateFeatureSpec, gate_parameter_count  # noqa: E402
from training.trainer import build_model, set_seed, train  # noqa: E402


STAGE5_DELTAS = {
    1.1: -0.002704965750376361,
    1.2: -0.00167229286829634,
    1.3: -0.000626445293426503,
    1.4: 0.0001323908170062301,
    1.5: 0.0006410628954568528,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6 larger-MoE replication")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Resume an existing Stage 6 output directory",
    )
    return parser.parse_args()


def load_gate_dataset(directory: Path) -> dict[str, np.ndarray]:
    return {
        path.stem: np.load(path, allow_pickle=False)
        for path in sorted(directory.glob("*.npy"))
    }


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_config = checkpoint["config"]
    model = build_model(training_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, training_config


def heuristic_signal(data: dict[str, np.ndarray], metric: str) -> np.ndarray:
    column = {"max_probability": 0, "margin": 2, "entropy": 3}[metric]
    return data["router"][:, column].astype(np.float64)


def measure_heuristic_frontier(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    metric: str,
    thresholds: list[float],
    use_amp: bool,
    split: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for threshold in thresholds:
        policy = RoutingPolicy(
            name=f"uncalibrated-{metric}-{threshold:.8g}",
            uncertainty_metric=metric,
            threshold=threshold,
        )
        measured = evaluate_policy(model, batches, policy, use_amp)
        output.append(
            {
                "split": split,
                "method": metric,
                "threshold": threshold,
                "loss": measured["loss"],
                "experts_per_token": measured["experts_per_token"],
                "fraction_k2": measured["experts_per_token"] - 1.0,
            }
        )
    return output


def validation_diagnostics(
    data: dict[str, np.ndarray], baseline_metrics: dict[str, Any]
) -> dict[str, Any]:
    losses = data["expert_losses"].astype(np.float32)
    top1 = data["router_top1"].astype(np.int64)
    top2 = data["router_top2"].astype(np.int64)
    best = losses.argmin(axis=1)
    rows = np.arange(len(losses))
    regret = np.maximum(losses[rows, top1] - losses.min(axis=1), 0.0)
    return {
        "router_entropy": baseline_metrics["routing_entropy"],
        "expert_utilization": baseline_metrics["expert_utilization"],
        "top1_oracle_accuracy": float((top1 == best).mean()),
        "top2_oracle_coverage": float((top2 == best[:, None]).any(axis=1).mean()),
        "mean_forced_top1_regret": float(regret.mean()),
        "positive_delta_fraction": float((data["delta"] > 0).mean()),
    }


def nearest_point(points: list[dict[str, Any]], target: float) -> dict[str, Any]:
    return min(points, key=lambda point: abs(point["experts_per_token"] - target))


def ci95(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0])
        return value, value
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    mean = float(values.mean())
    half = critical * float(values.std(ddof=1)) / math.sqrt(len(values))
    return mean - half, mean + half


def plot_checkpoint(
    path: Path,
    seed: int,
    baselines: dict[str, Any],
    gate_runs: list[dict[str, Any]],
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    margin = sorted(baselines["validation_frontiers"]["margin"], key=lambda x: x["experts_per_token"])
    axis.plot(
        [point["experts_per_token"] for point in margin],
        [point["loss"] for point in margin],
        "o-", color="#e15759", label="Margin heuristic",
    )
    for index, run in enumerate(gate_runs):
        points = sorted(run["validation_frontier"], key=lambda x: x["experts_per_token"])
        axis.plot(
            [point["experts_per_token"] for point in points],
            [point["loss"] for point in points],
            "o-", color="#f28e2b", alpha=0.55,
            label="MLP compute gate" if index == 0 else None,
        )
    for name, marker, color in [("top1", "s", "black"), ("top2", "s", "black"), ("oracle", "*", "#17becf")]:
        point = baselines[name]
        axis.scatter(point["experts_per_token"], point["loss"], marker=marker, s=80, color=color, label=name.title(), zorder=5)
    axis.set(
        xlabel="Average experts per token",
        ylabel="Validation loss",
        title=f"Larger MoE seed {seed}: measured frontier",
    )
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def create_aggregate_plots(
    output_dir: Path,
    matched: pd.DataFrame,
    checkpoint_summary: pd.DataFrame,
    aggregate_stats: dict[str, Any],
) -> None:
    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)
    checkpoint_means = (
        matched.groupby(["base_seed", "target_experts_per_token"], as_index=False)["delta_loss"]
        .mean()
    )
    figure, axis = plt.subplots(figsize=(7.6, 5.0))
    for seed, group in checkpoint_means.groupby("base_seed"):
        axis.plot(group["target_experts_per_token"], group["delta_loss"], "o-", alpha=0.55, label=f"MoE {seed}")
    grouped = checkpoint_means.groupby("target_experts_per_token")["delta_loss"]
    targets = np.asarray(sorted(grouped.groups), dtype=np.float64)
    means = np.asarray([grouped.get_group(target).mean() for target in targets])
    stds = np.asarray([grouped.get_group(target).std(ddof=1) for target in targets])
    axis.errorbar(targets, means, yerr=stds, fmt="o-", color="black", lw=2.2, capsize=4, label="Larger-model mean ± SD")
    axis.axhline(0, color="0.35", ls="--", lw=1)
    axis.set(xlabel="Target experts/token", ylabel="MLP loss − margin loss", title="Stage 6: learned gate versus router margin")
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(plots / "matched_compute_delta.png", dpi=180)
    plt.close(figure)

    pivot = checkpoint_means.pivot(index="base_seed", columns="target_experts_per_token", values="delta_loss")
    figure, axis = plt.subplots(figsize=(7.4, 3.9))
    image = axis.imshow(pivot.to_numpy(), cmap="RdBu_r", aspect="auto", vmin=-max(abs(pivot.to_numpy()).max(), 1e-6), vmax=max(abs(pivot.to_numpy()).max(), 1e-6))
    axis.set_xticks(range(len(pivot.columns)), [f"{value:.1f}" for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [str(value) for value in pivot.index])
    axis.set(xlabel="Target experts/token", ylabel="Larger MoE seed", title="Per-model mean Δloss (MLP − margin)")
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iloc[row, column]
            axis.text(column, row, f"{value:+.4f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="Δloss")
    figure.tight_layout()
    figure.savefig(plots / "checkpoint_delta_heatmap.png", dpi=180)
    plt.close(figure)

    margin_means = matched.groupby(["base_seed", "target_experts_per_token"], as_index=False).first()
    mlp_means = matched.groupby(["base_seed", "target_experts_per_token"], as_index=False).agg(
        mlp_loss=("mlp_validation_loss", "mean"),
        mlp_experts=("mlp_actual_experts_per_token", "mean"),
        margin_loss=("margin_validation_loss", "first"),
        margin_experts=("margin_actual_experts_per_token", "first"),
    )
    figure, axis = plt.subplots(figsize=(7.4, 4.9))
    for method, loss_column, expert_column, color in [
        ("Margin", "margin_loss", "margin_experts", "#e15759"),
        ("MLP gate", "mlp_loss", "mlp_experts", "#f28e2b"),
    ]:
        rows = []
        for target, group in mlp_means.groupby("target_experts_per_token"):
            rows.append((group[expert_column].mean(), group[loss_column].mean(), group[loss_column].std(ddof=1)))
        rows.sort()
        axis.errorbar([r[0] for r in rows], [r[1] for r in rows], yerr=[r[2] for r in rows], fmt="o-", capsize=4, color=color, label=f"{method} mean ± SD")
    axis.scatter(1.0, checkpoint_summary["top1_loss"].mean(), marker="s", color="black", label="Top-1 mean")
    axis.scatter(2.0, checkpoint_summary["top2_loss"].mean(), marker="s", color="0.35", label="Top-2 mean")
    axis.set(xlabel="Average experts per token", ylabel="Mean validation loss", title="Stage 6 quality–compute frontier across larger MoEs")
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(plots / "aggregate_quality_vs_compute.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    checkpoint_summary = checkpoint_summary.copy()
    advantage = checkpoint_means[checkpoint_means["target_experts_per_token"] == 1.2].set_index("base_seed")["delta_loss"]
    checkpoint_summary["delta_at_1_2"] = checkpoint_summary["base_seed"].map(advantage)
    axes[0].scatter(checkpoint_summary["top2_loss"], checkpoint_summary["delta_at_1_2"], color="#4e79a7")
    axes[0].axhline(0, color="0.5", ls="--")
    axes[0].set(xlabel="Top-2 validation loss", ylabel="MLP − margin at 1.2", title="Base quality vs gate advantage")
    axes[1].bar(checkpoint_summary["base_seed"].astype(str), checkpoint_summary["top2_loss"] - checkpoint_summary["oracle_loss"], color="#17becf")
    axes[1].set(xlabel="Fresh MoE seed", ylabel="Top-2 loss − oracle loss", title="Oracle adaptive-compute headroom")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(plots / "base_quality_vs_gate_advantage.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    headroom = checkpoint_summary["top2_loss"] - checkpoint_summary["oracle_loss"]
    axis.bar(checkpoint_summary["base_seed"].astype(str), headroom, color="#17becf")
    axis.set(
        xlabel="Fresh MoE seed",
        ylabel="Top-2 loss − oracle loss",
        title="Oracle adaptive-compute headroom across checkpoints",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(plots / "oracle_gap.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    for axis, column, label in [
        (axes[0], "router_entropy", "Router entropy"),
        (axes[1], "top1_oracle_accuracy", "Top-1 oracle accuracy"),
        (axes[2], "mean_forced_top1_regret", "Mean forced-Top-1 regret"),
    ]:
        axis.scatter(checkpoint_summary[column], checkpoint_summary["delta_at_1_2"], color="#4e79a7")
        axis.axhline(0, color="0.5", ls="--", lw=0.9)
        axis.set(xlabel=label, ylabel="MLP − margin at 1.2")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Router behavior versus learned-gate advantage (exploratory, N=3)")
    figure.tight_layout()
    figure.savefig(plots / "router_behavior_vs_gate_advantage.png", dpi=180)
    plt.close(figure)


def aggregate_results(output_dir: Path, config: dict[str, Any], seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    matched_rows = [row for result in seed_results for row in result["matched_compute"]]
    matched = pd.DataFrame(matched_rows)
    checkpoint_rows = [result["checkpoint_summary"] for result in seed_results]
    checkpoint_summary = pd.DataFrame(checkpoint_rows)
    aggregate_dir = output_dir / "aggregate"
    aggregate_dir.mkdir(exist_ok=True)
    matched.to_csv(aggregate_dir / "matched_compute.csv", index=False)
    write_json(aggregate_dir / "matched_compute.json", matched.to_dict(orient="records"))
    checkpoint_summary.to_csv(aggregate_dir / "checkpoint_summary.csv", index=False)
    tie = float(config["benchmark"]["tie_tolerance"])
    checkpoint_means = matched.groupby(["base_seed", "target_experts_per_token"])["delta_loss"].mean().reset_index()
    stats: dict[str, Any] = {}
    for target, group in checkpoint_means.groupby("target_experts_per_token"):
        values = group["delta_loss"].to_numpy(dtype=np.float64)
        low, high = ci95(values)
        individual = matched[matched["target_experts_per_token"] == target]["delta_loss"].to_numpy()
        stats[f"{target:.1f}"] = {
            "checkpoint_mean_delta": float(values.mean()),
            "checkpoint_median_delta": float(np.median(values)),
            "checkpoint_std_delta": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "ci95_t_interval": [low, high],
            "checkpoints_won": int((values < -tie).sum()),
            "checkpoints_tied": int((np.abs(values) <= tie).sum()),
            "checkpoints_lost": int((values > tie).sum()),
            "gate_seed_comparisons_won": int((individual < -tie).sum()),
            "gate_seed_comparisons_tied": int((np.abs(individual) <= tie).sum()),
            "gate_seed_comparisons_lost": int((individual > tie).sum()),
            "stage5_delta": STAGE5_DELTAS[float(target)],
            "stage6_minus_stage5": float(values.mean() - STAGE5_DELTAS[float(target)]),
            "between_checkpoint_std_of_means": float(values.std(ddof=1)),
            "mean_within_checkpoint_gate_seed_std": float(
                matched[matched["target_experts_per_token"] == target]
                .groupby("base_seed")["delta_loss"].std(ddof=1).mean()
            ),
        }

    primary_targets = ["1.1", "1.2", "1.3"]
    minimum_majority = len(seed_results) // 2 + 1
    strong = all(
        stats[target]["checkpoint_mean_delta"] < 0
        and stats[target]["checkpoints_won"] >= minimum_majority
        for target in primary_targets
    )
    partial = any(stats[target]["checkpoint_mean_delta"] < 0 for target in primary_targets)
    verdict = "strong" if strong else ("partial" if partial else "not replicated")
    prediction_rows = []
    for result in seed_results:
        for run in result["gate_runs"]:
            prediction_rows.append(
                {
                    "base_seed": result["base_seed"],
                    "gate_seed": run["gate_seed"],
                    **{key: run["prediction_metrics"][key] for key in ["pearson", "spearman", "mae", "rmse", "sign_accuracy"]},
                }
            )
    prediction = pd.DataFrame(prediction_rows)
    prediction.to_csv(aggregate_dir / "prediction_summary.csv", index=False)
    mean_delta_per_run = matched.groupby(["base_seed", "gate_seed"])["delta_loss"].mean()
    prediction_indexed = prediction.set_index(["base_seed", "gate_seed"])
    prediction_gain_correlation = float(np.corrcoef(prediction_indexed.loc[mean_delta_per_run.index, "pearson"], mean_delta_per_run.to_numpy())[0, 1])
    per_checkpoint_prediction = prediction.groupby("base_seed", as_index=False).mean(numeric_only=True)
    delta_at_1_2 = checkpoint_means[checkpoint_means["target_experts_per_token"] == 1.2].set_index("base_seed")["delta_loss"]
    per_checkpoint_prediction_correlation = float(
        np.corrcoef(
            per_checkpoint_prediction.set_index("base_seed").loc[delta_at_1_2.index, "pearson"],
            delta_at_1_2.to_numpy(),
        )[0, 1]
    )
    prediction_detail = {
        "per_checkpoint_means": per_checkpoint_prediction.to_dict(orient="records"),
        "per_gate_run_validation_deciles": [
            {
                "base_seed": result["base_seed"],
                "gate_seed": run["gate_seed"],
                "deciles": run["prediction_metrics"]["deciles"],
            }
            for result in seed_results
            for run in result["gate_runs"]
        ],
        "checkpoint_mean_pearson_vs_delta_at_1_2_correlation": per_checkpoint_prediction_correlation,
    }
    write_json(aggregate_dir / "prediction_summary.json", prediction_detail)

    checkpoint_indexed = checkpoint_summary.set_index("base_seed")
    aligned_delta = delta_at_1_2.loc[checkpoint_indexed.index].to_numpy()
    router_correlations = {
        column: float(np.corrcoef(checkpoint_indexed[column].to_numpy(), aligned_delta)[0, 1])
        for column in [
            "top2_loss",
            "router_entropy",
            "expert_utilization_std",
            "top1_oracle_accuracy",
            "top2_oracle_coverage",
            "mean_forced_top1_regret",
        ]
    }
    oracle_summary = {
        "all_checkpoints_positive_headroom": bool((checkpoint_summary["top2_loss"] > checkpoint_summary["oracle_loss"]).all()),
        "mean_top2_minus_oracle_loss": float((checkpoint_summary["top2_loss"] - checkpoint_summary["oracle_loss"]).mean()),
        "minimum_top2_minus_oracle_loss": float((checkpoint_summary["top2_loss"] - checkpoint_summary["oracle_loss"]).min()),
        "mean_oracle_experts_per_token": float(checkpoint_summary["oracle_experts_per_token"].mean()),
    }
    write_json(aggregate_dir / "oracle_summary.json", oracle_summary)
    aggregate = {
        "fresh_checkpoint_count": len(seed_results),
        "gate_runs": len(prediction),
        "matched_compute": stats,
        "prediction": {
            "mean_pearson": float(prediction["pearson"].mean()),
            "std_pearson": float(prediction["pearson"].std(ddof=1)),
            "mean_mae": float(prediction["mae"].mean()),
            "mean_rmse": float(prediction["rmse"].mean()),
            "mean_sign_accuracy": float(prediction["sign_accuracy"].mean()),
            "pearson_vs_mean_routing_delta_correlation": prediction_gain_correlation,
            "checkpoint_mean_pearson_vs_delta_at_1_2_correlation": per_checkpoint_prediction_correlation,
        },
        "exploratory_router_behavior_correlations_with_delta_at_1_2": router_correlations,
        "oracle": oracle_summary,
        "replication_verdict": verdict,
        "limitations": [
            "Only three checkpoints of one larger architecture were evaluated.",
            "Confidence intervals are wide and descriptive at N=3.",
            "Gate seeds are nested within checkpoints and are not independent base-model replications.",
            "Nearest measured points are approximate compute matches, not interpolation.",
            "Oracle decisions use validation targets and are non-deployable.",
            "Runtime reflects an unoptimized Python/PyTorch dynamic-dispatch implementation.",
        ],
    }
    write_json(aggregate_dir / "aggregate_metrics.json", aggregate)
    create_aggregate_plots(output_dir, matched, checkpoint_summary, aggregate)
    return aggregate


def make_report(config: dict[str, Any], aggregate: dict[str, Any], seed_results: list[dict[str, Any]]) -> str:
    targets = [float(value) for value in config["benchmark"]["matched_compute_targets"]]
    all_matched = pd.DataFrame(
        [row for result in seed_results for row in result["matched_compute"]]
    )
    max_mlp_compute_mismatch = float(
        (all_matched["mlp_actual_experts_per_token"] - all_matched["target_experts_per_token"])
        .abs()
        .max()
    )
    max_margin_compute_mismatch = float(
        (all_matched["margin_actual_experts_per_token"] - all_matched["target_experts_per_token"])
        .abs()
        .max()
    )
    delta_rows = []
    for result in seed_results:
        frame = pd.DataFrame(result["matched_compute"])
        values = [
            frame[frame["target_experts_per_token"] == target]["delta_loss"].mean()
            for target in targets
        ]
        delta_rows.append(
            "| " + str(result["base_seed"]) + " | "
            + " | ".join(f"{value:+.4f}" for value in values) + " |"
        )

    baseline_rows = []
    gate_rows = []
    for result in seed_results:
        summary = result["checkpoint_summary"]
        baseline_rows.append(
            f"| {result['base_seed']} | {summary['parameter_count'] / 1e6:.2f}M | "
            f"{summary['training_validation_loss']:.4f} | {summary['validation_perplexity']:.2f} | "
            f"{summary['router_entropy']:.4f} | {summary['expert_utilization_std']:.4f} |"
        )
        for run in result["gate_runs"]:
            selected = run["selected_validation"]
            benchmark = run["benchmark"]
            gate_rows.append(
                f"| {result['base_seed']} | {run['gate_seed']} | "
                f"{run['prediction_metrics']['pearson']:+.4f} | {selected['experts_per_token']:.3f} | "
                f"{selected['loss']:.4f} | {selected['fraction_k2']:.3f} | "
                f"{benchmark['tokens_per_second']:.0f} | {benchmark['peak_gpu_memory_mb']:.1f} |"
            )

    consistency = []
    negative_budgets = []
    all_model_budgets = []
    comparison_lines = []
    for target in targets:
        item = aggregate["matched_compute"][f"{target:.1f}"]
        if item["checkpoint_mean_delta"] < 0:
            negative_budgets.append(f"{target:.2f}")
        if item["checkpoints_won"] == aggregate["fresh_checkpoint_count"]:
            all_model_budgets.append(f"{target:.2f}")
        consistency.append(
            f"- {target:.2f}: mean {item['checkpoint_mean_delta']:+.4f}, "
            f"{item['checkpoints_won']}/{aggregate['fresh_checkpoint_count']} models won, "
            f"95% t-CI [{item['ci95_t_interval'][0]:+.4f}, {item['ci95_t_interval'][1]:+.4f}]"
        )
        comparison_lines.append(
            f"- {target:.2f}: Stage 6 {item['checkpoint_mean_delta']:+.4f} versus "
            f"Stage 5 {item['stage5_delta']:+.4f}"
        )

    primary = [aggregate["matched_compute"][key]["checkpoint_mean_delta"] for key in ["1.1", "1.2", "1.3"]]
    stage5_primary = [STAGE5_DELTAS[target] for target in [1.1, 1.2, 1.3]]
    magnitude_ratio = float(np.mean(np.abs(primary)) / np.mean(np.abs(stage5_primary)))
    magnitude = "larger" if magnitude_ratio > 1.2 else ("smaller" if magnitude_ratio < 0.8 else "similar")
    top2_tps = float(np.mean([result["checkpoint_summary"]["top2_tokens_per_second"] for result in seed_results]))
    mlp_tps = float(np.mean([run["benchmark"]["tokens_per_second"] for result in seed_results for run in result["gate_runs"]]))
    mlp_peak_allocated = max(run["benchmark"]["peak_gpu_memory_mb"] for result in seed_results for run in result["gate_runs"])
    mlp_peak_reserved = max(run["benchmark"]["peak_gpu_memory_reserved_mb"] for result in seed_results for run in result["gate_runs"])
    speedup = mlp_tps / top2_tps
    runtime_statement = (
        f"a measured {speedup:.2f}× throughput improvement"
        if speedup > 1.0
        else f"no runtime improvement ({speedup:.2f}× fixed-Top-2 throughput)"
    )
    adaptive_depth = (
        "Yes, as a separate predeclared follow-up; the larger-model matched-compute result is directionally robust."
        if aggregate["replication_verdict"] == "strong"
        else "No—not from Stage 6 alone; the larger-model compute-gating result is not sufficiently consistent."
    )
    verdict_text = {
        "strong": "The learned-gate advantage survived scale under the predeclared majority criterion",
        "partial": "The learned-gate advantage survived only partially at larger scale",
        "not replicated": "The learned-gate advantage did not survive scale",
    }[aggregate["replication_verdict"]]
    negative_text = ", ".join(negative_budgets) if negative_budgets else "none"
    all_model_text = ", ".join(all_model_budgets) if all_model_budgets else "none"

    return f"""# Stage 6 larger-MoE replication report

## Result

**{verdict_text}.** Three independently initialized 35.62M-parameter MoEs were
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
{chr(10).join(baseline_rows)}

Full per-layer eight-expert utilization vectors are in each model's
`baseline_metrics.json` and `baseline_evaluation.json`.

## Per-model mean MLP − margin loss

| MoE seed | 1.10 | 1.20 | 1.30 | 1.40 | 1.50 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(delta_rows)}

## Aggregate matched-compute result

{chr(10).join(consistency)}

These are nearest measured operating points rather than interpolated values.
The maximum absolute target mismatch is {max_mlp_compute_mismatch:.4f}
experts/token for the MLP and {max_margin_compute_mismatch:.4f} for margin.

## Gate prediction and selected-policy benchmarks

| MoE | Gate seed | Pearson | Experts/token | Validation loss | Fraction K=2 | Tokens/s | Peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(gate_rows)}

These throughput measurements use the calibration-selected operating point;
fewer expert executions are not treated as proof of a speedup.

## Answers to the research questions

1. **Did the advantage survive scale?** The predeclared verdict is **{aggregate['replication_verdict']}**.
2. **At which budgets?** Aggregate delta is negative at {negative_text} experts/token.
3. **Consistent across all three models?** All three larger MoEs win at {all_model_text} experts/token.
4. **Effect versus Stage 5:** The mean absolute 1.10–1.30 effect is **{magnitude}** ({magnitude_ratio:.2f}× Stage 5). Exact comparisons are below.
5. **Oracle headroom:** All models retain positive headroom: {aggregate['oracle']['all_checkpoints_positive_headroom']}; mean Top-2 minus oracle loss is {aggregate['oracle']['mean_top2_minus_oracle_loss']:.4f}.
6. **Runtime:** MLP gating averages {mlp_tps:.0f} tokens/s versus {top2_tps:.0f} for fixed Top-2: {runtime_statement}. Peak selected-gate memory is {mlp_peak_allocated:.1f} MiB allocated and {mlp_peak_reserved:.1f} MiB reserved.
7. **Adaptive depth next?** {adaptive_depth}

## Comparison with Stage 5

{chr(10).join(comparison_lines)}

## Limitations

{chr(10).join('- ' + item for item in aggregate['limitations'])}

This project is inspired by calibrated decision models, but it does not
reproduce Jev or RLCD. Stage 6 changes model scale and expert count while
keeping the learned compute-gate method fixed.
"""


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_config = yaml.safe_load((PROJECT_ROOT / config["base_config"]).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.run_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = PROJECT_ROOT / config["experiment"]["output_dir"] / f"{config['experiment']['name']}-{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json(output_dir / "config.json", config)
        protocol = {
            "status": "predeclared before larger-checkpoint evaluation",
            "base_seeds": config["experiment"]["base_moe_seeds"],
            "gate_seeds": config["experiment"]["gate_seeds"],
            "budgets": config["benchmark"]["matched_compute_targets"],
            "method_frozen_from_stage4": True,
            "stage5_method_unchanged_except_feature_width": True,
            "adaptive_k_values": [1, 2],
        }
        write_json(output_dir / "replication_protocol.json", protocol)
    else:
        output_dir = args.run_dir.resolve()
        if not (output_dir / "replication_protocol.json").exists():
            raise FileNotFoundError("--run-dir is not a Stage 6 replication directory")

    data_config = base_config["data"]
    paths = prepare_wikitext2(
        PROJECT_ROOT / data_config["cache_dir"],
        data_config.get("max_train_tokens"),
        data_config.get("max_validation_tokens"),
    )
    sequence_length = int(base_config["model"]["sequence_length"])
    train_dataset = TokenBlockDataset(paths["train"], sequence_length)
    validation_dataset = TokenBlockDataset(paths["validation"], sequence_length)
    batch_size = int(config["data"]["batch_size"])
    use_amp = device.type == "cuda" and config["benchmark"]["mixed_precision"] == "fp16"
    expected_parameter_count = build_model(base_config).parameter_count()
    if not 25_000_000 <= expected_parameter_count <= 40_000_000:
        raise RuntimeError(
            f"Stage 6 model has {expected_parameter_count:,} parameters; expected 25–40M"
        )
    seed_results: list[dict[str, Any]] = []

    for base_seed in [int(seed) for seed in config["experiment"]["base_moe_seeds"]]:
        seed_dir = output_dir / f"moe_seed_{base_seed}"
        checkpoint_path = seed_dir / "checkpoint.pt"
        if not checkpoint_path.exists():
            if seed_dir.exists():
                raise RuntimeError(f"Incomplete training directory exists: {seed_dir}. Move it aside before retrying this seed.")
            training_config = copy.deepcopy(base_config)
            training_config["experiment"]["name"] = f"stage6-larger-moe-seed-{base_seed}"
            training_config["experiment"]["seed"] = base_seed
            print(json.dumps({"phase": "base_training_start", "base_seed": base_seed}), flush=True)
            train(
                training_config,
                PROJECT_ROOT,
                run_dir_override=seed_dir,
                checkpoint_filename="checkpoint.pt",
                summary_filename="baseline_metrics.json",
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        baseline_metrics = json.loads((seed_dir / "baseline_metrics.json").read_text(encoding="utf-8"))
        if (
            baseline_metrics["parameter_count"] != expected_parameter_count
            or baseline_metrics["steps"] != int(base_config["training"]["max_steps"])
        ):
            raise RuntimeError(f"Training parity failed for seed {base_seed}: {baseline_metrics}")
        model, training_config = load_model(checkpoint_path, device)
        fingerprint = checkpoint_parameter_fingerprint(model)
        moe_layers = [int(layer) for layer in training_config["model"]["moe_layers"]]
        spec = GateFeatureSpec(True, int(training_config["model"]["model_dim"]), tuple(moe_layers))

        train_start = int(config["data"]["gate_train_block_start"])
        train_blocks = int(config["data"]["gate_train_blocks"])
        cal_start = int(config["data"]["gate_calibration_block_start"])
        cal_blocks = int(config["data"]["gate_calibration_blocks"])
        if train_start + train_blocks >= cal_start:
            raise RuntimeError("Gate training/calibration shifted target regions overlap")
        train_loader = DataLoader(Subset(train_dataset, range(train_start, train_start + train_blocks)), batch_size=batch_size, shuffle=False, num_workers=0)
        cal_loader = DataLoader(Subset(train_dataset, range(cal_start, cal_start + cal_blocks)), batch_size=batch_size, shuffle=False, num_workers=0)
        val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        train_batches = preload_batches(train_loader, device, math.ceil(train_blocks / batch_size))
        cal_batches = preload_batches(cal_loader, device, math.ceil(cal_blocks / batch_size))
        val_batches = preload_batches(val_loader, device, int(config["data"]["validation_batches"]))

        cache = seed_dir / "feature_cache"
        if (cache / "complete.json").exists():
            train_data = load_gate_dataset(cache / "gate_train")
            cal_data = load_gate_dataset(cache / "gate_calibration")
            val_data = load_gate_dataset(cache / "final_validation")
        else:
            cache.mkdir(exist_ok=True)
            train_data = extract_gate_dataset(model, train_batches, moe_layers, config, f"seed_{base_seed}_train")
            save_gate_dataset(cache / "gate_train", train_data)
            cal_data = extract_gate_dataset(model, cal_batches, moe_layers, config, f"seed_{base_seed}_cal")
            save_gate_dataset(cache / "gate_calibration", cal_data)
            val_data = extract_gate_dataset(model, val_batches, moe_layers, config, f"seed_{base_seed}_val")
            save_gate_dataset(cache / "final_validation", val_data)
            write_json(cache / "complete.json", {"complete": True})

        features = {
            "train": feature_matrix(train_data, spec),
            "calibration": feature_matrix(cal_data, spec),
            "validation": feature_matrix(val_data, spec),
        }
        mean, std = compute_feature_stats(features["train"])
        baseline_path = seed_dir / "baseline_evaluation.json"
        if baseline_path.exists():
            baselines = json.loads(baseline_path.read_text(encoding="utf-8"))
        else:
            calibration_top2 = evaluate_policy(model, cal_batches, RoutingPolicy("top2", fixed_k=2), use_amp)
            validation_frontiers: dict[str, list[dict[str, Any]]] = {}
            selected_heuristics: dict[str, Any] = {}
            for metric in ["max_probability", "entropy", "margin"]:
                thresholds = candidate_thresholds(heuristic_signal(cal_data, metric), list(config["thresholds"]["quantiles"]))
                cal_curve = measure_heuristic_frontier(model, cal_batches, metric, thresholds, use_amp, "gate_calibration")
                selected = select_calibration_point(cal_curve, calibration_top2["loss"], float(config["thresholds"]["max_calibration_loss_increase"]))
                val_curve = measure_heuristic_frontier(model, val_batches, metric, thresholds, use_amp, "final_validation_exploratory_frontier")
                validation_frontiers[metric] = val_curve
                selected_heuristics[metric] = {
                    "calibration": selected,
                    "validation": next(point for point in val_curve if point["threshold"] == selected["threshold"]),
                }
            top2 = benchmark_policy(model, val_batches, RoutingPolicy("top2", fixed_k=2), use_amp, int(config["benchmark"]["warmup_batches"]), int(config["benchmark"]["timing_repeats"]))
            top1 = benchmark_policy(model, val_batches, RoutingPolicy("top1", fixed_k=1), use_amp, int(config["benchmark"]["warmup_batches"]), int(config["benchmark"]["timing_repeats"]))
            oracle_masks = prediction_masks(val_data, val_data["delta"], 0.0, moe_layers, sum(x.size(0) for x, _ in val_batches), sequence_length)
            oracle = benchmark_policy(model, val_batches, RoutingPolicy("oracle", fixed_k=2), use_amp, int(config["benchmark"]["warmup_batches"]), int(config["benchmark"]["timing_repeats"]), oracle_masks)
            baselines = {
                "calibration_top2_loss": calibration_top2["loss"],
                "validation_frontiers": validation_frontiers,
                "selected_heuristics": selected_heuristics,
                "top2": {"loss": top2["loss"], "experts_per_token": top2["experts_per_token"], "tokens_per_second": top2["tokens_per_second"], "peak_gpu_memory_mb": top2["peak_gpu_memory_mb"], "peak_gpu_memory_reserved_mb": top2["peak_gpu_memory_reserved_mb"]},
                "top1": {"loss": top1["loss"], "experts_per_token": top1["experts_per_token"], "tokens_per_second": top1["tokens_per_second"]},
                "oracle": {"loss": oracle["loss"], "experts_per_token": oracle["experts_per_token"], "tokens_per_second": oracle["tokens_per_second"]},
                "router_diagnostics": validation_diagnostics(val_data, baseline_metrics),
            }
            write_json(baseline_path, baselines)

        gate_runs: list[dict[str, Any]] = []
        gate_root = seed_dir / "gate_runs"
        gate_root.mkdir(exist_ok=True)
        gate_training_config = {"training": config["gate"]}
        for gate_seed in [int(seed) for seed in config["experiment"]["gate_seeds"]]:
            gate_dir = gate_root / f"seed_{gate_seed}"
            result_path = gate_dir / "result.json"
            if result_path.exists():
                gate_runs.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            gate_dir.mkdir(exist_ok=False)
            set_seed(gate_seed)
            gate = make_gate("mlp", spec.input_dim, int(config["gate"]["hidden_dim"]))
            training_metrics = train_gate(gate, features["train"], train_data["delta"], features["calibration"], cal_data["delta"], mean, std, gate_training_config, gate_seed, device)
            cal_prediction = predict_gate(gate, features["calibration"], mean, std, device)
            val_prediction = predict_gate(gate, features["validation"], mean, std, device)
            metrics = prediction_metrics(val_prediction, val_data["delta"])
            thresholds = candidate_thresholds(cal_prediction, list(config["thresholds"]["quantiles"]))
            cal_frontier = measure_online_gate_frontier(model, cal_batches, gate, spec, mean, std, thresholds, use_amp, f"calibration_seed_{base_seed}_{gate_seed}")
            selected = select_calibration_point(cal_frontier, baselines["calibration_top2_loss"], float(config["thresholds"]["max_calibration_loss_increase"]))
            val_frontier = measure_online_gate_frontier(model, val_batches, gate, spec, mean, std, thresholds, use_amp, f"validation_seed_{base_seed}_{gate_seed}")
            selected_validation = next(point for point in val_frontier if point["threshold"] == selected["threshold"])
            benchmark = benchmark_online_gate(model, val_batches, gate, spec, mean, std, float(selected["threshold"]), use_amp, int(config["benchmark"]["warmup_batches"]), int(config["benchmark"]["timing_repeats"]))
            result = {
                "base_seed": base_seed,
                "gate_seed": gate_seed,
                "parameters": gate_parameter_count(gate),
                "training_metrics": training_metrics,
                "prediction_metrics": metrics,
                "calibration_frontier": cal_frontier,
                "selected_calibration": selected,
                "validation_frontier": val_frontier,
                "selected_validation": selected_validation,
                "benchmark": benchmark,
            }
            torch.save(
                {
                    "state_dict": {name: value.detach().cpu() for name, value in gate.state_dict().items()},
                    "feature_mean": mean,
                    "feature_std": std,
                    "selected_threshold": selected["threshold"],
                    "base_seed": base_seed,
                    "gate_seed": gate_seed,
                },
                gate_dir / "gate.pt",
            )
            write_json(result_path, result)
            gate_runs.append(result)
            del gate
            if device.type == "cuda":
                torch.cuda.empty_cache()

        matched_rows: list[dict[str, Any]] = []
        margin_frontier = baselines["validation_frontiers"]["margin"]
        for run in gate_runs:
            for target in [float(value) for value in config["benchmark"]["matched_compute_targets"]]:
                margin = nearest_point(margin_frontier, target)
                mlp = nearest_point(run["validation_frontier"], target)
                matched_rows.append(
                    {
                        "base_seed": base_seed,
                        "gate_seed": run["gate_seed"],
                        "target_experts_per_token": target,
                        "mlp_actual_experts_per_token": mlp["experts_per_token"],
                        "margin_actual_experts_per_token": margin["experts_per_token"],
                        "mlp_validation_loss": mlp["loss"],
                        "margin_validation_loss": margin["loss"],
                        "delta_loss": mlp["loss"] - margin["loss"],
                        "mlp_threshold": mlp["threshold"],
                        "margin_threshold": margin["threshold"],
                    }
                )
        summary = {
            "base_seed": base_seed,
            "parameter_count": baseline_metrics["parameter_count"],
            "train_loss": baseline_metrics["train_loss"],
            "training_validation_loss": baseline_metrics["validation_loss"],
            "validation_perplexity": baseline_metrics["validation_perplexity"],
            "training_tokens_per_second": baseline_metrics["tokens_per_second"],
            "training_peak_gpu_memory_mb": baseline_metrics["peak_gpu_memory_mb"],
            "training_peak_gpu_memory_reserved_mb": baseline_metrics["peak_gpu_memory_reserved_mb"],
            "top2_loss": baselines["top2"]["loss"],
            "top1_loss": baselines["top1"]["loss"],
            "oracle_loss": baselines["oracle"]["loss"],
            "oracle_experts_per_token": baselines["oracle"]["experts_per_token"],
            "top2_tokens_per_second": baselines["top2"]["tokens_per_second"],
            "top2_peak_gpu_memory_mb": baselines["top2"]["peak_gpu_memory_mb"],
            "top2_peak_gpu_memory_reserved_mb": baselines["top2"]["peak_gpu_memory_reserved_mb"],
            "expert_utilization_std": float(
                np.std(np.asarray(baselines["router_diagnostics"]["expert_utilization"], dtype=np.float64))
            ),
            **baselines["router_diagnostics"],
        }
        result = {"base_seed": base_seed, "checkpoint_summary": summary, "baselines": baselines, "gate_runs": gate_runs, "matched_compute": matched_rows}
        write_json(seed_dir / "replication_result.json", result)
        per_checkpoint = output_dir / "plots" / "per_checkpoint"
        per_checkpoint.mkdir(parents=True, exist_ok=True)
        plot_checkpoint(per_checkpoint / f"moe_seed_{base_seed}.png", base_seed, baselines, gate_runs)
        if checkpoint_parameter_fingerprint(model) != fingerprint:
            raise RuntimeError(f"Frozen model weights changed for seed {base_seed}")
        seed_results.append(result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"phase": "checkpoint_complete", "base_seed": base_seed}), flush=True)

    aggregate = aggregate_results(output_dir, config, seed_results)
    write_json(output_dir / "oracle_comparison.json", aggregate["oracle"])
    report = make_report(config, aggregate, seed_results)
    (output_dir / "STAGE6_REPORT.md").write_text(report, encoding="utf-8")
    (PROJECT_ROOT / "STAGE6_REPORT.md").write_text(report, encoding="utf-8")
    final = {
        "output_directory": str(output_dir),
        "larger_moe_checkpoints_completed": len(seed_results),
        "parameter_count": expected_parameter_count,
        "matched_compute": aggregate["matched_compute"],
        "mean_mlp_delta_prediction_pearson": aggregate["prediction"]["mean_pearson"],
        "oracle_headroom_consistency": aggregate["oracle"],
        "replication_verdict": aggregate["replication_verdict"],
    }
    write_json(output_dir / "summary.json", final)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
