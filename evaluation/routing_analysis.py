from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from evaluation.calibration_analysis import confidence_bins, distribution_alignment


def _percentiles(values: np.ndarray, points: list[int]) -> dict[str, float]:
    return {
        f"p{point}": float(np.percentile(values, point)) for point in points
    }


def regret_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        **_percentiles(values, [90, 95, 99]),
        "maximum": float(values.max()),
    }


def gap_statistics(values: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        **_percentiles(values, [25, 50, 75, 90, 95, 99]),
        "threshold_fractions": {},
    }
    for threshold in [0.001, 0.01, 0.05, 0.1]:
        output["threshold_fractions"][f"less_than_{threshold}"] = float(
            (values < threshold).mean()
        )
    for threshold in [0.1, 0.25, 0.5]:
        output["threshold_fractions"][f"greater_than_{threshold}"] = float(
            (values > threshold).mean()
        )
    return output


def _utilization(values: np.ndarray, num_experts: int) -> list[float]:
    counts = np.bincount(values.astype(np.int64), minlength=num_experts)
    return (counts / max(counts.sum(), 1)).tolist()


def compute_metrics(
    frame: pd.DataFrame,
    confidence_edges: list[float],
    num_experts: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    per_layer: dict[str, Any] = {}
    confidence_output: dict[str, Any] = {}
    utilization_output: dict[str, Any] = {}
    groups: list[tuple[str, pd.DataFrame]] = [
        (str(int(layer)), layer_frame) for layer, layer_frame in frame.groupby("layer")
    ]
    groups.append(("aggregate", frame))
    for name, group in groups:
        top1 = group["router_top1"].to_numpy(dtype=np.int64)
        oracle = group["oracle_expert"].to_numpy(dtype=np.int64)
        top2 = np.stack(group["router_top2"].to_numpy()).astype(np.int64)
        correct = top1 == oracle
        covered = np.any(top2 == oracle[:, None], axis=1)
        regret = group["top1_regret"].to_numpy(dtype=np.float64)
        mixture_delta = group["top2_mixture_delta"].to_numpy(dtype=np.float64)
        gaps = group["loss_gap"].to_numpy(dtype=np.float64)
        confidence_values = group["router_confidence"].to_numpy(dtype=np.float64)
        confusion = np.zeros((num_experts, num_experts), dtype=np.int64)
        np.add.at(confusion, (top1, oracle), 1)
        bins = confidence_bins(group, confidence_edges)
        nonempty_bins = [item for item in bins if item["count"]]
        exploratory_ece = sum(
            item["count"]
            / len(group)
            * abs(item["mean_router_confidence"] - item["oracle_top1_accuracy"])
            for item in nonempty_bins
        )
        per_layer[name] = {
            "evaluated_token_layer_decisions": int(len(group)),
            "router_top1_oracle_accuracy": float(correct.mean()),
            "router_top2_oracle_coverage": float(covered.mean()),
            "top1_regret": regret_statistics(regret),
            "top2_mixture_minus_best_single": regret_statistics(mixture_delta),
            "best_vs_second_gap": gap_statistics(gaps),
            "mean_router_entropy": float(group["router_entropy"].mean()),
            "mean_oracle_entropy": float(group["oracle_entropy"].mean()),
            "mean_baseline_top2_loss": float(group["baseline_top2_loss"].mean()),
            "fraction_top2_mixture_beats_best_single": float(
                (mixture_delta < 0).mean()
            ),
            "confidence_correctness_correlation": float(
                np.corrcoef(confidence_values, correct.astype(np.float64))[0, 1]
            ),
            "confidence_regret_correlation": float(
                np.corrcoef(confidence_values, regret)[0, 1]
            ),
            "router_top1_vs_oracle_confusion": confusion.tolist(),
            "distribution_alignment": distribution_alignment(group),
            "exploratory_confidence_ece": float(exploratory_ece),
        }
        confidence_output[name] = bins
        utilization_output[name] = {
            "oracle_best": _utilization(oracle, num_experts),
            "router_top1": _utilization(top1, num_experts),
            "router_top2_assignments": _utilization(top2.reshape(-1), num_experts),
        }

    gaps = frame["loss_gap"].to_numpy(dtype=np.float64)
    q1, q3 = np.percentile(gaps, [25, 75])
    threshold = float(q3 + 1.5 * (q3 - q1))
    strong_mask = gaps > threshold
    threshold_method = "Tukey upper fence (Q3 + 1.5 * IQR)"
    if not strong_mask.any():
        threshold = float(np.percentile(gaps, 95))
        strong_mask = gaps >= threshold
        threshold_method = "P95 fallback because the Tukey set was empty"
    strong = frame.loc[strong_mask]
    strong_top2 = np.stack(strong["router_top2"].to_numpy()).astype(np.int64)
    strong_oracle = strong["oracle_expert"].to_numpy(dtype=np.int64)
    strong_summary = {
        "threshold": threshold,
        "threshold_method": threshold_method,
        "count": int(len(strong)),
        "fraction": float(len(strong) / len(frame)),
        "router_top1_oracle_accuracy": float(
            (strong["router_top1"].to_numpy() == strong_oracle).mean()
        ),
        "router_top2_oracle_coverage": float(
            np.any(strong_top2 == strong_oracle[:, None], axis=1).mean()
        ),
        "mean_top1_regret": float(strong["top1_regret"].mean()),
        "oracle_expert_utilization": _utilization(strong_oracle, num_experts),
        "per_layer": {},
    }
    for layer, layer_frame in frame.groupby("layer"):
        layer_strong = layer_frame[layer_frame["loss_gap"] > threshold]
        layer_oracle = layer_strong["oracle_expert"].to_numpy(dtype=np.int64)
        layer_top2 = np.stack(layer_strong["router_top2"].to_numpy()).astype(
            np.int64
        )
        strong_summary["per_layer"][str(int(layer))] = {
            "count": int(len(layer_strong)),
            "fraction": float(len(layer_strong) / len(layer_frame)),
            "router_top1_oracle_accuracy": float(
                (layer_strong["router_top1"].to_numpy() == layer_oracle).mean()
            ),
            "router_top2_oracle_coverage": float(
                np.any(layer_top2 == layer_oracle[:, None], axis=1).mean()
            ),
            "mean_top1_regret": float(layer_strong["top1_regret"].mean()),
        }
    return per_layer, confidence_output, utilization_output, strong_summary


def select_case_studies(frame: pd.DataFrame, limit: int = 25) -> dict[str, Any]:
    wrong = frame[frame["router_top1"] != frame["oracle_expert"]].copy()
    confidence_threshold = float(wrong["router_confidence"].quantile(0.9))
    regret_threshold = float(wrong["top1_regret"].quantile(0.9))
    confident = wrong[
        (wrong["router_confidence"] >= confidence_threshold)
        & (wrong["top1_regret"] >= regret_threshold)
    ].copy()
    confident["ranking_score"] = (
        confident["router_confidence"] * confident["top1_regret"]
    )
    uncertain_threshold = float(frame["router_confidence"].quantile(0.1))
    uncertain = frame[
        (frame["router_top1"] == frame["oracle_expert"])
        & (frame["router_confidence"] <= uncertain_threshold)
    ].sort_values(["router_confidence", "loss_gap"], ascending=[True, False])
    fields = [
        "layer",
        "sequence_id",
        "position",
        "context",
        "input_byte",
        "target_byte",
        "router_probabilities",
        "expert_losses",
        "router_top1",
        "router_top2",
        "oracle_expert",
        "router_confidence",
        "loss_gap",
        "top1_regret",
        "top2_mixture_delta",
    ]
    return {
        "selection": {
            "confident_mistake_confidence_threshold_p90_among_mistakes": confidence_threshold,
            "confident_mistake_regret_threshold_p90_among_mistakes": regret_threshold,
            "uncertain_correct_confidence_threshold_global_p10": uncertain_threshold,
        },
        "confident_mistakes": confident.sort_values(
            "ranking_score", ascending=False
        ).head(limit)[fields].to_dict(orient="records"),
        "uncertain_but_correct": uncertain.head(limit)[fields].to_dict(
            orient="records"
        ),
    }


def _style_axis(axis: plt.Axes, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)


def create_plots(
    frame: pd.DataFrame,
    output_dir: Path,
    confidence_data: dict[str, Any],
    utilization_data: dict[str, Any],
    num_experts: int,
) -> None:
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140})
    layer_values = sorted(frame["layer"].unique())

    figure, axis = plt.subplots(figsize=(7, 4.2))
    upper = float(frame["loss_gap"].quantile(0.995))
    for layer in layer_values:
        values = frame.loc[
            (frame["layer"] == layer) & (frame["loss_gap"] <= upper),
            "loss_gap",
        ]
        axis.hist(values, bins=60, density=True, histtype="step", linewidth=1.5, label=f"Layer {layer}")
    _style_axis(axis, "Best-vs-second expert loss gap", "Density")
    axis.set_title("Central 99.5% of observed gaps")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_gap_distribution.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1, label="identity")
    for layer in layer_values:
        bins = [item for item in confidence_data[str(layer)] if item["count"] >= 20]
        axis.plot(
            [item["mean_router_confidence"] for item in bins],
            [item["oracle_top1_accuracy"] for item in bins],
            marker="o",
            label=f"Layer {layer}",
        )
    axis.set_xlim(0.2, 1.0)
    axis.set_ylim(0.0, 1.0)
    _style_axis(axis, "Mean router Top-1 confidence", "Oracle Top-1 accuracy")
    axis.set_title("Confidence bins with at least 20 decisions")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "router_confidence_reliability.png")
    plt.close(figure)

    for layer in layer_values:
        subset = frame[frame["layer"] == layer]
        matrix = np.zeros((num_experts, num_experts), dtype=np.int64)
        np.add.at(
            matrix,
            (
                subset["router_top1"].to_numpy(dtype=np.int64),
                subset["oracle_expert"].to_numpy(dtype=np.int64),
            ),
            1,
        )
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        figure, axis = plt.subplots(figsize=(5.2, 4.5))
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for row in range(num_experts):
            for column in range(num_experts):
                axis.text(
                    column,
                    row,
                    f"{normalized[row, column]:.2f}\n({matrix[row, column]})",
                    ha="center",
                    va="center",
                    color="white" if normalized[row, column] > 0.55 else "black",
                    fontsize=8,
                )
        axis.set_xticks(range(num_experts), [f"E{i + 1}" for i in range(num_experts)])
        axis.set_yticks(range(num_experts), [f"E{i + 1}" for i in range(num_experts)])
        axis.set_xlabel("Oracle-best expert")
        axis.set_ylabel("Router Top-1 expert")
        axis.set_title(f"Layer {layer}")
        figure.colorbar(image, ax=axis, label="Row fraction")
        figure.tight_layout()
        figure.savefig(output_dir / f"router_vs_oracle_confusion_layer{layer}.png")
        plt.close(figure)

    figure, axes = plt.subplots(1, len(layer_values), figsize=(6.2 * len(layer_values), 4), squeeze=False)
    x = np.arange(num_experts)
    for axis, layer in zip(axes[0], layer_values):
        values = utilization_data[str(layer)]
        axis.bar(x - 0.18, values["oracle_best"], width=0.36, label="Oracle best")
        axis.bar(x + 0.18, values["router_top1"], width=0.36, label="Router Top-1")
        axis.set_xticks(x, [f"E{i + 1}" for i in x])
        axis.set_ylim(0, max(max(values["oracle_best"]), max(values["router_top1"])) * 1.2)
        axis.set_title(f"Layer {layer}")
        _style_axis(axis, "Expert", "Fraction of tokens")
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "oracle_expert_utilization.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.2))
    upper = float(frame["top1_regret"].quantile(0.995))
    for layer in layer_values:
        values = frame.loc[
            (frame["layer"] == layer) & (frame["top1_regret"] <= upper),
            "top1_regret",
        ]
        axis.hist(values, bins=60, density=True, histtype="step", linewidth=1.5, label=f"Layer {layer}")
    _style_axis(axis, "Forced router Top-1 regret", "Density")
    axis.set_title("Central 99.5% of observed regrets")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "routing_regret_distribution.png")
    plt.close(figure)

    figure, axes = plt.subplots(
        1, len(layer_values), figsize=(6 * len(layer_values), 5), squeeze=False
    )
    for axis, layer in zip(axes[0], layer_values):
        subset = frame[frame["layer"] == layer]
        density = axis.hexbin(
            subset["router_entropy"],
            subset["oracle_entropy"],
            gridsize=35,
            mincnt=1,
            bins="log",
            cmap="viridis",
        )
        limit = np.log(num_experts)
        axis.plot([0, limit], [0, limit], "--", color="0.35", linewidth=1)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_title(f"Layer {layer}")
        _style_axis(axis, "Router entropy (nats)", "Soft-oracle entropy (nats)")
        figure.colorbar(density, ax=axis, label="log count")
    figure.tight_layout()
    figure.savefig(output_dir / "router_entropy_vs_oracle_entropy.png")
    plt.close(figure)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
