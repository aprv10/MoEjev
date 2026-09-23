from __future__ import annotations

"""Read-only mechanism analysis of the frozen Stage 6 validation decisions."""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.stage4_compute_gate import feature_matrix, make_gate, predict_gate  # noqa: E402
from evaluation.stage3_benchmark import evaluate_policy, preload_batches  # noqa: E402
from evaluation.stage6_larger_moe import load_model  # noqa: E402
from evaluation.oracle import checkpoint_parameter_fingerprint  # noqa: E402
from data.dataset import TokenBlockDataset  # noqa: E402
from moe.adaptive_routing import RoutingPolicy  # noqa: E402
from moe.compute_gate import GateFeatureSpec  # noqa: E402

BUDGETS = (1.1, 1.2, 1.3)
HIGH_VALUE_FRACTIONS = (0.05, 0.10, 0.20)
COLORS = {"MLP": "#277da1", "margin": "#e76f51"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage6-dir",
        type=Path,
        default=ROOT / "results" / "stage6-larger-moe-20260922-234631",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def top_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, math.ceil(len(values) * fraction))
    indices = np.argsort(-values, kind="stable")[:count]
    mask = np.zeros(len(values), dtype=bool)
    mask[indices] = True
    return mask


def rank_correlation(score: np.ndarray, target: np.ndarray) -> float:
    score_rank = pd.Series(score).rank(method="average").to_numpy()
    target_rank = pd.Series(target).rank(method="average").to_numpy()
    return float(np.corrcoef(score_rank, target_rank)[0, 1])


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "positive_fraction": float(np.mean(values > 0)),
    }


def selected_metrics(target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if not np.any(mask):
        raise ValueError("A focus operating point selected no K=2 decisions")
    chosen = target[mask]
    answer: dict[str, Any] = {
        "decisions": len(target),
        "selected_count": int(mask.sum()),
        "fraction_k2": float(mask.mean()),
        "experts_per_token": float(1 + mask.mean()),
        "precision_positive_delta": float(np.mean(chosen > 0)),
        "mean_selected_delta": float(np.mean(chosen)),
        "median_selected_delta": float(np.median(chosen)),
        "p90_selected_delta": float(np.quantile(chosen, 0.9)),
        "captured_delta_per_decision": float(np.sum(chosen) / len(target)),
    }
    for fraction in HIGH_VALUE_FRACTIONS:
        high = top_mask(target, fraction)
        name = f"top{int(fraction * 100)}"
        answer[f"recall_{name}"] = float(np.sum(mask & high) / high.sum())
        answer[f"precision_{name}"] = float(np.sum(mask & high) / mask.sum())
    return answer


def set_metrics(target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    chosen = target[mask]
    return {
        "count": int(mask.sum()),
        "mean_delta": float(np.mean(chosen)) if len(chosen) else None,
        "median_delta": float(np.median(chosen)) if len(chosen) else None,
        "p90_delta": float(np.quantile(chosen, 0.9)) if len(chosen) else None,
        "fraction_k2_helps": float(np.mean(chosen > 0)) if len(chosen) else None,
    }


def byte_display(byte: int) -> str:
    return repr(chr(byte)) if 32 <= byte <= 126 else f"0x{byte:02x}"


def diagnostic_examples(
    target: np.ndarray,
    mlp_mask: np.ndarray,
    margin_mask: np.ndarray,
    prediction: np.ndarray,
    data: dict[str, np.ndarray],
    validation_bytes: np.ndarray | None,
    base_seed: int,
    gate_seed: int,
    sequence_length: int,
) -> list[dict[str, Any]]:
    output = []
    for name, mask in (
        ("MLP selects, margin misses", mlp_mask & ~margin_mask),
        ("margin selects, MLP misses", margin_mask & ~mlp_mask),
    ):
        positive = np.flatnonzero(mask & (target > 0))
        for index in positive[np.argsort(-target[positive], kind="stable")[:2]]:
            sequence = int(data["sequence_id"][index])
            position = int(data["position"][index])
            byte_index = sequence * sequence_length + position
            byte = (
                byte_display(int(validation_bytes[byte_index]))
                if validation_bytes is not None and byte_index < len(validation_bytes)
                else "unavailable"
            )
            output.append(
                {
                    "base_seed": base_seed,
                    "gate_seed": gate_seed,
                    "layer": int(data["layer"][index]),
                    "sequence_id": sequence,
                    "position": position,
                    "byte": byte,
                    "router_top1_probability": float(data["router"][index, 0]),
                    "router_top2_probability": float(data["router"][index, 1]),
                    "router_margin": float(data["router"][index, 2]),
                    "mlp_predicted_delta": float(prediction[index]),
                    "true_delta": float(target[index]),
                    "mlp_k2": bool(mlp_mask[index]),
                    "margin_k2": bool(margin_mask[index]),
                    "example_type": name,
                }
            )
    return output


def layer_mask(data: dict[str, np.ndarray], selected: np.ndarray,
               layer: int, num_sequences: int, sequence_length: int) -> torch.Tensor:
    rows = np.flatnonzero(data["layer"] == layer)
    output = np.zeros((num_sequences, sequence_length), dtype=bool)
    output[data["sequence_id"][rows].astype(np.int64),
           data["position"][rows].astype(np.int64)] = selected[rows]
    return torch.from_numpy(output)


def plot_all(output: Path, recovery: pd.DataFrame, layers: pd.DataFrame,
             increments: pd.DataFrame, capture: pd.DataFrame,
             sets: pd.DataFrame, seeds: pd.DataFrame,
             online_layers: pd.DataFrame) -> None:
    plots = output / "plots"
    plots.mkdir(exist_ok=True)

    # Each base model contributes one mean across its three gate seeds.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric, ylabel in (
        (axes[0], "recall_top5", "Recall of true top 5%"),
        (axes[1], "recall_top10", "Recall of true top 10%"),
    ):
        for method in ("MLP", "margin"):
            group = recovery[recovery.method == method].groupby("budget")[metric]
            x = np.array(sorted(group.groups))
            y = np.array([group.get_group(b).mean() for b in x])
            ax.plot(x, y, "o-", lw=2.3, color=COLORS[method], label=method)
            for base_seed, sub in recovery[recovery.method == method].groupby("base_seed"):
                by_budget = sub.groupby("budget")[metric].mean()
                ax.plot(by_budget.index, by_budget.values, color=COLORS[method], alpha=0.22, lw=1)
        ax.set(xlabel="Target experts / token", ylabel=ylabel)
        ax.set_xticks(BUDGETS)
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False)
    fig.suptitle("Recovery of the highest-value second-expert decisions", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots / "high_value_recall.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for method in ("MLP", "margin"):
        group = recovery[recovery.method == method].groupby("budget")
        for ax, metric, ylabel in (
            (axes[0], "mean_selected_delta", "Mean true delta among K=2"),
            (axes[1], "precision_positive_delta", "Fraction of K=2 where delta > 0"),
        ):
            summary = group[metric].mean()
            ax.plot(summary.index, summary.values, "o-", lw=2.3, color=COLORS[method], label=method)
            ax.set(xlabel="Target experts / token", ylabel=ylabel)
            ax.set_xticks(BUDGETS)
            ax.grid(alpha=0.18)
    axes[0].legend(frameon=False)
    fig.suptitle("Value of decisions receiving a second expert", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots / "selected_token_delta_comparison.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))
    subset = layers[layers.budget == 1.1]
    per_base = subset.groupby(["base_seed", "layer"], as_index=False).agg(
        recall_difference=("recall_top10_difference", "mean"),
    )
    per_base = per_base.merge(
        online_layers.groupby(["base_seed", "layer"], as_index=False)
        .mlp_minus_margin_loss.mean(), on=["base_seed", "layer"], validate="one_to_one"
    )
    for base_seed, group in per_base.groupby("base_seed"):
        axes[0].plot(group.layer, group.mlp_minus_margin_loss, "o-", label=f"MoE {base_seed}")
        axes[1].plot(group.layer, group.recall_difference, "o-", label=f"MoE {base_seed}")
    axes[0].axhline(0, color="0.35", lw=1)
    axes[1].axhline(0, color="0.35", lw=1)
    axes[0].set(ylabel="Single-layer validation loss: MLP − margin")
    axes[1].set(ylabel="Layer-local top 10% recall: MLP − margin")
    for ax in axes:
        ax.set(xlabel="MoE layer (zero based)")
        ax.set_xticks(sorted(per_base.layer.unique()))
        ax.grid(alpha=0.18)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Layer contributions near 1.10 experts / token", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots / "per_layer_advantage.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for method in ("MLP", "margin"):
        by_fraction = capture[capture.method == method].groupby("fraction_k2")
        axes[0].plot(
            sorted(by_fraction.groups),
            [by_fraction.get_group(f).captured_delta_per_decision.mean() for f in sorted(by_fraction.groups)],
            "-", lw=2.3, color=COLORS[method], label=method,
        )
        part = increments[increments.method == method]
        summary = part.groupby("increment").mean_true_delta.mean()
        axes[1].plot(summary.index, summary.values, "o-", lw=2.3, color=COLORS[method], label=method)
    axes[0].set(xlabel="Fraction receiving K=2", ylabel="Cumulative true delta captured / decision")
    axes[1].set(xlabel="Newly admitted budget band", ylabel="Mean true delta in band")
    axes[0].set_xlim(0, 0.32)
    for ax in axes:
        ax.grid(alpha=0.18)
    axes[1].tick_params(axis="x", rotation=20)
    axes[0].legend(frameon=False)
    fig.suptitle("What each additional second-expert budget buys", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots / "budget_capture_curve.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.3, 7.4))
    for base_seed, part in seeds.groupby("base_seed"):
        axes[0, 0].scatter(base_seed, part.iloc[0].delta_p95, s=80, label=str(base_seed))
        axes[0, 1].scatter(base_seed, part.iloc[0].router_entropy_mean, s=80)
    for base_seed, part in recovery.groupby("base_seed"):
        means = part.groupby(["budget", "method"]).recall_top10.mean().unstack()
        axes[1, 0].plot(means.index, means.MLP - means.margin, "o-", label=f"MoE {base_seed}")
        own = sets[(sets.base_seed == base_seed) & (sets.group.isin(["MLP only", "margin only"]))]
        piv = own.groupby(["budget", "group"]).mean_delta.mean().unstack()
        axes[1, 1].plot(piv.index, piv["MLP only"] - piv["margin only"], "o-", label=f"MoE {base_seed}")
    axes[0, 0].set(ylabel="True delta P95", xlabel="Base seed")
    axes[0, 1].set(ylabel="Router entropy mean", xlabel="Base seed")
    axes[1, 0].set(ylabel="Top 10% recall difference", xlabel="Target experts / token")
    axes[1, 1].set(ylabel="Unique-set mean delta difference", xlabel="Target experts / token")
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    axes[1, 0].axhline(0, color="0.4", lw=1)
    axes[1, 1].axhline(0, color="0.4", lw=1)
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Seed 602 beside the other two base models", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots / "seed602_diagnostic.png", dpi=190)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    stage6 = args.stage6_dir.resolve()
    if not (stage6 / "summary.json").is_file():
        raise FileNotFoundError(f"Stage 6 results not found: {stage6}")
    output = (args.output_dir or ROOT / "results" / f"stage6_5-mechanism-{datetime.now():%Y%m%d-%H%M%S}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    matched = pd.read_csv(stage6 / "aggregate" / "matched_compute.csv")
    stage6_config = json_read(stage6 / "config.json")
    base_config = json_read(stage6 / "moe_seed_601" / "config.resolved.json")
    model_spec = base_config["model"]
    spec = GateFeatureSpec(True, int(model_spec["model_dim"]), tuple(model_spec["moe_layers"]))
    sequence_length = int(model_spec["sequence_length"])
    val_path = ROOT / "data" / "cache" / f"wikitext2_validation_{base_config['data']['max_validation_tokens']}_bytes.npy"
    validation_bytes = np.load(val_path, mmap_mode="r") if val_path.exists() else None

    recovery_rows: list[dict[str, Any]] = []
    set_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    increment_rows: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    online_layer_rows: list[dict[str, Any]] = []

    if validation_bytes is None:
        raise FileNotFoundError(f"Stage 6 validation byte cache is required: {val_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_dataset = TokenBlockDataset(val_path, sequence_length)
    loader = DataLoader(validation_dataset, batch_size=int(stage6_config["data"]["batch_size"]),
                        shuffle=False, num_workers=0)
    batches = preload_batches(loader, device, int(stage6_config["data"]["validation_batches"]))
    num_sequences = sum(inputs.size(0) for inputs, _ in batches)
    use_amp = device.type == "cuda" and stage6_config["benchmark"]["mixed_precision"] == "fp16"

    for base_seed in stage6_config["experiment"]["base_moe_seeds"]:
        seed_dir = stage6 / f"moe_seed_{base_seed}"
        model, _ = load_model(seed_dir / "checkpoint.pt", device)
        fingerprint = checkpoint_parameter_fingerprint(model)
        cache = seed_dir / "feature_cache" / "final_validation"
        data = {name: np.load(cache / f"{name}.npy", allow_pickle=False)
                for name in ("hidden", "router", "delta", "layer", "sequence_id", "position")}
        features = feature_matrix(data, spec)
        target = data["delta"].astype(np.float64)
        margin_score = -data["router"][:, 2].astype(np.float64)
        entropy = data["router"][:, 3].astype(np.float64)
        seed_rows.append({
            "base_seed": base_seed,
            "decisions": len(target),
            **{f"delta_{key}": value for key, value in distribution(target).items()},
            "router_entropy_mean": float(entropy.mean()),
            "router_entropy_median": float(np.median(entropy)),
        })
        margin_rank = rank_correlation(margin_score, target)
        margin_masks = {fraction: top_mask(margin_score, fraction) for fraction in HIGH_VALUE_FRACTIONS}
        true_masks = {fraction: top_mask(target, fraction) for fraction in HIGH_VALUE_FRACTIONS}
        ranking_rows.append({
            "base_seed": base_seed, "gate_seed": None, "method": "margin",
            "spearman": margin_rank,
            **{f"top{int(fraction * 100)}_overlap": float(np.mean(margin_masks[fraction][true_masks[fraction]]))
               for fraction in HIGH_VALUE_FRACTIONS},
        })
        margin_layer_losses: dict[int, float] = {}
        for gate_seed in stage6_config["experiment"]["gate_seeds"]:
            checkpoint = torch.load(seed_dir / "gate_runs" / f"seed_{gate_seed}" / "gate.pt",
                                    map_location="cpu", weights_only=False)
            gate = make_gate("mlp", spec.input_dim, int(stage6_config["gate"]["hidden_dim"]))
            gate.load_state_dict(checkpoint["state_dict"], strict=True)
            prediction = predict_gate(gate, features, checkpoint["feature_mean"], checkpoint["feature_std"], torch.device("cpu"))
            del gate
            rank_masks = {fraction: top_mask(prediction, fraction) for fraction in HIGH_VALUE_FRACTIONS}
            ranking_rows.append({
                "base_seed": base_seed, "gate_seed": gate_seed, "method": "MLP",
                "spearman": rank_correlation(prediction, target),
                **{f"top{int(fraction * 100)}_overlap": float(np.mean(rank_masks[fraction][true_masks[fraction]]))
                   for fraction in HIGH_VALUE_FRACTIONS},
            })
            masks_by_budget: dict[str, dict[float, np.ndarray]] = {"MLP": {}, "margin": {}}
            for budget in BUDGETS:
                point = matched[(matched.base_seed == base_seed) & (matched.gate_seed == gate_seed) &
                                (np.isclose(matched.target_experts_per_token, budget))].iloc[0]
                mlp_mask = prediction > float(point.mlp_threshold)
                margin_mask = data["router"][:, 2] < float(point.margin_threshold)
                masks_by_budget["MLP"][budget] = mlp_mask
                masks_by_budget["margin"][budget] = margin_mask
                for method, mask in (("MLP", mlp_mask), ("margin", margin_mask)):
                    recovery_rows.append({"base_seed": base_seed, "gate_seed": gate_seed,
                                          "budget": budget, "method": method,
                                          **selected_metrics(target, mask)})
                    online_fraction = float(point.mlp_actual_experts_per_token - 1) if method == "MLP" else float(point.margin_actual_experts_per_token - 1)
                    agreement_rows.append({"base_seed": base_seed, "gate_seed": gate_seed, "budget": budget,
                                           "method": method, "cached_fraction_k2": float(mask.mean()),
                                           "online_fraction_k2": online_fraction,
                                           "cached_minus_online": float(mask.mean() - online_fraction)})
                groups = {
                    "both": mlp_mask & margin_mask,
                    "MLP only": mlp_mask & ~margin_mask,
                    "margin only": ~mlp_mask & margin_mask,
                    "neither": ~mlp_mask & ~margin_mask,
                }
                assert sum(int(group.sum()) for group in groups.values()) == len(target)
                for name, mask in groups.items():
                    set_rows.append({"base_seed": base_seed, "gate_seed": gate_seed,
                                     "budget": budget, "group": name, **set_metrics(target, mask)})
                for layer in spec.moe_layers:
                    in_layer = data["layer"] == layer
                    layer_delta = target[in_layer]
                    local_mlp = mlp_mask[in_layer]
                    local_margin = margin_mask[in_layer]
                    mlp_metrics = selected_metrics(layer_delta, local_mlp)
                    margin_metrics = selected_metrics(layer_delta, local_margin)
                    # The isolated per-token deltas provide an attribution proxy.
                    # Joint online changes across layers are not additive.
                    layer_rows.append({
                        "base_seed": base_seed, "gate_seed": gate_seed, "budget": budget,
                        "layer": layer, "decisions": int(in_layer.sum()),
                        "proxy_mlp_minus_margin": float((np.sum(layer_delta[local_margin]) - np.sum(layer_delta[local_mlp])) / len(layer_delta)),
                        "mlp_fraction_k2": mlp_metrics["fraction_k2"],
                        "margin_fraction_k2": margin_metrics["fraction_k2"],
                        "mlp_recall_top5": mlp_metrics["recall_top5"],
                        "margin_recall_top5": margin_metrics["recall_top5"],
                        "mlp_recall_top10": mlp_metrics["recall_top10"],
                        "margin_recall_top10": margin_metrics["recall_top10"],
                        "recall_top10_difference": mlp_metrics["recall_top10"] - margin_metrics["recall_top10"],
                        "mlp_mean_selected_delta": mlp_metrics["mean_selected_delta"],
                        "margin_mean_selected_delta": margin_metrics["mean_selected_delta"],
                        "layer_delta_mean": float(layer_delta.mean()),
                        "layer_delta_p90": float(np.quantile(layer_delta, 0.9)),
                        "layer_entropy_mean": float(entropy[in_layer].mean()),
                    })
                if budget == 1.1 and gate_seed == 11:
                    example_rows.extend(diagnostic_examples(target, mlp_mask, margin_mask,
                                                            prediction, data, validation_bytes,
                                                            base_seed, gate_seed, sequence_length))
                if budget == 1.1:
                    for layer in spec.moe_layers:
                        if layer not in margin_layer_losses:
                            margin_override = {layer: layer_mask(data, margin_mask, layer, num_sequences, sequence_length)}
                            margin_layer_losses[layer] = evaluate_policy(
                                model, batches, RoutingPolicy("top2", fixed_k=2), use_amp,
                                margin_override,
                            )["loss"]
                        mlp_override = {layer: layer_mask(data, mlp_mask, layer, num_sequences, sequence_length)}
                        mlp_loss = evaluate_policy(
                            model, batches, RoutingPolicy("top2", fixed_k=2), use_amp,
                            mlp_override,
                        )["loss"]
                        online_layer_rows.append({
                            "base_seed": base_seed, "gate_seed": gate_seed,
                            "budget": budget, "layer": layer,
                            "mlp_single_layer_validation_loss": mlp_loss,
                            "margin_single_layer_validation_loss": margin_layer_losses[layer],
                            "mlp_minus_margin_loss": mlp_loss - margin_layer_losses[layer],
                        })

            for method in ("MLP", "margin"):
                previous = np.zeros(len(target), dtype=bool)
                for budget in BUDGETS:
                    selected = masks_by_budget[method][budget]
                    admitted = selected & ~previous
                    assert not np.any(previous & ~selected), "Budget masks are not nested"
                    increment_rows.append({
                        "base_seed": base_seed, "gate_seed": gate_seed,
                        "method": method,
                        "increment": f"{1 if budget == 1.1 else budget - 0.1:.1f} → {budget:.1f}",
                        "budget": budget, "new_count": int(admitted.sum()),
                        "new_fraction": float(admitted.mean()),
                        "mean_true_delta": float(target[admitted].mean()),
                        "positive_fraction": float(np.mean(target[admitted] > 0)),
                    })
                    previous = selected
                score = prediction if method == "MLP" else margin_score
                ordering = np.argsort(-score, kind="stable")
                cumulative = np.cumsum(target[ordering])
                for fraction in np.linspace(0.01, 0.31, 31):
                    count = max(1, math.ceil(len(target) * fraction))
                    capture_rows.append({"base_seed": base_seed, "gate_seed": gate_seed,
                                         "method": method, "fraction_k2": round(float(fraction), 4),
                                         "captured_delta_per_decision": float(cumulative[count - 1] / len(target)),
                                         "mean_selected_delta": float(cumulative[count - 1] / count)})
        if checkpoint_parameter_fingerprint(model) != fingerprint:
            raise RuntimeError(f"Frozen base MoE changed during analysis: {base_seed}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"analyzed_base_seed": base_seed, "decisions": len(target)}), flush=True)

    frames = {
        "high_value_recovery": pd.DataFrame(recovery_rows),
        "selected_sets": pd.DataFrame(set_rows),
        "per_layer": pd.DataFrame(layer_rows),
        "budget_increments": pd.DataFrame(increment_rows),
        "capture_curve": pd.DataFrame(capture_rows),
        "ranking": pd.DataFrame(ranking_rows),
        "seed_diagnostics": pd.DataFrame(seed_rows),
        "examples": pd.DataFrame(example_rows),
        "cache_online_agreement": pd.DataFrame(agreement_rows),
        "per_layer_online": pd.DataFrame(online_layer_rows),
    }
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    plot_all(output, frames["high_value_recovery"], frames["per_layer"],
             frames["budget_increments"], frames["capture_curve"],
             frames["selected_sets"], frames["seed_diagnostics"],
             frames["per_layer_online"])
    summary = {
        "source_stage6": str(stage6),
        "output_dir": str(output),
        "base_seeds": stage6_config["experiment"]["base_moe_seeds"],
        "gate_seeds": stage6_config["experiment"]["gate_seeds"],
        "budgets": list(BUDGETS),
        "target_definition": "isolated top1 unit-weight loss minus normal top2 loss at the same token, with other layer decisions at the original Top2 baseline",
        "analysis_context": "cached final-validation pre-MoE hidden and router values from the original fixed-Top2 pass",
        "cache_online_max_fraction_gap": float(frames["cache_online_agreement"].cached_minus_online.abs().max()),
        "high_value_definition": "true delta_loss ranked over all token-layer decisions within a base model",
        "precision_definition": "precision_topX is the fraction of K=2 choices in the true top X%; precision_positive_delta is the fraction with true delta > 0",
        "per_layer_difference_definition": "negative isolated proxy MLP-minus-margin loss means MLP captures more true delta; it is not the joint online model loss",
        "per_layer_online_definition": "measured validation loss with exactly one MoE layer using a cached K=2 mask and the other MoE layers fixed Top2",
    }
    json_write(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
