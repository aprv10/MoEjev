from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def scale_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    log_probabilities -= log_probabilities.max(axis=1, keepdims=True)
    scaled = np.exp(log_probabilities)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _temperature_objective(
    log_temperature: float,
    probabilities: np.ndarray,
    hard_targets: np.ndarray | None,
    soft_targets: np.ndarray | None,
) -> float:
    scaled = scale_probabilities(probabilities, math.exp(log_temperature))
    if hard_targets is not None:
        return float(
            -np.log(np.clip(scaled[np.arange(len(scaled)), hard_targets], 1e-12, 1.0)).mean()
        )
    if soft_targets is None:
        raise ValueError("One target form is required")
    return float(
        -np.sum(soft_targets * np.log(np.clip(scaled, 1e-12, 1.0)), axis=1).mean()
    )


def fit_temperature(
    probabilities: np.ndarray,
    hard_targets: np.ndarray | None = None,
    soft_targets: np.ndarray | None = None,
    minimum: float = 0.05,
    maximum: float = 20.0,
    iterations: int = 100,
) -> dict[str, float]:
    """Fit a scalar temperature by bounded golden-section search in log space."""
    if (hard_targets is None) == (soft_targets is None):
        raise ValueError("Provide exactly one of hard_targets or soft_targets")
    lower = math.log(minimum)
    upper = math.log(maximum)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value = _temperature_objective(
        left, probabilities, hard_targets, soft_targets
    )
    right_value = _temperature_objective(
        right, probabilities, hard_targets, soft_targets
    )
    for _ in range(iterations):
        if left_value < right_value:
            upper = right
            right = left
            right_value = left_value
            left = upper - ratio * (upper - lower)
            left_value = _temperature_objective(
                left, probabilities, hard_targets, soft_targets
            )
        else:
            lower = left
            left = right
            left_value = right_value
            right = lower + ratio * (upper - lower)
            right_value = _temperature_objective(
                right, probabilities, hard_targets, soft_targets
            )
    best_log_temperature = (lower + upper) / 2.0
    return {
        "temperature": math.exp(best_log_temperature),
        "objective_before": _temperature_objective(
            0.0, probabilities, hard_targets, soft_targets
        ),
        "objective_after": _temperature_objective(
            best_log_temperature, probabilities, hard_targets, soft_targets
        ),
    }


def calibration_metrics(
    frame: pd.DataFrame,
    temperatures: dict[int, float],
    num_experts: int,
    num_bins: int = 15,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    metrics: dict[str, Any] = {}
    reliability: dict[str, list[dict[str, Any]]] = {}
    groups = [
        (str(int(layer)), layer_frame) for layer, layer_frame in frame.groupby("layer")
    ]
    groups.append(("aggregate", frame))
    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_soft_targets: list[np.ndarray] = []
    all_regrets: list[np.ndarray] = []
    for name, group in groups:
        if name == "aggregate":
            probabilities = np.concatenate(all_probabilities)
            targets = np.concatenate(all_targets)
            soft_targets = np.concatenate(all_soft_targets)
            regrets = np.concatenate(all_regrets)
        else:
            layer = int(name)
            raw = np.stack(group["router_probabilities"].to_numpy()).astype(np.float64)
            probabilities = scale_probabilities(raw, temperatures.get(layer, 1.0))
            targets = group["oracle_expert"].to_numpy(dtype=np.int64)
            soft_targets = np.stack(
                group["soft_oracle_distribution"].to_numpy()
            ).astype(np.float64)
            regrets = group["top1_regret"].to_numpy(dtype=np.float64)
            all_probabilities.append(probabilities)
            all_targets.append(targets)
            all_soft_targets.append(soft_targets)
            all_regrets.append(regrets)
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        correct = predictions == targets
        one_hot = np.eye(num_experts, dtype=np.float64)[targets]
        nll = -np.log(
            np.clip(probabilities[np.arange(len(probabilities)), targets], 1e-12, 1.0)
        ).mean()
        brier = np.square(probabilities - one_hot).sum(axis=1).mean()
        soft_brier = np.square(probabilities - soft_targets).sum(axis=1).mean()
        soft_cross_entropy = -np.sum(
            soft_targets * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1
        ).mean()
        edges = np.linspace(1.0 / num_experts, 1.0, num_bins + 1)
        bins: list[dict[str, Any]] = []
        ece = 0.0
        for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (confidence >= lower) & (
                confidence <= upper if bin_index == num_bins - 1 else confidence < upper
            )
            count = int(mask.sum())
            bin_data = {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_confidence": float(confidence[mask].mean()) if count else None,
                "accuracy": float(correct[mask].mean()) if count else None,
            }
            if count:
                ece += count / len(confidence) * abs(
                    bin_data["mean_confidence"] - bin_data["accuracy"]
                )
            bins.append(bin_data)
        metrics[name] = {
            "ece": float(ece),
            "brier": float(brier),
            "nll": float(nll),
            "soft_oracle_brier": float(soft_brier),
            "soft_oracle_cross_entropy": float(soft_cross_entropy),
            "top1_accuracy": float(correct.mean()),
            "mean_confidence": float(confidence.mean()),
            "confidence_correctness_correlation": float(
                np.corrcoef(confidence, correct.astype(np.float64))[0, 1]
            ),
            "confidence_regret_correlation": float(
                np.corrcoef(confidence, regrets)[0, 1]
            ),
        }
        reliability[name] = bins
    return metrics, reliability

