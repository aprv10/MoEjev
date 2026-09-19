from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def byte_category(value: int) -> str:
    if value < 0:
        return "sequence_boundary"
    if value == 10:
        return "newline"
    if value >= 128:
        return "non_ascii_utf8"
    character = chr(value)
    if character.isspace():
        return "whitespace"
    if character.isalpha():
        return "alphabetic"
    if character.isdigit():
        return "digit"
    if character.isprintable() and not character.isalnum():
        return "punctuation"
    if value < 32 or value == 127:
        return "control"
    return "other_ascii"


def position_bucket(position: int, sequence_length: int) -> str:
    fraction = position / sequence_length
    if fraction < 0.25:
        return "first_quarter"
    if fraction < 0.5:
        return "second_quarter"
    if fraction < 0.75:
        return "third_quarter"
    return "fourth_quarter"


def _distribution(values: pd.Series) -> dict[str, float]:
    counts = values.value_counts()
    total = max(int(counts.sum()), 1)
    return {str(key): float(value / total) for key, value in counts.items()}


def analyze_specialization(
    frame: pd.DataFrame, num_experts: int, sequence_length: int
) -> dict[str, Any]:
    enriched = frame.copy()
    enriched["input_category"] = enriched["input_byte"].map(byte_category)
    enriched["target_category"] = enriched["target_byte"].map(byte_category)
    enriched["preceding_category"] = enriched["preceding_byte"].map(byte_category)
    enriched["following_category"] = enriched["following_byte"].map(byte_category)
    enriched["position_bucket"] = enriched["position"].map(
        lambda value: position_bucket(int(value), sequence_length)
    )
    features = [
        "input_category",
        "target_category",
        "preceding_category",
        "following_category",
        "position_bucket",
    ]
    result: dict[str, Any] = {"layers": {}}
    for layer, layer_frame in enriched.groupby("layer"):
        overall = {feature: _distribution(layer_frame[feature]) for feature in features}
        expert_results: dict[str, Any] = {}
        for expert in range(num_experts):
            subset = layer_frame[layer_frame["oracle_expert"] == expert]
            feature_results: dict[str, Any] = {}
            for feature in features:
                distribution = _distribution(subset[feature])
                enrichment = {
                    category: probability
                    / max(overall[feature].get(category, 0.0), 1e-12)
                    for category, probability in distribution.items()
                }
                feature_results[feature] = {
                    "distribution": distribution,
                    "enrichment_vs_layer": enrichment,
                }
            expert_results[str(expert)] = {
                "count": int(len(subset)),
                "features": feature_results,
            }
        result["layers"][str(int(layer))] = {
            "overall": overall,
            "experts": expert_results,
        }
    return result


def specialization_observations(
    specialization: dict[str, Any], minimum_count: int = 20
) -> list[str]:
    observations: list[str] = []
    for layer, layer_data in specialization["layers"].items():
        for expert, expert_data in layer_data["experts"].items():
            if expert_data["count"] < minimum_count:
                continue
            candidates: list[tuple[float, str, str]] = []
            for feature, details in expert_data["features"].items():
                for category, ratio in details["enrichment_vs_layer"].items():
                    probability = details["distribution"][category]
                    expected_count = probability * expert_data["count"]
                    if expected_count >= minimum_count:
                        candidates.append((ratio, feature, category))
            if not candidates:
                continue
            ratio, feature, category = max(candidates)
            if ratio >= 1.1:
                probability = expert_data["features"][feature]["distribution"][
                    category
                ]
                observations.append(
                    f"Layer {layer} expert {int(expert) + 1}: {category} is the "
                    f"largest supported enrichment in {feature} ({ratio:.2f}x; "
                    f"{probability:.2%} within this expert's wins)."
                )
    return observations


def representative_examples(
    frame: pd.DataFrame, num_experts: int, examples_per_expert: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    columns = [
        "sequence_id",
        "position",
        "context",
        "input_byte",
        "target_byte",
        "loss_gap",
        "router_probabilities",
        "expert_losses",
    ]
    for layer, layer_frame in frame.groupby("layer"):
        layer_output: dict[str, Any] = {}
        for expert in range(num_experts):
            subset = layer_frame[layer_frame["oracle_expert"] == expert].sort_values(
                "loss_gap", ascending=False
            )
            # Repeated byte windows are common in fixed blocks; keep contexts unique.
            subset = subset.drop_duplicates(subset=["context"])
            layer_output[str(expert)] = subset.head(examples_per_expert)[
                columns
            ].to_dict(orient="records")
        output[str(int(layer))] = layer_output
    return output
