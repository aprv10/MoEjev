from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def confidence_bins(
    frame: pd.DataFrame, edges: list[float]
) -> list[dict[str, Any]]:
    confidence = frame["router_confidence"].to_numpy(dtype=np.float64)
    correct = (
        frame["router_top1"].to_numpy() == frame["oracle_expert"].to_numpy()
    )
    regret = frame["top1_regret"].to_numpy(dtype=np.float64)
    output: list[dict[str, Any]] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        include_upper = upper == edges[-1]
        mask = (confidence >= lower) & (
            confidence <= upper if include_upper else confidence < upper
        )
        count = int(mask.sum())
        output.append(
            {
                "lower": float(lower),
                "upper": float(min(upper, 1.0)),
                "count": count,
                "mean_router_confidence": (
                    float(confidence[mask].mean()) if count else None
                ),
                "oracle_top1_accuracy": (
                    float(correct[mask].mean()) if count else None
                ),
                "mean_top1_regret": float(regret[mask].mean()) if count else None,
            }
        )
    return output


def distribution_alignment(frame: pd.DataFrame) -> dict[str, float]:
    router = np.stack(frame["router_probabilities"].to_numpy()).astype(np.float64)
    oracle = np.stack(frame["soft_oracle_distribution"].to_numpy()).astype(
        np.float64
    )
    epsilon = 1e-12
    router = np.clip(router, epsilon, 1.0)
    oracle = np.clip(oracle, epsilon, 1.0)
    kl = np.sum(oracle * (np.log(oracle) - np.log(router)), axis=1)
    cross_entropy = -np.sum(oracle * np.log(router), axis=1)
    brier = np.sum(np.square(router - oracle), axis=1)
    cosine = np.sum(router * oracle, axis=1) / (
        np.linalg.norm(router, axis=1) * np.linalg.norm(oracle, axis=1) + epsilon
    )
    return {
        "mean_kl_q_to_p": float(kl.mean()),
        "mean_cross_entropy": float(cross_entropy.mean()),
        "mean_brier_squared_distance": float(brier.mean()),
        "mean_cosine_similarity": float(cosine.mean()),
    }

