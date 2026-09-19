from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from .experts import SparseMoE
from .model import TinyMoELanguageModel


@dataclass(frozen=True)
class RoutingPolicy:
    name: str
    fixed_k: int | None = None
    uncertainty_metric: str | None = None
    threshold: float | None = None
    temperatures: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fixed_k not in (None, 1, 2):
            raise ValueError("fixed_k must be 1, 2, or None")
        if self.fixed_k is None:
            if self.uncertainty_metric not in {"max_probability", "entropy", "margin"}:
                raise ValueError("Adaptive policies need a supported uncertainty metric")
            if self.threshold is None:
                raise ValueError("Adaptive policies need a threshold")


@dataclass
class PolicyForwardOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    expert_assignments: int
    routing_decisions: int


def temperature_scaled_probabilities(
    logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return F.softmax(logits.float() / temperature, dim=-1)


def uncertainty_signal(
    probabilities: torch.Tensor, metric: str
) -> torch.Tensor:
    top_values = torch.topk(probabilities, 2, dim=-1).values
    if metric == "max_probability":
        return top_values[:, 0]
    if metric == "entropy":
        return -torch.sum(
            probabilities * torch.log(probabilities.clamp_min(1e-12)), dim=-1
        )
    if metric == "margin":
        return top_values[:, 0] - top_values[:, 1]
    raise ValueError(f"Unsupported uncertainty metric: {metric}")


def adaptive_k2_mask(
    probabilities: torch.Tensor, metric: str, threshold: float
) -> torch.Tensor:
    signal = uncertainty_signal(probabilities, metric)
    if metric == "entropy":
        return signal > threshold
    return signal < threshold


def route_sparse_moe(
    moe: SparseMoE,
    hidden_states: torch.Tensor,
    temperature: float,
    policy: RoutingPolicy,
    k2_mask_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int]:
    original_shape = hidden_states.shape
    flat = hidden_states.reshape(-1, original_shape[-1])
    logits = moe.router.projection(flat)
    probabilities = temperature_scaled_probabilities(logits, temperature)
    top_values, top_indices = torch.topk(probabilities, 2, dim=-1)
    normalized_weights = top_values / top_values.sum(dim=-1, keepdim=True)

    if k2_mask_override is not None:
        k2_mask = k2_mask_override.reshape(-1).to(device=flat.device, dtype=torch.bool)
    elif policy.fixed_k == 2:
        k2_mask = torch.ones(flat.size(0), dtype=torch.bool, device=flat.device)
    elif policy.fixed_k == 1:
        k2_mask = torch.zeros(flat.size(0), dtype=torch.bool, device=flat.device)
    else:
        k2_mask = adaptive_k2_mask(
            probabilities, policy.uncertainty_metric or "", float(policy.threshold)
        )

    slot_weights = normalized_weights.clone()
    slot_weights[:, 0] = torch.where(
        k2_mask, normalized_weights[:, 0], torch.ones_like(normalized_weights[:, 0])
    )
    slot_weights[:, 1] = torch.where(
        k2_mask, normalized_weights[:, 1], torch.zeros_like(normalized_weights[:, 1])
    )
    combined = torch.zeros_like(flat)
    for expert_id, expert in enumerate(moe.experts):
        for slot in range(2):
            selected = top_indices[:, slot] == expert_id
            if slot == 1:
                selected = selected & k2_mask
            token_positions = torch.where(selected)[0]
            if token_positions.numel() == 0:
                continue
            expert_output = expert(flat.index_select(0, token_positions))
            weights = slot_weights[token_positions, slot]
            combined.index_add_(
                0,
                token_positions,
                (expert_output * weights.to(expert_output.dtype).unsqueeze(-1)).to(
                    combined.dtype
                ),
            )
    assignments = int(flat.size(0) + k2_mask.sum().item())
    return combined.reshape(original_shape), assignments, flat.size(0)


def forward_with_routing_policy(
    model: TinyMoELanguageModel,
    input_ids: torch.Tensor,
    targets: torch.Tensor | None,
    policy: RoutingPolicy,
    oracle_k2_masks: dict[int, torch.Tensor] | None = None,
) -> PolicyForwardOutput:
    batch, sequence = input_ids.shape
    positions = torch.arange(sequence, device=input_ids.device)
    hidden = model.embedding_dropout(
        model.token_embedding(input_ids) + model.position_embedding(positions)
    )
    expert_assignments = 0
    routing_decisions = 0
    for layer_index, block in enumerate(model.blocks):
        hidden = hidden + block.attention(block.attention_norm(hidden))
        ffn_inputs = block.ffn_norm(hidden)
        if isinstance(block.feed_forward, SparseMoE):
            override = (
                oracle_k2_masks.get(layer_index)
                if oracle_k2_masks is not None
                else None
            )
            moe_output, assignments, decisions = route_sparse_moe(
                block.feed_forward,
                ffn_inputs,
                policy.temperatures.get(layer_index, 1.0),
                policy,
                override,
            )
            hidden = hidden + moe_output
            expert_assignments += assignments
            routing_decisions += decisions
        else:
            hidden = hidden + block.feed_forward(ffn_inputs)
    logits = model.language_model_head(model.final_norm(hidden))
    loss = None
    if targets is not None:
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
    return PolicyForwardOutput(
        logits=logits,
        loss=loss,
        expert_assignments=expert_assignments,
        routing_decisions=routing_decisions,
    )

