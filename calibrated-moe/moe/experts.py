from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .router_baseline import TopKRouter


class FeedForwardExpert(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass
class MoEOutput:
    hidden_states: torch.Tensor
    auxiliary_loss: torch.Tensor
    z_loss: torch.Tensor
    utilization: torch.Tensor
    routing_entropy: torch.Tensor


class SparseMoE(nn.Module):
    """Token-level sparse MoE that computes only the selected expert paths."""

    def __init__(
        self,
        model_dim: int,
        expert_hidden_dim: int,
        num_experts: int,
        top_k: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.router = TopKRouter(model_dim, num_experts, top_k)
        self.experts = nn.ModuleList(
            FeedForwardExpert(model_dim, expert_hidden_dim, dropout)
            for _ in range(num_experts)
        )

    def forward(self, hidden_states: torch.Tensor) -> MoEOutput:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, original_shape[-1])
        routed = self.router(flat)
        combined = torch.zeros_like(flat)

        # Each loop processes only tokens assigned to that expert. index_add_
        # combines weighted Top-K outputs back into the original token order.
        for expert_id, expert in enumerate(self.experts):
            token_positions, slots = torch.where(
                routed.expert_indices == expert_id
            )
            if token_positions.numel() == 0:
                continue
            expert_output = expert(flat.index_select(0, token_positions))
            weights = routed.expert_weights[token_positions, slots]
            combined.index_add_(
                0,
                token_positions,
                (
                    expert_output
                    * weights.to(expert_output.dtype).unsqueeze(-1)
                ).to(combined.dtype),
            )

        entropy = -torch.sum(
            routed.probabilities
            * torch.log(routed.probabilities.clamp_min(1e-9)),
            dim=-1,
        ).mean()
        return MoEOutput(
            hidden_states=combined.reshape(original_shape),
            auxiliary_loss=routed.auxiliary_loss,
            z_loss=routed.z_loss,
            utilization=routed.utilization,
            routing_entropy=entropy,
        )
