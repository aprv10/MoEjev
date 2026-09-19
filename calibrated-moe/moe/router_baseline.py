from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RouterOutput:
    probabilities: torch.Tensor
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    auxiliary_loss: torch.Tensor
    z_loss: torch.Tensor
    utilization: torch.Tensor


class TopKRouter(nn.Module):
    """Standard learned token-level softmax router with Top-K selection."""

    def __init__(self, model_dim: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.projection = nn.Linear(model_dim, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> RouterOutput:
        logits = self.projection(hidden_states)
        # Router probabilities stay in fp32 even when expert computation uses fp16.
        probabilities = F.softmax(logits.float(), dim=-1)
        top_values, expert_indices = torch.topk(
            probabilities, self.top_k, dim=-1
        )
        expert_weights = top_values / top_values.sum(dim=-1, keepdim=True)

        flat_indices = expert_indices.reshape(-1)
        utilization = torch.bincount(
            flat_indices, minlength=self.num_experts
        ).to(probabilities.dtype)

        # Switch-style balancing: probability mass and actual assignments should
        # both approach a uniform distribution. Counts are diagnostic/non-smooth;
        # gradients flow through the probability term.
        importance = probabilities.mean(dim=0)
        load = utilization / utilization.sum().clamp_min(1.0)
        auxiliary_loss = self.num_experts * torch.sum(importance * load)
        z_loss = torch.mean(torch.logsumexp(logits.float(), dim=-1).square())

        return RouterOutput(
            probabilities=probabilities,
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            auxiliary_loss=auxiliary_loss,
            z_loss=z_loss,
            utilization=utilization,
        )

