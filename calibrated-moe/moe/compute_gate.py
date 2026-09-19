from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .adaptive_routing import (
    PolicyForwardOutput,
    RoutingPolicy,
    route_sparse_moe,
    temperature_scaled_probabilities,
)
from .experts import SparseMoE
from .model import TinyMoELanguageModel


ROUTER_FEATURE_NAMES = ["top1_probability", "top2_probability", "margin", "entropy"]


class LinearComputeGate(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.output = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(features).squeeze(-1)


class MLPComputeGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class GateFeatureSpec:
    include_hidden: bool
    model_dim: int
    moe_layers: tuple[int, ...]

    @property
    def input_dim(self) -> int:
        return (self.model_dim if self.include_hidden else 0) + 4 + len(
            self.moe_layers
        )


def router_scalar_features(probabilities: torch.Tensor) -> torch.Tensor:
    top_values = torch.topk(probabilities, 2, dim=-1).values
    entropy = -torch.sum(
        probabilities * torch.log(probabilities.clamp_min(1e-12)), dim=-1
    )
    return torch.stack(
        [
            top_values[:, 0],
            top_values[:, 1],
            top_values[:, 0] - top_values[:, 1],
            entropy,
        ],
        dim=-1,
    )


def assemble_gate_features(
    pre_moe_hidden: torch.Tensor,
    router_probabilities: torch.Tensor,
    layer_index: int,
    spec: GateFeatureSpec,
) -> torch.Tensor:
    scalars = router_scalar_features(router_probabilities)
    layer_ids = torch.zeros(
        pre_moe_hidden.size(0),
        len(spec.moe_layers),
        device=pre_moe_hidden.device,
        dtype=pre_moe_hidden.dtype,
    )
    layer_ids[:, spec.moe_layers.index(layer_index)] = 1.0
    pieces = [scalars.to(pre_moe_hidden.dtype), layer_ids]
    if spec.include_hidden:
        pieces.insert(0, pre_moe_hidden)
    return torch.cat(pieces, dim=-1)


def gate_parameter_count(gate: nn.Module) -> int:
    return sum(parameter.numel() for parameter in gate.parameters())


def forward_with_compute_gate(
    model: TinyMoELanguageModel,
    input_ids: torch.Tensor,
    targets: torch.Tensor | None,
    gate: nn.Module,
    feature_spec: GateFeatureSpec,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    threshold: float,
) -> PolicyForwardOutput:
    batch, sequence = input_ids.shape
    positions = torch.arange(sequence, device=input_ids.device)
    hidden = model.embedding_dropout(
        model.token_embedding(input_ids) + model.position_embedding(positions)
    )
    assignments = 0
    decisions = 0
    routing_policy = RoutingPolicy("learned-compute-gate", fixed_k=2)
    for layer_index, block in enumerate(model.blocks):
        hidden = hidden + block.attention(block.attention_norm(hidden))
        ffn_inputs = block.ffn_norm(hidden)
        if isinstance(block.feed_forward, SparseMoE):
            flat = ffn_inputs.reshape(-1, ffn_inputs.size(-1))
            router_logits = block.feed_forward.router.projection(flat)
            probabilities = temperature_scaled_probabilities(router_logits, 1.0)
            # Stage 4 caches hidden features as float16 to keep the training
            # artifact compact. Quantize identically online so threshold
            # decisions match the fitted representation exactly.
            gate_hidden = flat.float().to(torch.float16).float()
            features = assemble_gate_features(
                gate_hidden, probabilities, layer_index, feature_spec
            )
            # The gate is intentionally kept in fp32; expert matmuls still use
            # the surrounding inference autocast context.
            with torch.autocast(device_type=input_ids.device.type, enabled=False):
                prediction = gate(
                    (features.float() - feature_mean) / feature_std
                )
            k2_mask = prediction > threshold
            moe_output, layer_assignments, layer_decisions = route_sparse_moe(
                block.feed_forward,
                ffn_inputs,
                1.0,
                routing_policy,
                k2_mask,
            )
            hidden = hidden + moe_output
            assignments += layer_assignments
            decisions += layer_decisions
        else:
            hidden = hidden + block.feed_forward(ffn_inputs)
    logits = model.language_model_head(model.final_norm(hidden))
    loss = None
    if targets is not None:
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
    return PolicyForwardOutput(logits, loss, assignments, decisions)
