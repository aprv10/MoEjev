from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from moe.experts import SparseMoE
from moe.model import TinyMoELanguageModel


@dataclass
class OracleBatch:
    router_probabilities: torch.Tensor
    router_top2: torch.Tensor
    expert_losses: torch.Tensor
    baseline_top2_losses: torch.Tensor
    pre_moe_hidden: torch.Tensor


def _continue_from(
    model: TinyMoELanguageModel, hidden: torch.Tensor, first_block: int
) -> torch.Tensor:
    for block in model.blocks[first_block:]:
        hidden = block(hidden).hidden_states
    return hidden


def _token_losses(
    model: TinyMoELanguageModel,
    hidden: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    logits = model.language_model_head(model.final_norm(hidden))
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)


def evaluate_layer_oracle(
    model: TinyMoELanguageModel,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    layer_index: int,
    intervention_chunk_size: int,
) -> OracleBatch:
    """Evaluate exact, isolated single-token expert interventions.

    At the target layer, the baseline Top-2 output is retained at every token
    except one. For that token only, the MoE output is replaced by one forced
    expert's output with unit weight. The unchanged downstream model is then
    evaluated at the same position. Batched sequence clones contain exactly one
    intervention each, so counterfactual tokens cannot affect one another.
    """
    if not 0 <= layer_index < len(model.blocks):
        raise ValueError(f"Invalid layer index {layer_index}")
    target_block = model.blocks[layer_index]
    if not isinstance(target_block.feed_forward, SparseMoE):
        raise ValueError(f"Layer {layer_index} is not an MoE layer")
    if intervention_chunk_size < 1:
        raise ValueError("intervention_chunk_size must be positive")

    batch_size, sequence_length = input_ids.shape
    positions = torch.arange(sequence_length, device=input_ids.device)
    hidden = model.embedding_dropout(
        model.token_embedding(input_ids) + model.position_embedding(positions)
    )
    for block in model.blocks[:layer_index]:
        hidden = block(hidden).hidden_states

    attention_hidden = hidden + target_block.attention(
        target_block.attention_norm(hidden)
    )
    expert_inputs = target_block.ffn_norm(attention_hidden)
    flat_inputs = expert_inputs.reshape(-1, expert_inputs.size(-1))
    flat_attention = attention_hidden.reshape(-1, attention_hidden.size(-1))
    moe = target_block.feed_forward
    routed = moe.router(flat_inputs)

    # Compute every expert once for this batch. This is analysis-only; normal
    # inference continues to use SparseMoE's selected-expert execution path.
    all_expert_outputs = torch.stack(
        [expert(flat_inputs) for expert in moe.experts], dim=0
    )
    token_indices = torch.arange(flat_inputs.size(0), device=input_ids.device)
    baseline_moe = torch.zeros_like(flat_inputs)
    for slot in range(routed.expert_indices.size(1)):
        chosen = routed.expert_indices[:, slot]
        selected = all_expert_outputs[chosen, token_indices]
        weights = routed.expert_weights[:, slot].to(selected.dtype).unsqueeze(-1)
        baseline_moe.add_((selected * weights).to(baseline_moe.dtype))
    baseline_hidden = attention_hidden + baseline_moe.reshape_as(attention_hidden)
    baseline_final = _continue_from(model, baseline_hidden, layer_index + 1)
    baseline_losses = _token_losses(model, baseline_final, targets).reshape(-1)

    num_tokens = batch_size * sequence_length
    num_experts = len(moe.experts)
    expert_losses = torch.empty(
        num_tokens,
        num_experts,
        device=input_ids.device,
        dtype=torch.float32,
    )

    # The final MoE layer needs no sequence clones because no downstream
    # attention can mix token positions after the intervention.
    if layer_index == len(model.blocks) - 1:
        flat_targets = targets.reshape(-1)
        for expert_id in range(num_experts):
            candidate = flat_attention + all_expert_outputs[expert_id].to(
                flat_attention.dtype
            )
            logits = model.language_model_head(model.final_norm(candidate))
            expert_losses[:, expert_id] = F.cross_entropy(
                logits, flat_targets, reduction="none"
            ).float()
    else:
        flat_targets = targets.reshape(-1)
        for expert_id in range(num_experts):
            for start in range(0, num_tokens, intervention_chunk_size):
                stop = min(start + intervention_chunk_size, num_tokens)
                flat_positions = torch.arange(start, stop, device=input_ids.device)
                sequence_indices = torch.div(
                    flat_positions, sequence_length, rounding_mode="floor"
                )
                token_positions = flat_positions.remainder(sequence_length)
                candidates = baseline_hidden.index_select(
                    0, sequence_indices
                ).clone()
                replacement = (
                    flat_attention.index_select(0, flat_positions)
                    + all_expert_outputs[expert_id]
                    .index_select(0, flat_positions)
                    .to(flat_attention.dtype)
                )
                row_indices = torch.arange(stop - start, device=input_ids.device)
                candidates[row_indices, token_positions] = replacement
                continued = _continue_from(model, candidates, layer_index + 1)
                selected_hidden = continued[row_indices, token_positions]
                logits = model.language_model_head(
                    model.final_norm(selected_hidden)
                )
                expert_losses[start:stop, expert_id] = F.cross_entropy(
                    logits,
                    flat_targets.index_select(0, flat_positions),
                    reduction="none",
                ).float()

    return OracleBatch(
        router_probabilities=routed.probabilities.detach().float(),
        router_top2=routed.expert_indices.detach(),
        expert_losses=expert_losses.detach(),
        baseline_top2_losses=baseline_losses.detach().float(),
        pre_moe_hidden=flat_inputs.detach().float(),
    )


def checkpoint_parameter_fingerprint(model: nn.Module) -> tuple[float, float]:
    """Cheap mutation guard used before and after the read-only analysis."""
    total_sum = 0.0
    total_square_sum = 0.0
    for parameter in model.parameters():
        values = parameter.detach().double()
        total_sum += float(values.sum())
        total_square_sum += float(values.square().sum())
    return total_sum, total_square_sum
