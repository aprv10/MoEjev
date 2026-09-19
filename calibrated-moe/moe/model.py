from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .experts import FeedForwardExpert, SparseMoE


class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.output = nn.Linear(model_dim, model_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, sequence, width = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch, sequence, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, sequence, width)
        return self.output(attended)


@dataclass
class BlockOutput:
    hidden_states: torch.Tensor
    auxiliary_loss: torch.Tensor
    z_loss: torch.Tensor
    utilization: torch.Tensor | None
    routing_entropy: torch.Tensor | None


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dense_hidden_dim: int,
        dropout: float,
        moe: SparseMoE | None = None,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_dim)
        self.attention = CausalSelfAttention(model_dim, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.feed_forward = moe or FeedForwardExpert(
            model_dim, dense_hidden_dim, dropout
        )

    def forward(self, inputs: torch.Tensor) -> BlockOutput:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        ffn_input = self.ffn_norm(hidden)
        if isinstance(self.feed_forward, SparseMoE):
            moe_output = self.feed_forward(ffn_input)
            hidden = hidden + moe_output.hidden_states
            return BlockOutput(
                hidden,
                moe_output.auxiliary_loss,
                moe_output.z_loss,
                moe_output.utilization,
                moe_output.routing_entropy,
            )
        hidden = hidden + self.feed_forward(ffn_input)
        zero = hidden.new_zeros(())
        return BlockOutput(hidden, zero, zero, None, None)


@dataclass
class LanguageModelOutput:
    logits: torch.Tensor
    language_model_loss: torch.Tensor | None
    auxiliary_loss: torch.Tensor
    z_loss: torch.Tensor
    loss: torch.Tensor | None
    expert_utilization: torch.Tensor
    routing_entropy: torch.Tensor


class TinyMoELanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        dense_hidden_dim: int,
        expert_hidden_dim: int,
        num_experts: int,
        top_k: int,
        moe_layers: list[int],
        dropout: float,
        load_balance_weight: float,
        router_z_loss_weight: float,
    ) -> None:
        super().__init__()
        invalid = set(moe_layers) - set(range(num_layers))
        if invalid:
            raise ValueError(f"Invalid MoE layer indices: {sorted(invalid)}")
        self.sequence_length = sequence_length
        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight
        self.router_z_loss_weight = router_z_loss_weight
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(sequence_length, model_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        blocks = []
        for layer_index in range(num_layers):
            moe = None
            if layer_index in moe_layers:
                moe = SparseMoE(
                    model_dim,
                    expert_hidden_dim,
                    num_experts,
                    top_k,
                    dropout,
                )
            blocks.append(
                TransformerBlock(
                    model_dim,
                    num_heads,
                    dense_hidden_dim,
                    dropout,
                    moe,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(model_dim)
        self.language_model_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.language_model_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> LanguageModelOutput:
        batch, sequence = input_ids.shape
        if sequence > self.sequence_length:
            raise ValueError("Input exceeds configured sequence length")
        positions = torch.arange(sequence, device=input_ids.device)
        hidden = self.embedding_dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )
        auxiliary_losses = []
        z_losses = []
        utilizations = []
        entropies = []
        for block in self.blocks:
            output = block(hidden)
            hidden = output.hidden_states
            if output.utilization is not None:
                auxiliary_losses.append(output.auxiliary_loss)
                z_losses.append(output.z_loss)
                utilizations.append(output.utilization)
                entropies.append(output.routing_entropy)
        logits = self.language_model_head(self.final_norm(hidden))
        zero = logits.new_zeros(())
        auxiliary_loss = torch.stack(auxiliary_losses).mean() if auxiliary_losses else zero
        z_loss = torch.stack(z_losses).mean() if z_losses else zero
        language_model_loss = None
        loss = None
        if targets is not None:
            language_model_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
            loss = (
                language_model_loss
                + self.load_balance_weight * auxiliary_loss
                + self.router_z_loss_weight * z_loss
            )
        utilization = (
            torch.stack(utilizations)
            if utilizations
            else torch.zeros(0, self.num_experts, device=logits.device)
        )
        routing_entropy = (
            torch.stack(entropies).mean() if entropies else zero
        )
        return LanguageModelOutput(
            logits,
            language_model_loss,
            auxiliary_loss,
            z_loss,
            loss,
            utilization,
            routing_entropy,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

