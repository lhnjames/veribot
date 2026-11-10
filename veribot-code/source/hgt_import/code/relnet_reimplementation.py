"""Two-layer gated relation-wise mean convolution for a new routing baseline.

This is a documented reimplementation from the manuscript's brief RelNet
description, not a recovered copy of the historical RelNet architecture.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GatedRelationLayer(nn.Module):
    def __init__(self, hidden_channels: int, relation_names: tuple[str, ...], dropout: float):
        super().__init__()
        self.relation_names = relation_names
        self.messages = nn.ModuleDict({
            name: nn.Linear(hidden_channels, hidden_channels, bias=False)
            for name in relation_names
        })
        self.gate = nn.Linear(hidden_channels * 2, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = dropout

    def forward(self, h: torch.Tensor, edge_by_relation: dict[str, torch.Tensor],
                edge_weight_by_relation: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        aggregates = []
        n = h.shape[0]
        for name in self.relation_names:
            edges = edge_by_relation[name]
            src, dst = edges[0], edges[1]
            messages = self.messages[name](h[src])
            weight = (None if edge_weight_by_relation is None else
                      edge_weight_by_relation[name].to(device=h.device, dtype=h.dtype))
            if weight is not None:
                if weight.ndim != 1 or weight.numel() != dst.numel():
                    raise ValueError(f"edge weight shape differs for {name}")
                messages = messages * weight[:, None]
            summed = torch.zeros_like(h)
            summed.index_add_(0, dst, messages)
            degree = torch.zeros((n, 1), device=h.device, dtype=h.dtype)
            degree.index_add_(0, dst, (
                torch.ones((dst.numel(), 1), device=h.device, dtype=h.dtype)
                if weight is None else weight[:, None]))
            aggregates.append(summed / degree.clamp(min=1))
        aggregate = torch.stack(aggregates).mean(dim=0)
        gated = torch.sigmoid(self.gate(torch.cat([h, aggregate], dim=-1)))
        updated = self.norm(h + gated * F.gelu(aggregate))
        return F.dropout(updated, p=self.dropout, training=self.training)


class GatedRelNet(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 128,
                 relation_names: tuple[str, ...] = ("follows", "rev_follows"),
                 num_layers: int = 2, dropout: float = 0.3,
                 out_channels: int = 1):
        super().__init__()
        self.relation_names = relation_names
        self.input = nn.Linear(input_channels, hidden_channels)
        self.layers = nn.ModuleList([
            GatedRelationLayer(hidden_channels, relation_names, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_by_relation: dict[str, torch.Tensor],
                edge_weight_by_relation: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        h = F.gelu(self.input(x))
        for layer in self.layers:
            h = layer(h, edge_by_relation, edge_weight_by_relation)
        return self.classifier(h).squeeze(-1)
