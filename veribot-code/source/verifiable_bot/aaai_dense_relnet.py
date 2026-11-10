"""Matrix implementation of the manuscript's relation-wise mean convolution.

The layer parameters and equations are identical to GatedRelNet. A row-normalized
adjacency replaces an explicit edge list so dense native projections do not
materialize one 128-dimensional message per projected edge.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def normalized_adjacency(edge: torch.Tensor, node_count: int) -> torch.Tensor:
    """Return A[dst, src] = 1 / indegree(dst), matching edge-wise mean."""
    if edge.dtype != torch.long or edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("relation edges must have shape [2, edge_count]")
    if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= node_count):
        raise ValueError("relation edge leaves user inventory")
    adjacency = torch.zeros((node_count, node_count), dtype=torch.float32)
    if edge.numel():
        src, dst = edge
        adjacency.index_put_((dst, src), torch.ones_like(src, dtype=torch.float32),
                             accumulate=True)
        degree = adjacency.sum(dim=1, keepdim=True)
        adjacency = adjacency / degree.clamp(min=1)
    return adjacency


class DenseGatedRelationLayer(nn.Module):
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

    def forward(self, h: torch.Tensor,
                adjacency_by_relation: dict[str, torch.Tensor]) -> torch.Tensor:
        aggregates = []
        for name in self.relation_names:
            adjacency = adjacency_by_relation[name]
            if adjacency.shape != (len(h), len(h)):
                raise ValueError(f"relation matrix {name} has wrong shape")
            aggregates.append(adjacency @ self.messages[name](h))
        aggregate = torch.stack(aggregates).mean(dim=0)
        gated = torch.sigmoid(self.gate(torch.cat([h, aggregate], dim=-1)))
        updated = self.norm(h + gated * F.gelu(aggregate))
        return F.dropout(updated, p=self.dropout, training=self.training)


class DenseGatedRelNet(nn.Module):
    """Same two-layer gated RelNet using precomputed mean-adjacency operators."""

    def __init__(self, input_channels: int, hidden_channels: int = 128,
                 relation_names: tuple[str, ...] = ("shared_subreddit",),
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.relation_names = relation_names
        self.input = nn.Linear(input_channels, hidden_channels)
        self.layers = nn.ModuleList([
            DenseGatedRelationLayer(hidden_channels, relation_names, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor,
                adjacency_by_relation: dict[str, torch.Tensor]) -> torch.Tensor:
        h = F.gelu(self.input(x))
        for layer in self.layers:
            h = layer(h, adjacency_by_relation)
        return self.classifier(h).squeeze(-1)
