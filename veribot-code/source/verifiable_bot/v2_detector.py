"""Account and relation gated detector for source-aware evidence experiments.

The classifier accepts a node feature tensor plus relation-specific neighbor
means. Evidence auditing rebuilds both tensors from source records after an
edit; attention weights here are candidate hints, never verification results.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def knn_edges(x: torch.Tensor, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine top-k with HypBot's orientation and tie behavior."""
    normalized = F.normalize(x.float(), dim=-1)
    n = len(x)
    if n <= k:
        raise ValueError("k must be smaller than account count")
    chunk = max(1, min(512, n // 4))
    targets, sources = [], []
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        sim = normalized[start:end] @ normalized.T
        sim[:, start:end].fill_diagonal_(-2.0)
        neighbors = sim.topk(k, dim=-1).indices
        targets.append(torch.arange(start, end).unsqueeze(1).expand_as(neighbors).reshape(-1))
        sources.append(neighbors.reshape(-1))
    edge = torch.stack([torch.cat(sources), torch.cat(targets)])
    return edge, torch.zeros(edge.shape[1], dtype=torch.long)


def relation_means(projected_x: torch.Tensor, edge_index: torch.Tensor,
                   edge_type: torch.Tensor, num_relations: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic directed relation means, with empty relations zero masked.

    Uses chunks to avoid materializing all 1.7M MGTAB edge messages at once.
    Sources send to targets according to ``edge_index[0] -> edge_index[1]``.
    """
    n, d = projected_x.shape
    out = projected_x.new_zeros((n, num_relations, d))
    degree = projected_x.new_zeros((n, num_relations))
    for start in range(0, edge_index.shape[1], 50_000):
        stop = min(start + 50_000, edge_index.shape[1])
        src, dst = edge_index[:, start:stop]
        rel = edge_type[start:stop]
        flat = dst * num_relations + rel
        out.view(n * num_relations, d).index_add_(0, flat, projected_x[src])
        degree.view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=degree.dtype))
    out = out / degree.clamp_min(1).unsqueeze(-1)
    return out, degree


class EvidenceGatedDetector(nn.Module):
    """Late fusion lets self features survive noisy or derived graph edges."""

    def __init__(self, feature_dim: int, graph_dim: int, num_relations: int,
                 hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.node = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden),
                                  nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.GELU())
        self.relation = nn.Sequential(nn.LayerNorm(graph_dim), nn.Linear(graph_dim, hidden),
                                      nn.GELU())
        self.relation_id = nn.Embedding(num_relations, hidden)
        self.relation_score = nn.Sequential(nn.Linear(2 * hidden + 1, hidden // 2),
                                            nn.GELU(), nn.Linear(hidden // 2, 1))
        self.node_head = nn.Linear(hidden, 1)
        self.graph_head = nn.Linear(hidden, 1)
        self.fusion_gate = nn.Sequential(nn.Linear(2 * hidden, hidden // 2),
                                         nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, x: torch.Tensor, neighbor_means: torch.Tensor,
                degree: torch.Tensor, return_components: bool = False):
        n, r, _ = neighbor_means.shape
        if degree.shape != (n, r) or x.shape[0] != n:
            raise ValueError("node, relation, and degree tensors have inconsistent shapes")
        own = self.node(x)
        rel = self.relation(neighbor_means)
        rel = rel + self.relation_id.weight.unsqueeze(0)
        score_input = torch.cat([
            own.unsqueeze(1).expand(-1, r, -1), rel,
            torch.log1p(degree).unsqueeze(-1),
        ], dim=-1)
        scores = self.relation_score(score_input).squeeze(-1)
        active = degree > 0
        scores = scores.masked_fill(~active, -1e9)
        weights = torch.softmax(scores, dim=1) * active
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        graph = (weights.unsqueeze(-1) * rel).sum(dim=1)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([own, graph], dim=-1))).squeeze(-1)
        gate = gate * active.any(dim=1)
        node_logit = self.node_head(own).squeeze(-1)
        graph_logit = self.graph_head(graph).squeeze(-1)
        logit = node_logit + gate * graph_logit
        if return_components:
            return logit, {"node_logit": node_logit, "graph_logit": graph_logit,
                           "graph_gate": gate, "relation_weights": weights}
        return logit


def focal_loss(logit: torch.Tensor, label: torch.Tensor,
               alpha_pos: float, gamma: float = 2.0) -> torch.Tensor:
    y = label.float()
    ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    p = torch.sigmoid(logit)
    pt = p * y + (1 - p) * (1 - y)
    alpha = alpha_pos * y + (1 - alpha_pos) * (1 - y)
    return (alpha * (1 - pt).pow(gamma) * ce).mean()
