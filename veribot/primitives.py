"""Zero-preserving structural primitives for leakage-free verifiable evidence networks.

Design guarantee:
- Dynamic content payload p_j and static type indicators t_j are structurally separated.
- PayloadGroundedEncoder maps p_j -> h_j via Periodic Linear Representations (PLR) modulated
  multiplicatively by t_j (FiLM) and ZeroPreservingMLP blocks.
- Every downstream layer in the scorer path uses bias=False linear maps, RMSNorm, GELU, and
  normalized convex pooling, plus a single global output scalar bias b_global.
- Mathematical identity: whenever selected payload p_j == 0 (for ANY mask or static type vector),
  h_j == 0 identically -> scorer logit == b_global identically (std == 0.0, AUC == 0.5000).
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Bias-free RMS normalization that preserves exact zeros: RMSNorm(0) == 0."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class ZeroPreservingMLP(nn.Module):
    """Bias-free residual-safe MLP: f(0) == 0 identically."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.norm = RMSNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm(self.fc1(x)))
        return self.fc2(self.drop(h))


class PayloadGroundedEncoder(nn.Module):
    """Encodes per-group dynamic payload p_j with PLR embeddings + type FiLM modulation.

    Guarantees output h_j == 0 identically whenever p_j == 0, regardless of static type t_j.
    Also returns static type slice t_j separately for gating / attention key disambiguation.
    """

    def __init__(
        self,
        input_dim: int,
        payload_cols: list[int] | np.ndarray,
        hidden: int = 128,
        num_freq: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        pay_idx = sorted({int(i) for i in payload_cols if 0 <= int(i) < input_dim})
        if not pay_idx:
            pay_idx = list(range(input_dim))
        pay_set = set(pay_idx)
        type_idx = [i for i in range(input_dim) if i not in pay_set]

        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.num_freq = int(num_freq)
        self.register_buffer("payload_idx", torch.tensor(pay_idx, dtype=torch.long))
        self.register_buffer("type_idx", torch.tensor(type_idx, dtype=torch.long))

        d_pay = len(pay_idx)
        d_type = len(type_idx)
        self.d_pay = d_pay
        self.d_type = d_type

        if num_freq > 0:
            self.freqs = nn.Parameter(torch.randn(d_pay, num_freq) * 0.15)
        else:
            self.register_parameter("freqs", None)

        plr_dim = d_pay * (1 + 2 * num_freq) if num_freq > 0 else d_pay
        self.pay_proj = nn.Linear(plr_dim, hidden, bias=False)
        self.pay_norm = RMSNorm(hidden)
        self.drop = nn.Dropout(dropout)

        if d_type > 0:
            self.type_film = nn.Linear(d_type, hidden, bias=True)
            self.gate_type_proj = nn.Sequential(
                nn.Linear(d_type, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, hidden // 2),
            )
        else:
            self.type_film = None
            self.gate_type_proj = None

        self.res_mlp = ZeroPreservingMLP(hidden, hidden * 2, hidden, dropout=dropout)

    def split_payload_and_type(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = x.index_select(-1, self.payload_idx)
        if self.d_type > 0:
            t = x.index_select(-1, self.type_idx)
        else:
            t = x.new_zeros(*x.shape[:-1], 0)
        return p, t

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p, t = self.split_payload_and_type(x)
        if self.freqs is not None:
            angles = (2.0 * math.pi) * p.unsqueeze(-1) * self.freqs
            # sin(0) == 0 and cos(0) - 1 == 0 -> exact zero preservation on p == 0
            plr = torch.cat((p.unsqueeze(-1), torch.sin(angles), torch.cos(angles) - 1.0), dim=-1).flatten(-2)
        else:
            plr = p

        h = self.pay_norm(self.pay_proj(plr))
        if self.type_film is not None:
            h = h * (1.0 + 0.5 * torch.tanh(self.type_film(t)))
        h = h + self.res_mlp(h)
        h = self.drop(h)
        return h, t


class ZeroPreservingMultiQueryPool(nn.Module):
    """Multi-head zero-preserving set attention + normalized mean pool (preserves f(0) == 0)."""

    def __init__(self, hidden: int, num_queries: int = 4, dropout: float = 0.15):
        super().__init__()
        self.num_queries = int(num_queries)
        self.queries = nn.Parameter(torch.randn(num_queries, hidden) * 0.05)
        self.key_proj = nn.Linear(hidden, hidden, bias=False)
        self.val_proj = nn.Linear(hidden, hidden, bias=False)
        self.out_proj = ZeroPreservingMLP(hidden * (num_queries + 1), hidden * 2, hidden, dropout=dropout)

    def forward(
        self,
        h: torch.Tensor,
        safe_mask: torch.Tensor,
        m_f: torch.Tensor,
        cnt_f: torch.Tensor,
    ) -> torch.Tensor:
        k = self.key_proj(h)  # (B, M, D)
        v = self.val_proj(h)  # (B, M, D)
        # (B, Q, M)
        scores = torch.einsum("qd,bmd->bqm", self.queries, k) / math.sqrt(h.shape[-1])
        scores = scores.masked_fill(~safe_mask.unsqueeze(1), -1e9)
        w = torch.softmax(scores, dim=-1) * m_f.unsqueeze(1)
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        att_pools = torch.einsum("bqm,bmd->bqd", w, v).reshape(h.shape[0], -1)
        mean_pool = (h * m_f.unsqueeze(-1)).sum(dim=1) / cnt_f.clamp(min=1.0)
        return self.out_proj(torch.cat((mean_pool, att_pools), dim=-1))
