"""Clean Reverse Model B: Heterogeneous Dual-Query Set-Transformer Challenger.

Architecturally independent from Forward Model A (Self-Attention + Dual Cross-Attention Queries
vs Forward Model A's within-family pooling + cross-family bilinear heads):
1. Same PayloadGroundedEncoder zero-payload invariant (zero_all & zero_payload == constant scalar).
2. ZeroPreservingSetTransformer layers (bias=False QKV/FFN + RMSNorm) strictly masked to valid
   tokens (or valid & hard_mask when an external witness/counterexample mask is evaluated).
3. Dual Query Cross-Attention:
   - q_bot attends over E+_B (Bot witnesses)
   - q_human attends over E-_B (Human counterexamples)
4. Zero cardinality terms concatenated into classification heads.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from verifiable_bot.clean_primitives import PayloadGroundedEncoder, RMSNorm, ZeroPreservingMLP


class ZeroPreservingTransformerBlock(nn.Module):
    """Pre-RMSNorm bias-free multi-head self-attention + FFN; f(0) == 0 identically."""

    def __init__(self, hidden: int, heads: int = 4, dropout: float = 0.15):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm1 = RMSNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.drop = nn.Dropout(dropout)
        self.norm2 = RMSNorm(hidden)
        self.ffn = ZeroPreservingMLP(hidden, hidden * 2, hidden, dropout=dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, m, _ = h.shape
        x = self.norm1(h)
        q = self.q_proj(x).view(b, m, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, m, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, m, self.heads, self.head_dim).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
        w = self.drop(torch.softmax(att, dim=-1))
        out = torch.matmul(w, v).transpose(1, 2).contiguous().view(b, m, self.hidden)
        h = (h + self.drop(self.o_proj(out))) * mask.unsqueeze(-1)
        h = (h + self.ffn(self.norm2(h))) * mask.unsqueeze(-1)
        return h


class ZeroPreservingQueryHead(nn.Module):
    """Cross-attention query pooling over a masked token set; outputs constant bias on zero payload."""

    def __init__(self, hidden: int, dropout: float = 0.15):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden) * 0.05)
        self.key_proj = nn.Linear(hidden, hidden, bias=False)
        self.val_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = ZeroPreservingMLP(hidden * 2, hidden, 1, dropout=dropout)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, h_ctx: torch.Tensor, h_local: torch.Tensor, mask: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
        k = self.key_proj(h_ctx)
        v = self.val_proj(h_ctx)
        scores = torch.matmul(self.query.expand(h_ctx.shape[0], 1, -1), k.transpose(-2, -1)).squeeze(1) / math.sqrt(
            h_ctx.shape[-1]
        )
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=1) * chosen
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        q_pool = (v * w.unsqueeze(-1)).sum(dim=1)
        mean_pool = (h_local * chosen.unsqueeze(-1)).sum(dim=1) / chosen.sum(dim=1, keepdim=True).clamp(min=1.0)
        logit = self.mlp(torch.cat((mean_pool, q_pool), dim=1)).squeeze(-1) + self.bias
        return torch.where(chosen.sum(dim=1) > 0, logit, torch.full_like(logit, -10.0))


class CleanChallengerNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        payload_cols: list[int] | np.ndarray,
        hidden: int = 160,
        heads: int = 4,
        layers: int = 2,
        num_freq: int = 6,
        dropout: float = 0.15,
        local_gate: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.payload_cols = [int(c) for c in payload_cols]
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.layers = int(layers)
        self.num_freq = int(num_freq)
        self.dropout = float(dropout)
        self.local_gate = bool(local_gate)

        self.encoder = PayloadGroundedEncoder(
            input_dim=input_dim,
            payload_cols=payload_cols,
            hidden=hidden,
            num_freq=num_freq,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList([ZeroPreservingTransformerBlock(hidden, heads=heads, dropout=dropout) for _ in range(layers)])

        type_feat_dim = hidden // 2 if self.encoder.d_type > 0 else 0
        # When local_gate=True, gates observe strictly per-group local payload/type (no global Transformer ctx)
        gate_in_dim = (hidden if local_gate else hidden * 2) + type_feat_dim
        self.bot_gate = nn.Sequential(nn.Linear(gate_in_dim, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.human_gate = nn.Sequential(nn.Linear(gate_in_dim, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

        self.bot_head = ZeroPreservingQueryHead(hidden, dropout=dropout)
        self.human_head = ZeroPreservingQueryHead(hidden, dropout=dropout)
        self.raw_ce_scale = nn.Parameter(torch.tensor(0.0))

    def _safe_valid(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.any(dim=1).all():
            return mask
        out = mask.clone()
        out[~mask.any(dim=1), 0] = True
        return out

    def _run_transformer(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe = self._safe_valid(mask)
        out = h * mask.unsqueeze(-1)
        for blk in self.blocks:
            out = blk(out, safe)
        return out * mask.unsqueeze(-1)

    def _select_gate(
        self,
        h_local: torch.Tensor,
        h_ctx: torch.Tensor,
        t_vec: torch.Tensor,
        valid: torch.Tensor,
        gate_net: nn.Module,
        *,
        mode: str = "selected",
        temperature: float = 0.67,
        hard_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.local_gate:
            parts = [h_local]
        else:
            denom = valid.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
            ctx = ((h_ctx * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom).expand_as(h_local)
            parts = [h_local, ctx]
        if self.encoder.gate_type_proj is not None:
            parts.append(self.encoder.gate_type_proj(t_vec))
        gate_in = torch.cat(parts, dim=-1)

        logits = gate_net(gate_in).squeeze(-1).masked_fill(~valid, -1e9)
        prob = torch.sigmoid(logits) * valid
        if hard_mask is not None:
            chosen = hard_mask.to(dtype=h_local.dtype) * valid
        elif mode == "all":
            chosen = valid.to(dtype=h_local.dtype)
        elif mode == "selected":
            if self.training:
                u = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
                soft = torch.sigmoid((logits + torch.log(u) - torch.log1p(-u)) / temperature) * valid
                hard = (soft > 0.5).to(dtype=h_local.dtype)
                hard.scatter_(1, logits.argmax(dim=1, keepdim=True), 1.0)
                chosen = hard - soft.detach() + soft
            else:
                chosen = (prob >= 0.5).to(dtype=h_local.dtype)
                empty = chosen.sum(dim=1) == 0
                if empty.any():
                    chosen[empty, logits[empty].argmax(dim=1)] = 1.0
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return chosen * valid, prob, logits

    def forward(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        family_ids: torch.Tensor | None = None,
        *,
        mode: str = "selected",
        temperature: float = 0.67,
        hard_mask: torch.Tensor | None = None,
    ) -> dict:
        if x.ndim != 3 or valid.shape != x.shape[:2]:
            raise ValueError("Invalid source shape")
        h_local, t_vec = self.encoder(x)

        if hard_mask is not None:
            branch_valid = valid & hard_mask.bool()
            h_ctx = self._run_transformer(h_local, branch_valid)
            bot_chosen, bot_prob, _ = self._select_gate(
                h_local * branch_valid.unsqueeze(-1),
                h_ctx,
                t_vec,
                branch_valid,
                self.bot_gate,
                mode=mode,
                temperature=temperature,
                hard_mask=hard_mask,
            )
            hum_chosen, hum_prob, hum_logits = self._select_gate(
                h_local * branch_valid.unsqueeze(-1),
                h_ctx,
                t_vec,
                branch_valid,
                self.human_gate,
                mode=mode,
                temperature=temperature,
                hard_mask=hard_mask,
            )
            safe_b = self._safe_valid(branch_valid)
            bot_logit = self.bot_head(h_ctx, h_local, safe_b, bot_chosen)
            hum_logit = self.human_head(h_ctx, h_local, safe_b, hum_chosen)
            ctx_valid = branch_valid
        else:
            h_ctx_full = self._run_transformer(h_local, valid)
            bot_chosen, bot_prob, _ = self._select_gate(
                h_local, h_ctx_full, t_vec, valid, self.bot_gate, mode=mode, temperature=temperature
            )
            hum_chosen, hum_prob, hum_logits = self._select_gate(
                h_local, h_ctx_full, t_vec, valid, self.human_gate, mode=mode, temperature=temperature
            )
            if mode == "selected":
                bot_mask = (bot_chosen.detach() > 0.5) & valid
                h_ctx_bot = self._run_transformer(h_local, bot_mask)
                bot_logit = self.bot_head(h_ctx_bot, h_local, self._safe_valid(bot_mask), bot_chosen)

                hum_mask = (hum_chosen.detach() > 0.5) & valid
                h_ctx_hum = self._run_transformer(h_local, hum_mask)
                hum_logit = self.human_head(h_ctx_hum, h_local, self._safe_valid(hum_mask), hum_chosen)
            else:
                safe_v = self._safe_valid(valid)
                bot_logit = self.bot_head(h_ctx_full, h_local, safe_v, bot_chosen)
                hum_logit = self.human_head(h_ctx_full, h_local, safe_v, hum_chosen)
            ctx_valid = valid

        ce_scale = F.softplus(self.raw_ce_scale)
        signed_logit = bot_logit - ce_scale * F.relu(hum_logit)
        if hard_mask is not None:
            has_any = (hard_mask.to(dtype=x.dtype) * valid).sum(dim=1) > 0
            signed_logit = torch.where(has_any, signed_logit, torch.full_like(signed_logit, -10.0))

        return {
            "bot_logit": signed_logit,
            "bot_witness_logit": bot_logit,
            "human_counterexample_logit": hum_logit,
            "ce_scale": float(ce_scale.detach().cpu()),
            "selected": bot_chosen.detach(),
            "human_selected": hum_chosen.detach(),
            "gate_probability": bot_prob,
            "human_gate_probability": hum_prob,
            "human_gate_logits": hum_logits,
            "selected_count": bot_chosen.sum(dim=1).clamp(min=1.0),
            "human_selected_count": hum_chosen.sum(dim=1).clamp(min=1.0),
            "mean_gate_probability": bot_prob.sum() / ctx_valid.sum().clamp(min=1),
            "mean_human_gate_probability": hum_prob.sum() / ctx_valid.sum().clamp(min=1),
        }
