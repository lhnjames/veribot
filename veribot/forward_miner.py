"""Forward Evidence Miner (M_fwd): Leakage-Free Cross-Family & Bilinear Synergy Coalition Network.

Standard Academic Nomenclature:
- Model Name: Forward Evidence Miner (M_fwd)
- Key Modules: PayloadGroundedEncoder, Unbiased Local Gate, Cross-Family Bilinear Synergy Head
- Output: Bot Logit, Predecision Selected Evidence Subset E+_fwd, Additive Unary-Only Baseline
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from veribot.primitives import PayloadGroundedEncoder, ZeroPreservingMLP


class ForwardEvidenceMiner(nn.Module):
    def __init__(
        self,
        input_dim: int,
        payload_cols: list[int] | np.ndarray,
        hidden: int = 128,
        pair_dim: int = 32,
        num_families: int = 4,
        num_freq: int = 0,
        num_queries: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.payload_cols = [int(c) for c in payload_cols]
        self.hidden = int(hidden)
        self.pair_dim = int(pair_dim)
        self.num_families = int(num_families)
        self.num_freq = int(num_freq)
        self.num_queries = int(num_queries)
        self.dropout = float(dropout)

        self.encoder = PayloadGroundedEncoder(
            input_dim=input_dim,
            payload_cols=payload_cols,
            hidden=hidden,
            num_freq=num_freq,
            dropout=dropout,
        )
        type_feat_dim = hidden // 2 if self.encoder.d_type > 0 else 0
        self.gate = nn.Sequential(
            nn.Linear(hidden * 3 + type_feat_dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.unary_proj = ZeroPreservingMLP(hidden, hidden, hidden, dropout=dropout)
        self.pair_left = nn.Linear(hidden, pair_dim, bias=False)
        self.pair_right = nn.Linear(hidden, pair_dim, bias=False)
        self.cross_left = nn.Linear(hidden, pair_dim, bias=False)
        self.cross_right = nn.Linear(hidden, pair_dim, bias=False)

        head_in = hidden + pair_dim * 2
        self.head = ZeroPreservingMLP(head_in, hidden, 1, dropout=dropout)
        self.global_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        family_ids: torch.Tensor | None = None,
        *,
        mode: str = "selected",
        temperature: float = 0.67,
        hard_mask: torch.Tensor | None = None,
        pair_scale: float = 1.0,
    ) -> dict:
        if x.ndim != 3 or valid.shape != x.shape[:2]:
            raise ValueError("Invalid group tensor shape")
        if family_ids is None:
            family_ids = torch.zeros_like(valid, dtype=torch.long)

        h, t_vec = self.encoder(x)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
        ctx = ((h * valid.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom).expand_as(h)
        if self.encoder.gate_type_proj is not None:
            gate_in = torch.cat((h, ctx, h * ctx, self.encoder.gate_type_proj(t_vec)), dim=-1)
        else:
            gate_in = torch.cat((h, ctx, h * ctx), dim=-1)

        logits = self.gate(gate_in).squeeze(-1).masked_fill(~valid, -1e9)
        probability = torch.sigmoid(logits) * valid

        if hard_mask is not None:
            chosen = hard_mask.to(dtype=x.dtype) * valid
        elif mode == "all":
            chosen = valid.to(dtype=x.dtype)
        elif mode == "selected":
            if self.training:
                u = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
                noise = torch.log(u) - torch.log1p(-u)
                continuous = torch.sigmoid((logits + noise) / temperature) * valid
                hard = (continuous > 0.5).to(dtype=x.dtype)
                chosen = hard + continuous - continuous.detach()
            else:
                chosen = (probability > 0.5).to(dtype=x.dtype) * valid
        else:
            raise ValueError(f"Unknown mode: {mode}")

        b_sz, m_sz, _ = h.shape
        counts = chosen.sum(dim=1)
        m_f = chosen.unsqueeze(-1)

        # 1. Unary evidence summary
        h_unary = (self.unary_proj(h) * m_f).sum(dim=1) / counts.clamp(min=1.0).unsqueeze(-1)

        # 2. Family-conditioned witness pooling (Fams 0..3)
        fam_pools = []
        act_fams = []
        for fid in range(self.num_families):
            f_mask = chosen * (family_ids == fid).to(dtype=x.dtype)
            fc = f_mask.sum(dim=1)
            act_fams.append(fc > 0.5)
            f_pool = (h * f_mask.unsqueeze(-1)).sum(dim=1) / fc.clamp(min=1.0).unsqueeze(-1)
            fam_pools.append(f_pool)

        # 3. All-pair distinct bilinear interaction over selected groups (g1 != g2)
        u_p = self.pair_left(h)
        v_p = self.pair_right(h)
        sum_u = (u_p * m_f).sum(dim=1)
        sum_v = (v_p * m_f).sum(dim=1)
        outer_prod = sum_u * sum_v
        self_prod = ((u_p * v_p) * m_f).sum(dim=1)
        pair_denom = (counts * (counts - 1.0)).clamp(min=1.0).unsqueeze(-1)
        h_pair = (outer_prod - self_prod) / pair_denom
        h_pair = torch.where((counts >= 2.0).unsqueeze(-1), h_pair, torch.zeros_like(h_pair))

        # 4. Cross-family bilinear interaction (f(g1) != f(g2))
        num_pairs = 0
        h_cross = torch.zeros(b_sz, self.pair_dim, device=x.device, dtype=h.dtype)
        for i in range(self.num_families):
            for j in range(i + 1, self.num_families):
                pair_active = (act_fams[i] & act_fams[j]).unsqueeze(-1)
                term = self.cross_left(fam_pools[i]) * self.cross_right(fam_pools[j])
                h_cross = h_cross + torch.where(pair_active, term, torch.zeros_like(term))
                num_pairs += 1
        h_cross = h_cross / max(num_pairs, 1)

        head_feature = torch.cat((h_unary, pair_scale * h_pair, pair_scale * h_cross), dim=-1)
        bot_logit = self.head(head_feature).squeeze(-1) + self.global_bias

        # Additive unary-only baseline
        unary_feat = torch.cat((h_unary, torch.zeros_like(h_pair), torch.zeros_like(h_cross)), dim=-1)
        unary_logit = self.head(unary_feat).squeeze(-1) + self.global_bias

        active_family_count = torch.stack(act_fams, dim=1).float().sum(dim=1)
        return {
            "bot_logit": bot_logit,
            "unary_only_logit": unary_logit,
            "gate_probability": probability,
            "gate_logit": logits,
            "selected": (chosen > 0.5),
            "selected_count": counts,
            "active_families": active_family_count,
            "synergy_gain": bot_logit - unary_logit,
        }


# Backward compatibility alias
CleanCoalitionNet = ForwardEvidenceMiner
