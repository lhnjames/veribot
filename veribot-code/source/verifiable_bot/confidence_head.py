"""Active-GRPO confidence head (design spec section 6.1).

A small MLP reads the policy's own prefill hidden state for an account (before any rollout is
generated) and predicts, on a K=5 scale, how likely a full group of G rollouts for this account
is to be "informative" (i.e. not degenerate: not all-correct or all-wrong, which would give the
GRPO group-relative advantage nothing to learn from). Accounts voted confidently easy or
confidently hopeless are skipped before generation, saving both the rollout cost and the entire
downstream chain of LLM judge calls in rewards.py.

The head is trained with a calibration loss against the *empirical* group accuracy observed on
accounts that did run a full rollout (see train_grpo.py's CalibrationCallback) -- never against
skipped accounts, which is a known blind spot documented in spec section 10, item 2.
"""
from __future__ import annotations

import torch
from torch import nn


class ConfidenceHead(nn.Module):
    def __init__(self, hidden_size: int, k: int = 5, mlp_hidden: int = 256):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, k),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden.to(self.net[0].weight.dtype))


def compute_prefill_hidden(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Last-layer hidden state at each sequence's last real (non-padding) token.

    Assumes right-padding (attention_mask starts with 1s). Caller is responsible for setting
    tokenizer.padding_side = "right" before building input_ids/attention_mask.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    last_hidden = outputs.hidden_states[-1]
    lengths = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(last_hidden.shape[0], device=last_hidden.device)
    return last_hidden[batch_idx, lengths]


def skip_vote(probs: list[float], delta_p: float = 0.6, delta_m: float = 0.3, v_min: int = 2, k: int = 5) -> bool:
    """Multi-criterion vote deciding whether to skip full rollout for this account (spec 6.1)."""
    top = max(range(k), key=lambda i: probs[i])
    v1 = top in (0, k - 1)
    v2 = max(probs[0], probs[k - 1]) > delta_p
    v3 = max(probs[0] - probs[1], probs[k - 1] - probs[k - 2]) > delta_m
    return (int(v1) + int(v2) + int(v3)) >= v_min


def bucketize_accuracy(accuracy: float, k: int = 5) -> int:
    """y*(x) = min(floor(K * c_hat(x)), K - 1), spec 6.1."""
    return min(int(accuracy * k), k - 1)
