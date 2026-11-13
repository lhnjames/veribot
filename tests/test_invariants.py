"""Unit tests verifying core mathematical invariants of VeriBot.

Specifically tests:
1. Mask Invariance Theorem: std(M(0, m)) == 0.0 for any random masks m in {0, 1}^L.
2. Zero-Payload Decoupling: std(M(0_payload, m)) == 0.0, proving zero mask steganography.
3. Sparse Gradient Invariant: dL / dX_j == 0 for all unselected evidence tokens j not in C*.
4. Permutation Invariance: Challenger output is invariant to permutation of evidence tokens.
5. Auto Payload Column Resolution: Correctly separates dynamic payload from static group types.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from veribot.primitives import resolve_payload_cols_auto
from veribot.forward_miner import ForwardEvidenceMiner
from veribot.heterogeneous_challenger import HeterogeneousChallenger


def test_payload_auto_resolution():
    """Verify resolve_payload_cols_auto correctly discovers payload vs constant type columns."""
    dim = 64
    payload_true = list(range(0, 48))
    type_true = list(range(48, 64))

    cache = {
        "features": [],
        "group_names": [],
    }
    group_types = ["profile:main", "relation:follower", "content:tweet", "content:hashtag"]
    type_vectors = {gt: np.random.randn(16).astype(np.float32) for gt in group_types}

    for account_i in range(100):
        acc_feats = []
        acc_names = []
        for g_i in range(10):
            gt = group_types[g_i % len(group_types)]
            payload = np.random.randn(48).astype(np.float32)
            row = np.concatenate([payload, type_vectors[gt]])
            acc_feats.append(row)
            acc_names.append(f"{gt}:{g_i}")
        cache["features"].append(torch.from_numpy(np.stack(acc_feats)))
        cache["group_names"].append(acc_names)

    resolved = resolve_payload_cols_auto(cache, list(range(100)), spec="auto")
    assert sorted(resolved.tolist()) == payload_true, f"Mismatch: {resolved} vs {payload_true}"


def test_mask_invariance_forward_miner():
    """Verify std(M_fwd(0, m)) == 0.0 across random masks m."""
    input_dim = 64
    payload_cols = list(range(0, 48))
    model = ForwardEvidenceMiner(input_dim, payload_cols=payload_cols, hidden=64, pair_dim=16)
    model.eval()

    L = 20
    x_zero = torch.zeros(1, L, input_dim)
    v = torch.ones(1, L, dtype=torch.bool)
    fams = torch.randint(0, 4, (1, L), dtype=torch.long)

    logits = []
    with torch.inference_mode():
        for _ in range(50):
            # Random binary mask with at least 1 active group
            m = (torch.rand(1, L) > 0.5)
            if not m.any():
                m[0, 0] = True
            out = model(x_zero, v, fams, hard_mask=m)
            logits.append(float(out["bot_logit"][0]))

    logits = np.array(logits)
    assert np.std(logits) < 1e-6, f"Forward miner violated mask invariance: std={np.std(logits)}"


def test_mask_invariance_heterogeneous_challenger():
    """Verify std(M_rev(0, m)) == 0.0 across random masks m."""
    input_dim = 64
    payload_cols = list(range(0, 48))
    model = HeterogeneousChallenger(input_dim, payload_cols=payload_cols, hidden=64, heads=2, layers=2)
    model.eval()

    L = 20
    x_zero = torch.zeros(1, L, input_dim)
    v = torch.ones(1, L, dtype=torch.bool)
    fams = torch.randint(0, 4, (1, L), dtype=torch.long)

    logits = []
    with torch.inference_mode():
        for _ in range(50):
            m = (torch.rand(1, L) > 0.5)
            if not m.any():
                m[0, 0] = True
            out = model(x_zero, v, fams, hard_mask=m)
            logits.append(float(out["bot_logit"][0]))

    logits = np.array(logits)
    assert np.std(logits) < 1e-6, f"Challenger violated mask invariance: std={np.std(logits)}"


def test_sparse_gradient_isolation():
    """Verify that unselected evidence tokens receive zero gradient."""
    input_dim = 64
    payload_cols = list(range(0, 48))
    model = ForwardEvidenceMiner(input_dim, payload_cols=payload_cols, hidden=64, pair_dim=16)
    model.train()

    L = 10
    x = torch.randn(1, L, input_dim, requires_grad=True)
    v = torch.ones(1, L, dtype=torch.bool)
    fams = torch.randint(0, 4, (1, L), dtype=torch.long)

    # Force selection of only tokens 0, 1, 2
    hard_mask = torch.zeros(1, L, dtype=torch.bool)
    hard_mask[0, :3] = True

    out = model(x, v, fams, hard_mask=hard_mask)
    loss = out["bot_logit"].sum()
    loss.backward()

    # Unselected tokens 3..9 must have exactly zero gradients
    unselected_grad = x.grad[0, 3:].abs().sum().item()
    selected_grad = x.grad[0, :3].abs().sum().item()

    assert unselected_grad == 0.0, f"Leakage: unselected tokens received gradient: {unselected_grad}"
    assert selected_grad > 0.0, "Selected tokens failed to receive gradients."


def test_challenger_permutation_invariance():
    """Verify Challenger output is invariant to permutation of input evidence tokens."""
    input_dim = 64
    payload_cols = list(range(0, 48))
    model = HeterogeneousChallenger(input_dim, payload_cols=payload_cols, hidden=64, heads=2, layers=2)
    model.eval()

    L = 12
    x = torch.randn(1, L, input_dim)
    v = torch.ones(1, L, dtype=torch.bool)
    fams = torch.randint(0, 4, (1, L), dtype=torch.long)

    with torch.inference_mode():
        base_out = model(x, v, fams, mode="all")
        base_logit = float(base_out["bot_logit"][0])

        perm = torch.randperm(L)
        x_perm = x[:, perm]
        fams_perm = fams[:, perm]
        perm_out = model(x_perm, v, fams_perm, mode="all")
        perm_logit = float(perm_out["bot_logit"][0])

    assert abs(base_logit - perm_logit) < 1e-4, f"Permutation mismatch: {base_logit} vs {perm_logit}"


if __name__ == "__main__":
    test_payload_auto_resolution()
    test_mask_invariance_forward_miner()
    test_mask_invariance_heterogeneous_challenger()
    test_sparse_gradient_isolation()
    test_challenger_permutation_invariance()
    print("ALL 5 CORE INVARIANT UNIT TESTS PASSED SUCCESSFULLY!")
