"""Design doc 1.4 item 2: cross-validate that GRAPH_xx's attention-based ranking actually tracks
something the model depends on, not just an arbitrary internal number. For >=3 accounts, remove the
Top-1 (highest-attention) neighbor's edges entirely (both the "follows" edge and its mirrored
"rev_follows" edge -- the same pair build_features_and_graph.py constructed together) and re-run a
full forward pass, comparing the resulting change in the target's predicted bot probability against
removing a LOW-attention neighbor's edges (from the bottom of that same target's ranked neighbor
list, not an unrelated random account). If attention is doing its job, the high-attention neighbor's
removal should perturb the prediction by noticeably more than the low-attention neighbor's removal,
on average across the accounts tested.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from hgt_model import VerifiableHGT  # noqa: E402
from build_graph_units import (  # noqa: E402
    load_everything, run_forward_and_collect_attention, build_relation_edge_alpha,
    top_k_neighbors_for_target,
)


def remove_edge_pair(edge_index_dict, a_idx: int, b_idx: int, device):
    """Remove the (a_idx -> b_idx) edge from 'follows' and its mirror (b_idx -> a_idx) from
    'rev_follows' -- exactly the pair build_features_and_graph.py always creates together for any
    single underlying "a follows b" fact, regardless of which direction (follower/following/mutual)
    a_idx and b_idx play for some third target node."""
    out = {}
    for key, edges in edge_index_dict.items():
        if key == ("user", "follows", "user"):
            keep = ~((edges[0] == a_idx) & (edges[1] == b_idx))
            out[key] = edges[:, keep]
        elif key == ("user", "rev_follows", "user"):
            keep = ~((edges[0] == b_idx) & (edges[1] == a_idx))
            out[key] = edges[:, keep]
        else:
            out[key] = edges
    return out


def main():
    model, data, x_dict, edge_index_dict, frozen_threshold = load_everything()
    probs0, alpha, idx = run_forward_and_collect_attention(model, x_dict, edge_index_dict)

    follows_src, follows_dst, follows_alpha = build_relation_edge_alpha(
        edge_index_dict, alpha, idx, ("user", "follows", "user"))
    rev_src, rev_dst, rev_alpha = build_relation_edge_alpha(
        edge_index_dict, alpha, idx, ("user", "rev_follows", "user"))

    account_ids = data["user"].account_id
    n = len(account_ids)

    # pick 3 accounts with a reasonably long neighbor list (>=4) so "top-1" and "bottom" are
    # clearly distinct positions, not the same single neighbor
    candidates = []
    for target_idx in range(n):
        top_k = top_k_neighbors_for_target(follows_src, follows_dst, follows_alpha,
                                            rev_src, rev_dst, rev_alpha, target_idx, k=1000)
        if len(top_k) >= 4:
            candidates.append((target_idx, top_k))
        if len(candidates) >= 3:
            break

    results = []
    for target_idx, ranked in candidates:
        top1_attn, top1_idx, top1_dir, _ = ranked[0]
        bottom_attn, bottom_idx, bottom_dir, _ = ranked[-1]

        edge_removed_top1 = remove_edge_pair(edge_index_dict, target_idx, top1_idx, None) \
            if top1_dir in ("follower", "mutual") else None
        # direction convention: "follower" means original edge is (top1_idx -> target_idx) in
        # 'follows'; "following" means original edge is (target_idx -> top1_idx) in 'follows'.
        # For "mutual" both exist -- remove both pairs.
        def masked_edges(neighbor_idx, direction):
            e = dict(edge_index_dict)
            if direction in ("follower", "mutual"):
                e = remove_edge_pair(e, neighbor_idx, target_idx, None)
            if direction in ("following", "mutual"):
                e = remove_edge_pair(e, target_idx, neighbor_idx, None)
            return e

        edge_top1_removed = masked_edges(top1_idx, top1_dir)
        edge_bottom_removed = masked_edges(bottom_idx, bottom_dir)

        with torch.no_grad():
            out_top1 = model(x_dict, edge_top1_removed)
            out_bottom = model(x_dict, edge_bottom_removed)
        prob_top1_removed = torch.sigmoid(out_top1["bot_logit"])[target_idx].item()
        prob_bottom_removed = torch.sigmoid(out_bottom["bot_logit"])[target_idx].item()
        prob_orig = probs0[target_idx].item()

        delta_top1 = abs(prob_orig - prob_top1_removed)
        delta_bottom = abs(prob_orig - prob_bottom_removed)

        results.append({
            "target_account": account_ids[target_idx],
            "n_neighbors": len(ranked),
            "orig_prob": prob_orig,
            "top1_neighbor": account_ids[top1_idx], "top1_direction": top1_dir,
            "top1_attention": top1_attn, "prob_after_removing_top1": prob_top1_removed,
            "delta_after_removing_top1": delta_top1,
            "bottom_neighbor": account_ids[bottom_idx], "bottom_direction": bottom_dir,
            "bottom_attention": bottom_attn, "prob_after_removing_bottom": prob_bottom_removed,
            "delta_after_removing_bottom": delta_bottom,
            "top1_delta_exceeds_bottom_delta": delta_top1 > delta_bottom,
        })

    print(json.dumps(results, indent=2))
    n_pass = sum(1 for r in results if r["top1_delta_exceeds_bottom_delta"])
    print(f"\n{n_pass}/{len(results)} accounts: removing the top-attention neighbor perturbed the "
          f"prediction MORE than removing the lowest-attention neighbor.")
    with open(HERE / "data" / "ablation_validation_report.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
