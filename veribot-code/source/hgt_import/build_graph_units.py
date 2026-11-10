"""Build GRAPH_xx units (design doc section 1) from the clean HGT's REAL attention weights.

Pipeline per target account:
  1. One forward pass over the full graph (already-trained, frozen clean_hgt_best.pt).
  2. For the target's incoming "follows" edges (= its followers) and incoming "rev_follows" edges
     (= its following list), read off the last layer's real per-edge attention weight (mean over
     heads) from VerifiableHGTConv.saved_alpha, matched to edges via saved_index -- confirmed by
     extract_hgt_attention.py to have full coverage and correct direction semantics before this
     script was written.
  3. Rank ALL of the target's neighbors (both directions pooled together, matching the design doc's
     "among the neighbors the upstream classifier attended to most strongly") by that real attention
     weight, keep top-K (K=5 per design doc 1.2 Step B).
  4. For each of those top-K neighbors, read the SAME model's own predicted label for that neighbor
     node (bot_logit > frozen_threshold) -- also a real model output, not an external label lookup.
  5. Aggregate into a digit-free GRAPH_xx text using the design doc's fixed tier words: bot fraction
     among top-K > 0.6 -> "majority", < 0.4 -> "a small minority", else "about half". These
     thresholds are fixed constants (see extract_hgt_attention.TIER_THRESHOLDS), not fit per-batch.

Real neighbor account_ids are used internally (needed to look up each neighbor's own model
prediction and, for the audit step, to let a human check against ground truth) but NEVER appear in
the GRAPH_xx text itself -- only aggregate tier words do, per the design doc's "no digits, no raw
IDs" requirement for the0.6B-visible text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from hgt_model import VerifiableHGT  # noqa: E402

DATA_DIR = HERE / "data"
CKPT_DIR = HERE / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOP_K = 5
MAJORITY_MIN = 0.6
MINORITY_MAX = 0.4


def load_everything():
    data = torch.load(DATA_DIR / "clean_train_only_graph.pt", map_location="cpu", weights_only=False)
    ckpt = torch.load(CKPT_DIR / "clean_hgt_best.pt", map_location="cpu", weights_only=False)
    model = VerifiableHGT(
        metadata=ckpt["metadata"], in_channels=ckpt["in_channels"],
        hidden_channels=ckpt["config"]["hidden_channels"], out_channels=1,
        num_layers=ckpt["config"]["num_layers"], heads=ckpt["config"]["heads"],
        dropout=ckpt["config"]["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
    return model, data, x_dict, edge_index_dict, ckpt["frozen_threshold"]


def bot_fraction_tier(bot_frac: float) -> str:
    if bot_frac > MAJORITY_MIN:
        return "majority"
    if bot_frac < MINORITY_MAX:
        return "a small minority"
    return "about half"


def run_forward_and_collect_attention(model, x_dict, edge_index_dict):
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
    probs = torch.sigmoid(out["bot_logit"]).cpu()
    last_conv = model.convs[-1]
    alpha = last_conv.saved_alpha.mean(dim=-1).cpu()  # [total_edges_all_relations]
    idx = last_conv.saved_index.cpu()                  # dst node index per position in alpha
    return probs, alpha, idx


def build_relation_edge_alpha(edge_index_dict, alpha, idx, relation_key):
    """Return (src_tensor, dst_tensor, alpha_tensor) for exactly this relation's edges, sliced out
    of the concatenated (alpha, idx) using the same offset logic verified in
    extract_hgt_attention.verify_saved_alpha_coverage (relations concatenated in edge_index_dict's
    iteration order)."""
    offset = 0
    for key, edges in edge_index_dict.items():
        n_edges = edges.shape[1]
        if key == relation_key:
            edge_cpu = edges.cpu()
            alpha_slice = alpha[offset:offset + n_edges]
            idx_slice = idx[offset:offset + n_edges]
            # sanity: idx_slice must equal edge_cpu[1] (dst) in the same order
            assert torch.equal(idx_slice, edge_cpu[1]), \
                f"alpha/idx slice misaligned with edge_index for {relation_key}"
            return edge_cpu[0], edge_cpu[1], alpha_slice
        offset += n_edges
    raise KeyError(relation_key)


def top_k_neighbors_for_target(follows_src, follows_dst, follows_alpha,
                                rev_src, rev_dst, rev_alpha, target_idx: int, k: int):
    """Pool followers (from 'follows' incoming) and following (from 'rev_follows' incoming) by
    UNIQUE neighbor account, not by edge. A mutual connection (X follows target AND target follows
    X) produces one 'follows' incoming edge and one 'rev_follows' incoming edge for the SAME
    neighbor X -- treating those as two separate top-K slots would double-count that single
    account's bot/human prediction in bot_fraction_tier, inflating the apparent bot ratio. Each
    unique neighbor contributes exactly once, tagged "mutual" when both directions exist, using the
    LARGER of the two directions' attention weights as its ranking key (the model's strongest
    signal about that neighbor, not an arbitrary pick)."""
    by_neighbor: dict[int, dict] = {}
    follower_mask = follows_dst == target_idx
    for src, a in zip(follows_src[follower_mask].tolist(), follows_alpha[follower_mask].tolist()):
        by_neighbor[src] = {"neighbor_idx": src, "directions": {"follower": a}}
    following_mask = rev_dst == target_idx
    for src, a in zip(rev_src[following_mask].tolist(), rev_alpha[following_mask].tolist()):
        if src in by_neighbor:
            by_neighbor[src]["directions"]["following"] = a
        else:
            by_neighbor[src] = {"neighbor_idx": src, "directions": {"following": a}}

    candidates = []
    for src, rec in by_neighbor.items():
        dirs = rec["directions"]
        if len(dirs) == 2:
            direction = "mutual"
            max_attn = max(dirs.values())
        else:
            direction, max_attn = next(iter(dirs.items()))
        candidates.append((max_attn, src, direction, dirs))
    candidates.sort(key=lambda t: -t[0])
    return candidates[:k]


def main():
    model, data, x_dict, edge_index_dict, frozen_threshold = load_everything()
    probs, alpha, idx = run_forward_and_collect_attention(model, x_dict, edge_index_dict)
    preds = (probs >= frozen_threshold).int()

    follows_src, follows_dst, follows_alpha = build_relation_edge_alpha(
        edge_index_dict, alpha, idx, ("user", "follows", "user"))
    rev_src, rev_dst, rev_alpha = build_relation_edge_alpha(
        edge_index_dict, alpha, idx, ("user", "rev_follows", "user"))

    account_ids = data["user"].account_id
    n = len(account_ids)

    graph_facts_by_account: dict[str, dict] = {}
    n_with_neighbors = 0
    for target_idx in range(n):
        top_k = top_k_neighbors_for_target(follows_src, follows_dst, follows_alpha,
                                            rev_src, rev_dst, rev_alpha, target_idx, TOP_K)
        if not top_k:
            continue
        n_with_neighbors += 1
        neighbor_records = []
        bot_count = 0
        for attn, neighbor_idx, direction, dirs in top_k:
            neighbor_pred = int(preds[neighbor_idx].item())
            bot_count += neighbor_pred  # each unique neighbor contributes exactly once, see
                                         # top_k_neighbors_for_target's mutual-dedup docstring
            neighbor_records.append({
                "neighbor_account_id": account_ids[neighbor_idx],  # sidecar only, never in GRAPH_xx text
                "direction": direction,  # "follower" | "following" | "mutual"
                "per_direction_attention": dirs,  # raw attention for each direction that exists
                "ranking_attention": attn,  # max(dirs.values()), what top-K was sorted by
                "predicted_label": "bot" if neighbor_pred == 1 else "human",
            })
        bot_frac = bot_count / len(top_k)
        top1 = neighbor_records[0]
        top1_edge_desc = ("mutual follow" if top1["direction"] == "mutual"
                           else top1["direction"])
        graph_facts_by_account[account_ids[target_idx]] = {
            "top_k_neighbors": neighbor_records,  # sidecar (has real ids) -- not model-visible
            "bot_fraction_tier": bot_fraction_tier(bot_frac),
            "bot_count_among_top_k": bot_count,
            "top_k_size": len(top_k),
            "top1_direction": top1["direction"],
            "top1_predicted_label": top1["predicted_label"],
            "graph_01_text": f"Among the neighbors the upstream classifier attended to most "
                              f"strongly, {bot_fraction_tier(bot_frac)} are themselves predicted "
                              f"as bot.",
            "graph_02_text": f"The single most attended-to neighbor (edge type: "
                              f"{top1_edge_desc}) is predicted {top1['predicted_label']}.",
        }

    print(f"accounts with >=1 graph neighbor (GRAPH_xx eligible): {n_with_neighbors}/{n}")

    sidecar_path = DATA_DIR / "graph_facts_sidecar.json"
    with open(sidecar_path, "w") as f:
        json.dump(graph_facts_by_account, f, indent=2)
    print(f"sidecar (has real neighbor ids, internal use only) -> {sidecar_path}")

    model_visible_path = DATA_DIR / "graph_units_model_visible.json"
    model_visible = {
        uid: {"GRAPH_01": v["graph_01_text"], "GRAPH_02": v["graph_02_text"]}
        for uid, v in graph_facts_by_account.items()
    }
    with open(model_visible_path, "w") as f:
        json.dump(model_visible, f, indent=2)
    print(f"model-visible GRAPH_xx text (no ids, no digits) -> {model_visible_path}")


if __name__ == "__main__":
    main()
