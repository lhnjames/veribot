"""Extract model-visible GRAPH facts and upstream verdicts from real HGT attention.

Works for either source and scales to the full TwiBot-22 graph.  Neighbor candidates are ranked by
the last HGT layer's actual attention, pooled across follower/following relations, and deduplicated
by real neighbor identity.  The model-visible JSONL contains no raw ids, weights, counts, or digits;
the separate audit JSONL retains those values for independent recomputation.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from hgt_model import VerifiableHGT  # noqa: E402

TOP_K = 5


def tier(bot_fraction: float) -> str:
    if bot_fraction > 0.6:
        return "majority"
    if bot_fraction < 0.4:
        return "a small minority"
    return "about half"


def relation_slice(edge_dict, alpha, saved_dst, wanted):
    offset = 0
    for key, edge in edge_dict.items():
        count = edge.shape[1]
        if key == wanted:
            rel_alpha = alpha[offset:offset + count]
            rel_dst = saved_dst[offset:offset + count]
            if not torch.equal(rel_dst, edge[1]):
                raise RuntimeError(f"saved attention destination indices do not align for {wanted}")
            return edge[0], edge[1], rel_alpha
        offset += count
    raise KeyError(wanted)


def deduplicated_topk(
    n: int,
    follower_src: torch.Tensor,
    follower_dst: torch.Tensor,
    follower_alpha: torch.Tensor,
    following_src: torch.Tensor,
    following_dst: torch.Tensor,
    following_alpha: torch.Tensor,
):
    device = follower_src.device
    targets = torch.cat([follower_dst, following_dst])
    neighbors = torch.cat([follower_src, following_src])
    attention = torch.cat([follower_alpha, following_alpha]).float()
    direction = torch.cat([
        torch.zeros_like(follower_src, dtype=torch.long),
        torch.ones_like(following_src, dtype=torch.long),
    ])
    keys = targets.to(torch.long) * n + neighbors.to(torch.long)
    order = torch.argsort(keys)
    keys = keys[order]
    attention = attention[order]
    direction = direction[order]
    unique_keys, counts = torch.unique_consecutive(keys, return_counts=True)
    groups = torch.repeat_interleave(
        torch.arange(unique_keys.numel(), device=device), counts
    )
    maximum = torch.full(
        (unique_keys.numel(),), -torch.inf, device=device, dtype=torch.float32
    )
    maximum.scatter_reduce_(0, groups, attention, reduce="amax", include_self=True)
    negative_inf = torch.full_like(attention, -torch.inf)
    follower_weight = torch.full_like(maximum, -torch.inf)
    following_weight = torch.full_like(maximum, -torch.inf)
    follower_weight.scatter_reduce_(
        0, groups, torch.where(direction == 0, attention, negative_inf),
        reduce="amax", include_self=True,
    )
    following_weight.scatter_reduce_(
        0, groups, torch.where(direction == 1, attention, negative_inf),
        reduce="amax", include_self=True,
    )
    direction_bits = torch.zeros(unique_keys.numel(), device=device, dtype=torch.long)
    direction_bits.scatter_add_(0, groups, torch.where(direction == 0, 1, 2))
    unique_targets = torch.div(unique_keys, n, rounding_mode="floor")
    unique_neighbors = unique_keys.remainder(n)

    # Stable two-pass sort gives target ascending, then attention descending within each target.
    by_attention = torch.argsort(maximum, descending=True, stable=True)
    target_order = torch.argsort(unique_targets[by_attention], stable=True)
    ranked = by_attention[target_order]
    sorted_targets = unique_targets[ranked]
    target_counts = torch.bincount(sorted_targets, minlength=n)
    starts = torch.cumsum(target_counts, 0) - target_counts
    ranks = torch.arange(ranked.numel(), device=device) - torch.repeat_interleave(starts, target_counts)
    ranked = ranked[ranks < TOP_K]
    return {
        "target": unique_targets[ranked].cpu(),
        "neighbor": unique_neighbors[ranked].cpu(),
        "attention": maximum[ranked].cpu(),
        "direction_bits": direction_bits[ranked].cpu(),
        "follower_attention": follower_weight[ranked].cpu(),
        "following_attention": following_weight[ranked].cpu(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("twibot20", "twibot22"), required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("attention extraction is permitted only on the remote CUDA server")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    data = torch.load(args.graph, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = VerifiableHGT(
        metadata=checkpoint["metadata"], in_channels=checkpoint["in_channels"],
        hidden_channels=checkpoint["config"]["hidden_channels"], out_channels=1,
        num_layers=checkpoint["config"]["num_layers"], heads=checkpoint["config"]["heads"],
        dropout=checkpoint["config"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    x_dict = {key: value.to(device=device, dtype=torch.float32) for key, value in data.x_dict.items()}
    edge_dict = {key: value.to(device) for key, value in data.edge_index_dict.items()}
    with torch.inference_mode():
        output = model(x_dict, edge_dict)
    probs = torch.sigmoid(output["bot_logit"]).float().cpu()
    threshold = float(checkpoint["frozen_threshold"])
    predictions = (probs >= threshold).long()
    last = model.convs[-1]
    if last.saved_alpha is None or last.saved_index is None:
        raise RuntimeError("HGT layer did not expose attention")
    total_edges = sum(edge.shape[1] for edge in edge_dict.values())
    if last.saved_index.numel() != total_edges:
        raise RuntimeError(
            f"attention coverage mismatch: saved={last.saved_index.numel()} expected={total_edges}"
        )
    alpha = last.saved_alpha.mean(-1)
    saved_dst = last.saved_index
    follower_src, follower_dst, follower_alpha = relation_slice(
        edge_dict, alpha, saved_dst, ("user", "follows", "user")
    )
    following_src, following_dst, following_alpha = relation_slice(
        edge_dict, alpha, saved_dst, ("user", "rev_follows", "user")
    )
    top = deduplicated_topk(
        len(predictions), follower_src, follower_dst, follower_alpha,
        following_src, following_dst, following_alpha,
    )
    del model, x_dict, edge_dict, output, alpha, saved_dst
    torch.cuda.empty_cache()

    n = len(predictions)
    account_ids = data["user"].account_id
    top_counts = torch.bincount(top["target"], minlength=n)
    offsets = torch.cumsum(top_counts, 0) - top_counts
    visible_path = args.output / "graph_units_model_visible.jsonl"
    audit_path = args.output / "graph_facts_sidecar.jsonl"
    sample_rows = []
    rng = random.Random(42)
    sample_targets = set(rng.sample(torch.nonzero(top_counts > 0).flatten().tolist(), k=min(20, int((top_counts > 0).sum()))))
    eligible = 0
    with visible_path.open("w", encoding="utf-8") as visible, audit_path.open("w", encoding="utf-8") as audit:
        for target in range(n):
            start = int(offsets[target])
            count = int(top_counts[target])
            stop = start + count
            verdict = "bot" if int(predictions[target]) else "human"
            neighbor_rows = []
            bot_count = 0
            for position in range(start, stop):
                neighbor_idx = int(top["neighbor"][position])
                neighbor_verdict = "bot" if int(predictions[neighbor_idx]) else "human"
                bot_count += int(neighbor_verdict == "bot")
                bits = int(top["direction_bits"][position])
                direction = "mutual" if bits == 3 else ("follower" if bits == 1 else "following")
                row = {
                    "rank": position - start + 1,
                    "neighbor_account_id": account_ids[neighbor_idx],
                    "direction": direction,
                    "ranking_attention": float(top["attention"][position]),
                    "predicted_label": neighbor_verdict,
                }
                follower_weight = float(top["follower_attention"][position])
                following_weight = float(top["following_attention"][position])
                if follower_weight != float("-inf"):
                    row["follower_attention"] = follower_weight
                if following_weight != float("-inf"):
                    row["following_attention"] = following_weight
                neighbor_rows.append(row)
            facts = []
            if count:
                eligible += 1
                bot_tier = tier(bot_count / count)
                top_one = neighbor_rows[0]
                edge_desc = "mutual follow" if top_one["direction"] == "mutual" else top_one["direction"]
                facts = [
                    "Among the neighbors the upstream classifier attended to most strongly, "
                    f"{bot_tier} are themselves predicted as bot.",
                    "The single most attended-to neighbor (edge type: "
                    f"{edge_desc}) is predicted {top_one['predicted_label']}.",
                ]
                if any(any(ch.isdigit() for ch in fact) for fact in facts):
                    raise RuntimeError(f"model-visible GRAPH text contains a digit: {facts}")
            visible.write(json.dumps({
                "source": args.source, "account_id": account_ids[target],
                "upstream_verdict": verdict, "graph_facts": facts,
            }, ensure_ascii=False) + "\n")
            audit_row = {
                "source": args.source, "account_id": account_ids[target],
                "target_probability": float(probs[target]), "upstream_verdict": verdict,
                "frozen_threshold": threshold, "top_neighbors": neighbor_rows,
                "graph_facts": facts,
            }
            audit.write(json.dumps(audit_row, ensure_ascii=False) + "\n")
            if target in sample_targets:
                sample_rows.append(audit_row)
            if (target + 1) % 100000 == 0:
                print(f"wrote {target + 1}/{n} accounts", flush=True)

    report = {
        "source": args.source, "graph": str(args.graph), "checkpoint": str(args.checkpoint),
        "n_accounts": n, "n_graph_eligible": eligible,
        "graph_eligible_fraction": eligible / n, "top_k": TOP_K,
        "attention_coverage": {"saved_edges": total_edges, "expected_edges": total_edges},
        "direction_mapping": {
            "incoming_follows": "follower",
            "incoming_rev_follows": "following",
            "mutual_neighbors": "deduplicated; maximum directional attention used for ranking",
        },
        "frozen_threshold": threshold,
        "model_visible_path": str(visible_path), "audit_sidecar_path": str(audit_path),
    }
    (args.output / "graph_sidecar_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "graph_manual_audit_sample.json").write_text(
        json.dumps(sample_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
