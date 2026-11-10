"""Exploratory, relation-matched edge deletion audit on HGT validation accounts.

This tests model dependence under one specific intervention. It does not turn
attention into behavioral evidence or estimate a real-world causal effect.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from hgt_model import VerifiableHGT  # noqa: E402

FOLLOWS = ("user", "follows", "user")
REVERSE = ("user", "rev_follows", "user")


def validation_uid(uid: str) -> bool:
    key = f"hgt_val_v1:{uid}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 20 == 0


def relation_edges(edge_dict, alpha, saved_dst, wanted):
    offset = 0
    for key, edge in edge_dict.items():
        count = edge.shape[1]
        if key == wanted:
            a = alpha[offset:offset + count]
            dst = saved_dst[offset:offset + count]
            if not torch.equal(dst, edge[1]):
                raise RuntimeError(f"attention mismatch for {wanted}")
            return edge[0].cpu(), edge[1].cpu(), a.cpu()
        offset += count
    raise KeyError(wanted)


def without_social_edge(edge_dict, original_src: int, original_dst: int):
    fwd = edge_dict[FOLLOWS]
    rev = edge_dict[REVERSE]
    keep_fwd = ~((fwd[0] == original_src) & (fwd[1] == original_dst))
    keep_rev = ~((rev[0] == original_dst) & (rev[1] == original_src))
    if int((~keep_fwd).sum()) != 1 or int((~keep_rev).sum()) != 1:
        raise RuntimeError("expected exactly one observed edge and its mirror")
    return {FOLLOWS: fwd[:, keep_fwd], REVERSE: rev[:, keep_rev]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-targets", type=int, default=80)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("run this audit on the remote CUDA server")

    data = torch.load(args.graph, map_location="cpu", weights_only=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = VerifiableHGT(
        metadata=ckpt["metadata"], in_channels=ckpt["in_channels"],
        hidden_channels=ckpt["config"]["hidden_channels"], out_channels=1,
        num_layers=ckpt["config"]["num_layers"], heads=ckpt["config"]["heads"],
        dropout=ckpt["config"]["dropout"],
    ).cuda().eval()
    model.load_state_dict(ckpt["model_state"])
    x_dict = {key: value.cuda() for key, value in data.x_dict.items()}
    edge_dict = {key: value.cuda() for key, value in data.edge_index_dict.items()}

    with torch.inference_mode():
        original = model(x_dict, edge_dict)["bot_logit"].detach().cpu()
    conv = model.convs[-1]
    alpha = conv.saved_alpha.mean(-1)
    saved_dst = conv.saved_index
    if saved_dst.numel() != sum(edge.shape[1] for edge in edge_dict.values()):
        raise RuntimeError("saved attention does not cover all edges")

    groups = defaultdict(list)
    for relation in (FOLLOWS, REVERSE):
        src, dst, weights = relation_edges(edge_dict, alpha, saved_dst, relation)
        for neighbor, target, weight in zip(src.tolist(), dst.tolist(), weights.tolist()):
            groups[(target, relation)].append((weight, neighbor))

    ids = data["user"].account_id
    val_targets = [i for i, uid in enumerate(ids)
                   if validation_uid(uid) and not bool(data["user"].holdout_mask[i])]
    candidates = []
    for target in val_targets:
        choices = []
        for relation in (FOLLOWS, REVERSE):
            ranked = sorted(groups[(target, relation)], reverse=True)
            if len(ranked) >= 3 and ranked[0][0] > ranked[-1][0] + 1e-8:
                choices.append((len(ranked), relation, ranked))
        if choices:
            choices.sort(key=lambda entry: entry[0], reverse=True)
            _, relation, ranked = choices[0]
            candidates.append((target, relation, ranked))
    rng = random.Random(42)
    rng.shuffle(candidates)
    selected = candidates[:args.max_targets]
    threshold = float(ckpt["frozen_threshold"])
    rows = []
    for done, (target, relation, ranked) in enumerate(selected, 1):
        top_weight, top_neighbor = ranked[0]
        bottom_weight, bottom_neighbor = ranked[-1]

        def perturb(neighbor: int) -> float:
            if relation == FOLLOWS:
                src, dst = neighbor, target
            else:
                src, dst = target, neighbor
            modified = without_social_edge(edge_dict, src, dst)
            with torch.inference_mode():
                return float(model(x_dict, modified)["bot_logit"][target].cpu())

        top_logit = perturb(top_neighbor)
        bottom_logit = perturb(bottom_neighbor)
        original_logit = float(original[target])
        original_prob = float(torch.sigmoid(original[target]))
        top_prob = float(torch.sigmoid(torch.tensor(top_logit)))
        bottom_prob = float(torch.sigmoid(torch.tensor(bottom_logit)))
        rows.append({
            "account_id": ids[target], "relation": relation[1],
            "n_incoming_edges_same_relation": len(ranked),
            "original_probability": original_prob,
            "top_attention": top_weight, "bottom_attention": bottom_weight,
            "top_abs_logit_delta": abs(top_logit - original_logit),
            "bottom_abs_logit_delta": abs(bottom_logit - original_logit),
            "top_abs_probability_delta": abs(top_prob - original_prob),
            "bottom_abs_probability_delta": abs(bottom_prob - original_prob),
            "top_label_flip": (top_prob >= threshold) != (original_prob >= threshold),
            "bottom_label_flip": (bottom_prob >= threshold) != (original_prob >= threshold),
        })
        if done % 10 == 0:
            print(f"audited {done}/{len(selected)} validation accounts", flush=True)

    n = len(rows)
    report = {
        "scope": "exploratory HGT model-dependence audit on hgt_val only",
        "intervention": "delete one observed directed follows edge and its reverse-relation mirror; compare top and bottom last-layer attention within the same relation",
        "limitation": "single-edge deletion may be out of distribution; this is neither real-world causal evidence nor an explanation of a HGT/RelNet fused decision",
        "graph": str(args.graph), "checkpoint": str(args.checkpoint),
        "n_hgt_val": len(val_targets), "n_eligible": len(candidates), "n_audited": n,
        "top_larger_abs_logit_delta": sum(r["top_abs_logit_delta"] > r["bottom_abs_logit_delta"] for r in rows),
        "top_larger_abs_probability_delta": sum(r["top_abs_probability_delta"] > r["bottom_abs_probability_delta"] for r in rows),
        "mean_top_abs_logit_delta": sum(r["top_abs_logit_delta"] for r in rows) / n if n else None,
        "mean_bottom_abs_logit_delta": sum(r["bottom_abs_logit_delta"] for r in rows) / n if n else None,
        "top_label_flips": sum(r["top_label_flip"] for r in rows),
        "bottom_label_flips": sum(r["bottom_label_flip"] for r in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
