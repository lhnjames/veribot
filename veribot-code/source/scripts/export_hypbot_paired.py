"""Train the official local HypBot release and export paired test predictions.

Pass the local official HypBot checkout via --hypbot-repo. This wrapper uses
VeriBot's source-aligned account IDs and hashes, then checks that HypBot's own
kNN construction matches the shared graph before training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from train_v2_detector import file_sha256, load_data


def train_one(args, seed: int) -> dict:
    if not (args.hypbot_repo / "hypbot" / "model.py").exists():
        raise FileNotFoundError("--hypbot-repo must point to the official HypBot checkout")
    sys.path.insert(0, str(args.hypbot_repo.resolve()))
    from hypbot.data import knn_graph, mean_adjacency
    from hypbot.model import HypBot, eliminate_sectors
    from train import PRESETS, fit, select_threshold, set_seed

    set_seed(seed)
    x, y_np, split, ids, edge, _, hashes, _ = load_data(args, seed)
    device = torch.device(args.device)
    if args.dataset == "mgtab":
        adj = mean_adjacency(edge, len(x)).to(device)
        extra = {"graph_width": x.shape[1], "sparse_adj": True}
    else:
        hyp_edge = knn_graph(x, k=args.knn_k)
        if not torch.equal(hyp_edge, edge):
            raise ValueError("shared kNN graph differs from HypBot's own graph")
        adj = hyp_edge.to(device)
        extra = {}
    x = x.to(device)
    y = torch.as_tensor(y_np, device=device)
    tr = torch.as_tensor(split["train"])
    va = torch.as_tensor(split["val"])
    cfg = PRESETS[args.dataset]

    def build(k: int, gated: bool) -> HypBot:
        return HypBot(x.shape[1], K=k, gated=gated, **cfg["model"], **extra).to(device)

    if args.auto_k:
        truncation = args.truncation
        while True:
            set_seed(seed)
            model = fit(build(truncation, True), x, y, adj, tr, va, cfg["train"])
            k_node, k_graph = eliminate_sectors(model, x, adj, va, y[va.to(device)])
            if max(k_node, k_graph) <= truncation // 2 or truncation >= 256:
                break
            truncation *= 2
    else:
        model = fit(build(args.num_sectors, False), x, y, adj, tr, va, cfg["train"])

    probabilities = model.predict_prob(x, adj).cpu().numpy()
    threshold = select_threshold(probabilities[split["val"]], y_np[split["val"]])
    run_dir = args.output / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "model.pt"
    torch.save({"model_state": model.state_dict(), "seed": seed,
                "dataset": args.dataset, "threshold": threshold,
                "auto_k": args.auto_k, "num_sectors": args.num_sectors,
                "hashes": hashes}, checkpoint)
    manifest = {"dataset": args.dataset, "seed": seed, **hashes,
                "threshold": threshold, "model_sha256": file_sha256(checkpoint),
                "official_repo": str(args.hypbot_repo.resolve()),
                "auto_k": args.auto_k, "num_sectors": args.num_sectors}
    rows = [{"account_id": ids[int(i)], "label": int(y_np[i]),
             "probability": float(probabilities[i]), "split": "test"}
            for i in split["test"]]
    path = run_dir / "test_predictions.json"
    path.write_text(json.dumps({"manifest": manifest, "predictions": rows}, indent=2), encoding="utf-8")
    accuracy = float(np.mean((probabilities[split["test"]] >= threshold) == y_np[split["test"]]))
    return {"seed": seed, "test_acc": accuracy, "predictions": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypbot-repo", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mgtab", "fox8", "botsim"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--raw-source", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--auto-k", action="store_true")
    parser.add_argument("--truncation", type=int, default=16)
    parser.add_argument("--num-sectors", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    rows = [train_one(args, seed) for seed in args.seeds]
    print(json.dumps({"runs": rows, "mean_test_acc": float(np.mean([r["test_acc"] for r in rows]))}, indent=2))


if __name__ == "__main__":
    main()
