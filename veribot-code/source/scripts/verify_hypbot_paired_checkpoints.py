"""Recompute paired HypBot test probabilities from frozen checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

from train_v2_detector import file_sha256, load_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("botsim", "fox8", "mgtab"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for seed in (0, 1, 2):
        run = args.run_dir / f"seed{seed}"
        saved = json.loads((run / "test_predictions.json").read_text(encoding="utf-8"))
        manifest = saved["manifest"]
        checkpoint_path = run / "model.pt"
        if (manifest["dataset"] != args.dataset or manifest["seed"] != seed
                or file_sha256(checkpoint_path) != manifest["model_sha256"]):
            raise ValueError("HypBot checkpoint manifest differs")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (checkpoint["dataset"] != args.dataset or checkpoint["seed"] != seed
                or checkpoint["hashes"]["feature_sha256"] != manifest["feature_sha256"]
                or checkpoint["hashes"]["graph_sha256"] != manifest["graph_sha256"]
                or checkpoint["hashes"]["split_sha256"] != manifest["split_sha256"]
                or checkpoint["auto_k"] != manifest["auto_k"]
                or checkpoint["num_sectors"] != manifest["num_sectors"]
                or abs(checkpoint["threshold"] - manifest["threshold"]) > 1e-12):
            raise ValueError("HypBot checkpoint metadata differs")
        # load_data hashes this placeholder as raw_sha256; feature, graph and
        # split hashes below independently bind the actual inference tensors.
        placeholder = (args.data_dir / "features.pt" if args.dataset == "mgtab"
                       else args.data_dir / "features.csv")
        load_args = SimpleNamespace(
            dataset=args.dataset, data_dir=args.data_dir,
            account_file=args.account_file, knn_k=10, raw_source=[placeholder])
        x, labels, split, ids, edge, _, hashes, _ = load_data(load_args, seed)
        for key in ("feature_sha256", "graph_sha256", "split_sha256"):
            if hashes[key] != manifest[key]:
                raise ValueError(f"HypBot {key} differs from training tensor")
        repo = Path(manifest["official_repo"])
        if not (repo / "hypbot" / "model.py").is_file():
            raise FileNotFoundError("frozen HypBot model implementation missing")
        if str(repo.resolve()) not in sys.path:
            sys.path.insert(0, str(repo.resolve()))
        from hypbot.data import knn_graph, mean_adjacency
        from hypbot.model import HypBot
        from train import PRESETS
        if args.dataset == "mgtab":
            adj = mean_adjacency(edge, len(x))
            extra = {"graph_width": x.shape[1], "sparse_adj": True}
        else:
            adj = knn_graph(x, k=10)
            if not torch.equal(adj, edge):
                raise ValueError("HypBot kNN graph differs from saved training graph")
            extra = {}
        model = HypBot(x.shape[1], K=checkpoint["num_sectors"], gated=False,
                       **PRESETS[args.dataset]["model"], **extra)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        with torch.inference_mode():
            probability = model.predict_prob(x, adj).cpu().numpy()
        test = split["test"]
        if len(test) != len(saved["predictions"]):
            raise ValueError("HypBot test population differs")
        max_difference = 0.0
        decision_mismatch = 0
        for i, row in zip(test, saved["predictions"]):
            if ids[int(i)] != str(row["account_id"]) or labels[i] != row["label"]:
                raise ValueError("HypBot test account or label differs")
            value = float(probability[i])
            max_difference = max(max_difference, abs(value - row["probability"]))
            decision_mismatch += int((value >= manifest["threshold"]) !=
                                     (row["probability"] >= manifest["threshold"]))
        if max_difference > 1e-5 or decision_mismatch:
            raise ValueError("HypBot checkpoint does not reproduce saved test predictions")
        results.append({
            "seed": seed, "test_accounts": len(test),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "prediction_file_sha256": file_sha256(run / "test_predictions.json"),
            "max_probability_difference": max_difference,
            "decision_mismatches": decision_mismatch,
            "recomputed_acc": float(np.mean(
                (probability[test] >= manifest["threshold"]) == labels[test])),
        })
    result = {"dataset": args.dataset,
              "status": "all_paired_hypbot_checkpoints_replayed",
              "runs": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
