"""Train a HypBot native-edge variant on the same nodes, inputs and split.

This is a graph-controlled variant of the public HypBot implementation, not
the paper's official kNN-graph configuration. It symmetrizes observed reply
and retweet edges for mean neighbor aggregation; model/training presets stay
the same as the existing paired exporter.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate0_shallow_tree import sha256_file
from train_v2_detector import file_sha256, hash_bytes, load_data


SOURCE = {
    "botsim": Path("data/processed/botsim_v3/reply_sources.jsonl.gz"),
    "fox8": Path("data/processed/fox8_v3/interaction_event_index.json"),
}
SOURCE_MANIFEST = {
    "botsim": Path("data/processed/botsim_v3/manifest.json"),
    "fox8": Path("data/processed/fox8_v3/native_interaction_facts.json"),
}


def native_symmetric_graph(root: Path, dataset: str,
                           account_ids: list[str]) -> tuple[torch.Tensor, dict]:
    source = root / SOURCE[dataset]
    manifest = json.loads((root / SOURCE_MANIFEST[dataset]).read_text())
    expected = (manifest["reply_sources_sha256"] if dataset == "botsim"
                else manifest["event_index_sha256"])
    if sha256_file(source) != expected:
        raise ValueError("Native relation source index differs from audited manifest")
    index = {uid: pos for pos, uid in enumerate(account_ids)}
    if len(index) != len(account_ids):
        raise ValueError("Source account IDs are not unique")
    observed = set()
    if dataset == "botsim":
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                u = index.get(str(item["source_account_id"]))
                v = index.get(str(item["target_account_id"]))
                if u is not None and v is not None and u != v:
                    observed.add((u, v))
        if len(observed) != manifest["counts"]["unique_directed_reply_edges"]:
            raise ValueError("BotSim native edge population differs")
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        for item in payload["events"]:
            u = index.get(str(item["source_id"]))
            v = index.get(str(item["target_id"]))
            if u is not None and v is not None and u != v:
                observed.add((u, v))
        if len(observed) > manifest["native_typed_pairs"]:
            raise ValueError("Fox8 native edge population exceeds typed pairs")
    symmetric = sorted(observed | {(v, u) for u, v in observed})
    if not symmetric:
        raise ValueError("Native relation graph is empty")
    edge = np.asarray(symmetric, dtype=np.int64).T.copy()
    return torch.from_numpy(edge), {
        "source_index_sha256": sha256_file(source),
        "source_manifest_sha256": sha256_file(root / SOURCE_MANIFEST[dataset]),
        "observed_directed_pairs": len(observed),
        "symmetric_directed_pairs": len(symmetric),
        "graph_sha256": hash_bytes(edge.tobytes()),
        "graph_operation": "unique nonself native account pairs, symmetrized for mean aggregation",
    }


def train_one(args: argparse.Namespace, seed: int) -> dict:
    sys.path.insert(0, str(args.hypbot_repo.resolve()))
    from hypbot.data import mean_adjacency
    from hypbot.model import HypBot
    from train import PRESETS, fit, select_threshold, set_seed

    set_seed(seed)
    baseline_path = args.root / f"output/hypbot_{args.dataset}_paired/seed{seed}/test_predictions.json"
    baseline = json.loads(baseline_path.read_text())
    loader_args = argparse.Namespace(
        dataset=args.dataset, data_dir=args.data_dir,
        account_file=args.account_file, raw_source=[args.root / SOURCE[args.dataset]],
        knn_k=10)
    x, y_np, split, ids, _, _, hashes, _ = load_data(loader_args, seed)
    prior = baseline["manifest"]
    if (hashes["feature_sha256"] != prior["feature_sha256"]
            or hashes["split_sha256"] != prior["split_sha256"]
            or [ids[int(i)] for i in split["test"]] !=
            [row["account_id"] for row in baseline["predictions"]]):
        raise ValueError("HypBot native graph variant does not share inputs/split")
    native_edge, graph_meta = native_symmetric_graph(args.root, args.dataset, ids)
    device = torch.device(args.device)
    adj = mean_adjacency(native_edge, len(ids)).to(device)
    x = x.to(device)
    y = torch.as_tensor(y_np, device=device)
    tr = torch.as_tensor(split["train"])
    va = torch.as_tensor(split["val"])
    cfg = PRESETS[args.dataset]
    model = HypBot(x.shape[1], K=prior["num_sectors"], gated=False,
                   **cfg["model"], graph_width=x.shape[1],
                   sparse_adj=True).to(device)
    model = fit(model, x, y, adj, tr, va, cfg["train"])
    probability = model.predict_prob(x, adj).cpu().numpy()
    threshold = select_threshold(probability[split["val"]], y_np[split["val"]])
    output = args.output / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    torch.save({"model_state": model.state_dict(), "dataset": args.dataset,
                "seed": seed, "threshold": threshold,
                "num_sectors": prior["num_sectors"],
                "graph_meta": graph_meta,
                "feature_sha256": hashes["feature_sha256"],
                "split_sha256": hashes["split_sha256"]}, checkpoint)
    manifest_out = {
        "dataset": args.dataset, "seed": seed,
        "graph_variant": "native_symmetric_public_HypBot_implementation",
        "official_paper_graph_configuration": False,
        "official_repo": str(args.hypbot_repo.resolve()),
        "node_feature_sha256": hashes["feature_sha256"],
        "split_sha256": hashes["split_sha256"],
        "native_graph": graph_meta,
        "threshold": threshold,
        "num_sectors": prior["num_sectors"],
        "model_sha256": file_sha256(checkpoint),
        "paired_knn_baseline_predictions_sha256": sha256_file(baseline_path),
    }
    rows = [{"account_id": ids[int(i)], "label": int(y_np[i]),
             "probability": float(probability[i]), "split": "test"}
            for i in split["test"]]
    predictions_path = output / "test_predictions.json"
    predictions_path.write_text(json.dumps({"manifest": manifest_out,
                                             "predictions": rows}, indent=2) + "\n",
                                encoding="utf-8")
    acc = float(np.mean((probability[split["test"]] >= threshold) ==
                        y_np[split["test"]]))
    return {"seed": seed, "test_acc": acc,
            "test_accounts": len(rows),
            "checkpoint_sha256": file_sha256(checkpoint),
            "predictions_sha256": sha256_file(predictions_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("botsim", "fox8"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.root = Path(__file__).resolve().parents[2]
    args.hypbot_repo = Path("D:/Code/SAHG/SAHG/HypBot")
    args.data_dir = Path(f"D:/Code/SAHG/SAHG/code/data/{'botsim_24' if args.dataset == 'botsim' else 'fox8_23'}")
    args.account_file = (args.root / "data/raw/botsim_24/Users.csv"
                         if args.dataset == "botsim" else None)
    args.output = args.root / f"output/hypbot_{args.dataset}_native_graph_paired"
    torch.set_num_threads(4)
    runs = [train_one(args, seed) for seed in args.seeds]
    report = {"dataset": args.dataset,
              "scope": "same_node_features_split_native_graph_hypbot_variant",
              "official_paper_graph_configuration": False,
              "runs": runs,
              "mean_test_acc": float(np.mean([row["test_acc"] for row in runs]))}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
