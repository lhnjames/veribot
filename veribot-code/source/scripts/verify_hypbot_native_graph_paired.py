"""Independently replay native-edge HypBot variants and pair with compact evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_split_aligned_predictions import compare
from export_hypbot_native_graph_paired import SOURCE, native_symmetric_graph
from gate0_shallow_tree import sha256_file
from train_v2_detector import load_data


SELECTED = {
    "botsim": "output/v3_computable/botsim_group_tail_12",
    "fox8": "output/v3_fox8_native_spline_l2/group_tail_20_selected",
}


def verify_dataset(root: Path, dataset: str) -> dict:
    repo = Path("D:/Code/SAHG/SAHG/HypBot")
    sys.path.insert(0, str(repo.resolve()))
    from hypbot.data import mean_adjacency
    from hypbot.model import HypBot
    from train import PRESETS

    data_dir = Path(f"D:/Code/SAHG/SAHG/code/data/{'botsim_24' if dataset == 'botsim' else 'fox8_23'}")
    loader_args = argparse.Namespace(
        dataset=dataset, data_dir=data_dir,
        account_file=(root / "data/raw/botsim_24/Users.csv"
                      if dataset == "botsim" else None),
        raw_source=[root / SOURCE[dataset]], knn_k=10)
    folder = root / f"output/hypbot_{dataset}_native_graph_paired"
    summary = json.loads((folder / "summary.json").read_text())
    if [r["seed"] for r in summary["runs"]] != [0, 1, 2]:
        raise ValueError("Native-edge HypBot summary lacks three fixed seeds")
    runs = []
    for seed in (0, 1, 2):
        run = folder / f"seed{seed}"
        checkpoint_path = run / "model.pt"
        prediction_path = run / "test_predictions.json"
        frozen = json.loads(prediction_path.read_text())
        manifest = frozen["manifest"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                weights_only=True)
        x, y, split, ids, _, _, hashes, _ = load_data(loader_args, seed)
        edge, graph_meta = native_symmetric_graph(root, dataset, ids)
        if (sha256_file(checkpoint_path) != manifest["model_sha256"]
                or sha256_file(prediction_path) != summary["runs"][seed]["predictions_sha256"]
                or graph_meta != manifest["native_graph"]
                or graph_meta != checkpoint["graph_meta"]
                or hashes["feature_sha256"] != manifest["node_feature_sha256"]
                or hashes["split_sha256"] != manifest["split_sha256"]
                or checkpoint["split_sha256"] != hashes["split_sha256"]):
            raise ValueError("Native-edge graph, feature, split or weight hash differs")
        cfg = PRESETS[dataset]
        model = HypBot(x.shape[1], K=manifest["num_sectors"], gated=False,
                       **cfg["model"], graph_width=x.shape[1],
                       sparse_adj=True)
        model.load_state_dict(checkpoint["model_state"])
        adjacency = mean_adjacency(edge, len(ids))
        probability = model.predict_prob(x, adjacency).cpu().numpy()
        zero_adjacency = torch.sparse_csr_tensor(
            torch.zeros(len(ids) + 1, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.float32), size=(len(ids), len(ids)))
        no_graph_probability = model.predict_prob(x, zero_adjacency).cpu().numpy()
        saved = frozen["predictions"]
        if len(saved) != len(split["test"]):
            raise ValueError("Native-edge test population differs")
        max_error = 0.0
        for position, index in enumerate(split["test"]):
            item = saved[position]
            if item["account_id"] != ids[int(index)] or item["label"] != int(y[index]):
                raise ValueError("Native-edge test account or label differs")
            max_error = max(max_error, abs(float(item["probability"]) -
                                           float(probability[index])))
        if max_error > 1e-6:
            raise ValueError("Native-edge frozen probability replay differs")
        test = split["test"]
        full_class = probability[test] >= manifest["threshold"]
        no_graph_class = no_graph_probability[test] >= manifest["threshold"]
        selected = root / SELECTED[dataset] / f"seed{seed}/test_predictions.json"
        paired = compare(prediction_path, selected)
        official_knn = root / f"output/hypbot_{dataset}_paired/seed{seed}/test_predictions.json"
        original = compare(official_knn, selected)
        if abs(paired["baseline_acc"] - summary["runs"][seed]["test_acc"]) > 1e-12:
            raise ValueError("Native-edge paired ACC differs")
        runs.append({
            "seed": seed, "test_accounts": len(saved),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions_sha256": sha256_file(prediction_path),
            "max_frozen_probability_replay_difference": max_error,
            "native_graph": graph_meta,
            "native_graph_input_mask": {
                "changed_test_classes": int(np.count_nonzero(full_class != no_graph_class)),
                "full_test_correct": int(np.count_nonzero(full_class == y[test])),
                "masked_test_correct": int(np.count_nonzero(no_graph_class == y[test])),
                "maximum_test_probability_change": float(np.max(np.abs(
                    probability[test] - no_graph_probability[test]))),
                "intervention": "replace native mean adjacency with all-zero adjacency at frozen inference",
            },
            "native_hypbot_acc": paired["baseline_acc"],
            "compact_veribot_acc": paired["candidate_acc"],
            "paired_veribot_minus_native_hypbot": paired["paired_acc_difference"],
            "veribot_only_correct": paired["candidate_correct_baseline_wrong"],
            "native_hypbot_only_correct": paired["baseline_correct_candidate_wrong"],
            "paired_bootstrap_95pct_ci": paired["paired_account_bootstrap_95pct_ci"],
            "official_knn_hypbot_acc": original["baseline_acc"],
        })
    return {"dataset": dataset,
            "comparison": "same account population, original node features and split for both HypBot configurations; VeriBot uses same native relation source but its own evidence inputs",
            "official_hypbot_paper_graph": "kNN for these two datasets",
            "native_graph_variant_is_official_paper_model": False,
            "mean_native_hypbot_acc": float(np.mean([r["native_hypbot_acc"] for r in runs])),
            "mean_compact_veribot_acc": float(np.mean([r["compact_veribot_acc"] for r in runs])),
            "runs": runs}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(4)
    report = {
        "scope": "BotSim_and_Fox8_native_graph_HypBot_variant_checkpoint_replay_and_paired_compact_comparison",
        "datasets": [verify_dataset(root, dataset) for dataset in ("botsim", "fox8")],
    }
    path = root / "research/certificates/hypbot_native_graph_variant_paired_replay.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"datasets": [
        {"dataset": d["dataset"], "native": d["mean_native_hypbot_acc"],
         "veribot": d["mean_compact_veribot_acc"],
         "paired": [(r["seed"], r["paired_veribot_minus_native_hypbot"])
                    for r in d["runs"]]}
        for d in report["datasets"]]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
