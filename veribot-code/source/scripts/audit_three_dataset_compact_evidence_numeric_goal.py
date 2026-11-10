"""Pair the frozen 12/20/20-concept evidence models with HypBot by account."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_three_dataset_current_numeric_goal import compare, load_predictions, sha256_file


CONFIG = {
    "botsim": {
        "folder": "output/v3_computable/botsim_group_tail_12",
        "lock": "selection_lock.json", "predictions": "test_predictions.json",
        "baseline": "output/hypbot_botsim_paired", "n": 437, "concepts": 12},
    "fox8": {
        "folder": "output/v3_fox8_targeted_neighbor_20_concepts",
        "lock": "selection_lock_ab_group_tree.json",
        "predictions": "ab_group_tree_test_predictions.json",
        "baseline": "output/hypbot_fox8_paired", "n": 343, "concepts": 20},
    "mgtab": {
        "folder": "output/v3_mgtab_targeted_20_concept_ab_group_tree",
        "lock": "selection_lock.json", "predictions": "test_predictions.json",
        "baseline": "output/hypbot_mgtab_paired", "n": 1020, "concepts": 20},
}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    datasets = []
    for dataset, config in CONFIG.items():
        folder = root / config["folder"]
        lock_path = folder / config["lock"]
        lock_hash = sha256_file(lock_path)
        if dataset == "fox8":
            cards = json.loads((folder / "evidence_cards_manifest.json")
                               .read_text(encoding="utf-8"))
            cert = json.loads((root / "research/certificates/fox8_targeted_20_concept_cards.json")
                              .read_text(encoding="utf-8"))
            if cert["manifest_sha256"] != sha256_file(folder / "evidence_cards_manifest.json"):
                raise ValueError("Fox8 compact evidence cards differ")
        elif dataset == "mgtab":
            cards = json.loads((folder / "evidence_cards_manifest.json")
                               .read_text(encoding="utf-8"))
            cert = json.loads((root / "research/certificates/mgtab_targeted_20_concept_cards.json")
                              .read_text(encoding="utf-8"))
            if cert["manifest_sha256"] != sha256_file(folder / "evidence_cards_manifest.json"):
                raise ValueError("MGTAB compact evidence cards differ")
        runs = []
        for seed in (0, 1, 2):
            run_dir = folder / f"seed{seed}"
            candidate_path = run_dir / config["predictions"]
            base_path = root / config["baseline"] / f"seed{seed}/test_predictions.json"
            candidate = load_predictions(candidate_path)
            baseline = load_predictions(base_path)
            manifest = candidate["manifest"]
            if (manifest["dataset"] != dataset or manifest["seed"] != seed
                    or manifest["selection_lock_sha256"] != lock_hash
                    or baseline["manifest"]["dataset"] != dataset
                    or baseline["manifest"]["seed"] != seed
                    or baseline["manifest"]["model_sha256"] !=
                    sha256_file(base_path.parent / "model.pt")):
                raise ValueError("Frozen candidate or HypBot baseline differs")
            if dataset == "botsim":
                report = json.loads((run_dir / "report.json")
                                    .read_text(encoding="utf-8"))
                if (report["manifest"] != manifest or
                        report["cards_sha256"] !=
                        sha256_file(run_dir / "concept_evidence_cards.jsonl.gz")):
                    raise ValueError("BotSim twelve-concept evidence differs")
            elif dataset == "fox8":
                for key, name in (("model_sha256", "ab_group_tree.joblib"),
                                  ("reference_sha256", "label_blind_reference.json.gz"),
                                  ("evidence_sha256", "ab_group_tree_test_evidence.npz")):
                    if manifest[key] != sha256_file(run_dir / name):
                        raise ValueError("Fox8 compact model or evidence changed")
                if cards["runs"][seed]["sha256"] != sha256_file(
                        run_dir / "ab_group_tree_test_cards.jsonl.gz"):
                    raise ValueError("Fox8 account cards changed")
            else:
                for key, name in (("model_sha256", "model.joblib"),
                                  ("reference_sha256", "reference.json.gz"),
                                  ("evidence_sha256", "test_evidence.npz")):
                    if manifest[key] != sha256_file(run_dir / name):
                        raise ValueError("MGTAB compact model or evidence changed")
                if cards["runs"][seed]["sha256"] != sha256_file(
                        run_dir / "test_evidence_cards.jsonl.gz"):
                    raise ValueError("MGTAB account cards changed")
            paired = compare(candidate, baseline, config["n"])
            paired.update({"seed": seed,
                           "split_sha256": manifest["split_sha256"],
                           "candidate_predictions_sha256": sha256_file(candidate_path),
                           "hypbot_predictions_sha256": sha256_file(base_path)})
            if dataset == "fox8":
                native_path = root / ("output/hypbot_fox8_native_graph_paired/"
                                      f"seed{seed}/test_predictions.json")
                native = load_predictions(native_path)
                if native["manifest"]["model_sha256"] != sha256_file(
                        native_path.parent / "model.pt"):
                    raise ValueError("Fox8 native-graph HypBot checkpoint differs")
                paired["same_native_graph_hypbot"] = compare(
                    candidate, native, config["n"])
            runs.append(paired)
        datasets.append({
            "dataset": dataset, "concept_count": config["concepts"],
            "candidate_model": config["folder"],
            "official_hypbot_model": config["baseline"],
            "candidate_mean_acc": float(np.mean([r["candidate_acc"] for r in runs])),
            "hypbot_mean_acc": float(np.mean([r["hypbot_acc"] for r in runs])),
            "candidate_above_official_hypbot_each_seed": all(
                r["candidate_strictly_above_hypbot"] for r in runs),
            "runs": runs})
        print(json.dumps({"dataset": dataset,
                          "candidate_acc": [r["candidate_acc"] for r in runs],
                          "hypbot_acc": [r["hypbot_acc"] for r in runs]}), flush=True)
    output = {
        "scope": "three_dataset_compact_twelve_or_twenty_concept_evidence_numeric_goal",
        "all_three_datasets_above_official_hypbot_each_seed": all(
            d["candidate_above_official_hypbot_each_seed"] for d in datasets),
        "same_native_graph_fox8_comparison_separate": True,
        "prior_test_population_exploration": True,
        "independent_behavior_semantic_truth_evaluated": False,
        "datasets": datasets,
    }
    path = root / "research/certificates/three_dataset_compact_evidence_numeric_goal.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
