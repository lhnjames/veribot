"""Pair the current neutral evidence models with frozen native-edge HypBot.

This is an outcome comparison on the same accounts, not an input-matched
algorithm ablation. The native-edge HypBot is a graph variant of its official
configuration and was trained before this comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/certificates/neutral_evidence_native_hypbot_paired.json"
CURRENT = {
    "fox8": ROOT / "output/v3_fox8_twenty_neutral_anchor/fixed_test_report.json",
    "botsim": ROOT / "output/v3_botsim_neutral_threshold/fixed_test_report.json",
}


def compare(dataset: str) -> dict:
    current_path = CURRENT[dataset]
    current = json.loads(current_path.read_text(encoding="utf-8"))
    native_folder = ROOT / f"output/hypbot_{dataset}_native_graph_paired"
    native_summary_path = native_folder / "summary.json"
    native_summary = json.loads(native_summary_path.read_text(encoding="utf-8"))
    runs = []
    for seed in (0, 1, 2):
        row = current["runs"][seed]
        if row["seed"] != seed or native_summary["runs"][seed]["seed"] != seed:
            raise ValueError("Seed order changed")
        matrix_path = Path(row["prediction_matrix_path"])
        native_path = native_folder / f"seed{seed}/test_predictions.json"
        if (sha256_file(matrix_path) != row["prediction_matrix_sha256"] or
                sha256_file(native_path) != native_summary["runs"][seed]["predictions_sha256"]):
            raise ValueError("Saved predictions changed")
        native = json.loads(native_path.read_text(encoding="utf-8"))
        manifest = native["manifest"]
        split_key = "split_sha256" if dataset == "fox8" else "source_split_sha256"
        if (manifest["dataset"] != dataset or manifest["seed"] != seed or
                manifest["split_sha256"] != row[split_key] or
                manifest["official_paper_graph_configuration"] is not False):
            raise ValueError("Dataset, split, or graph variant differs")
        with np.load(matrix_path, allow_pickle=False) as matrix:
            ids = matrix["account_ids"].astype(str)
            current_pred = matrix["predicted_bot"].astype(bool)
            current_prob = matrix["probability"].astype(float)
            threshold = float(matrix["threshold"])
        if not np.array_equal(current_pred, current_prob >= threshold):
            raise ValueError("Current predictions disagree with saved scores")
        native_by_id = {str(item["account_id"]): item for item in native["predictions"]}
        if len(native_by_id) != len(ids) or set(native_by_id) != set(ids):
            raise ValueError("Test account IDs differ or duplicate")
        y = np.asarray([int(native_by_id[uid]["label"]) for uid in ids])
        native_pred = np.asarray([float(native_by_id[uid]["probability"]) >=
                                  float(manifest["threshold"]) for uid in ids])
        current_correct = current_pred == y
        native_correct = native_pred == y
        if (int(current_correct.sum()) != row["correct"] or
                int(native_correct.sum()) != round(native_summary["runs"][seed]["test_acc"] * len(ids))):
            raise ValueError("Saved ACC disagrees with recomputed labels")
        delta = current_correct.astype(np.int8) - native_correct.astype(np.int8)
        rng = np.random.default_rng(20261104 + seed)
        bootstrap = np.empty(10_000)
        for i in range(len(bootstrap)):
            bootstrap[i] = delta[rng.integers(0, len(delta), len(delta))].mean()
        runs.append({
            "seed": seed,
            "test_accounts": len(ids),
            "neutral_evidence_correct": int(current_correct.sum()),
            "native_hypbot_correct": int(native_correct.sum()),
            "neutral_only_correct": int(np.sum(current_correct & ~native_correct)),
            "native_only_correct": int(np.sum(native_correct & ~current_correct)),
            "paired_acc_difference": float(delta.mean()),
            "paired_account_bootstrap_95pct_ci": [float(v) for v in
                                                   np.quantile(bootstrap, [0.025, 0.975])],
            "neutral_prediction_sha256": sha256_file(matrix_path),
            "native_hypbot_prediction_sha256": sha256_file(native_path),
            "split_sha256": manifest["split_sha256"],
        })
    return {"dataset": dataset,
            "official_paper_configuration": False,
            "comparison_scope": "same test accounts and native relation source; node features and classifiers differ",
            "neutral_report_sha256": sha256_file(current_path),
            "native_summary_sha256": sha256_file(native_summary_path),
            "all_seeds_strictly_above_native_variant": all(
                r["neutral_evidence_correct"] > r["native_hypbot_correct"] for r in runs),
            "runs": runs}


def main() -> None:
    report = {"scope": "frozen_neutral_evidence_vs_native_edge_HypBot_paired_outcomes",
              "test_population_previously_explored_by_other_candidates": True,
              "datasets": [compare(dataset) for dataset in ("fox8", "botsim")]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for dataset in report["datasets"]:
        print(dataset["dataset"], [(r["neutral_evidence_correct"],
                                    r["native_hypbot_correct"]) for r in dataset["runs"]])
    print(OUT)


if __name__ == "__main__":
    main()
