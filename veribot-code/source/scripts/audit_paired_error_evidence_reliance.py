"""Join fixed HypBot/VeriBot outcomes to exact per-account evidence dependence."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np

from gate0_shallow_tree import sha256_file


SELECTED = {
    "botsim": "output/v3_computable/botsim_group_tail_12",
    "fox8": "output/v3_fox8_native_spline_l2/group_tail_20_selected",
    "mgtab": "output/v3_mgtab_ab_spline_content/direct_selected_20",
}
DEPENDENCE = "research/three_dataset_compact_decision_dependence_2026-11-03.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _graph_concept(dataset: str, name: str) -> bool:
    if dataset == "botsim":
        return name.startswith("native_reply_graph:")
    if dataset == "fox8":
        return name.startswith(("native_reply:", "native_retweet:"))
    return (name.startswith("published_") and
            "tweet_projection" not in name and "profile_coordinate" not in name)


def _summary(dataset: str, rows: list[dict]) -> dict:
    pivotal = Counter(name for row in rows
                      for name in row["individual_pivotal_concepts"])
    deletion_sizes = [len(row["smallest_support_deletion_to_flip"])
                      for row in rows if row["smallest_support_deletion_to_flip"] is not None]
    sufficient_sizes = [len(row["smallest_sufficient_same_direction_subset"])
                        for row in rows]
    return {
        "accounts": len(rows),
        "evidence_changes_intercept_class": sum(
            row["evidence_changes_intercept_class"] for row in rows),
        "at_least_one_individually_pivotal_concept": sum(
            bool(row["individual_pivotal_concepts"]) for row in rows),
        "at_least_one_individually_pivotal_graph_concept": sum(
            any(_graph_concept(dataset, name)
                for name in row["individual_pivotal_concepts"]) for row in rows),
        "graph_concept_in_minimum_sufficient_subset": sum(
            any(_graph_concept(dataset, name)
                for name in row["smallest_sufficient_same_direction_subset"])
            for row in rows),
        "support_deletion_can_flip": len(deletion_sizes),
        "median_minimum_support_deletion_size": float(np.median(deletion_sizes))
        if deletion_sizes else None,
        "median_minimum_same_direction_sufficient_size": float(np.median(
            sufficient_sizes)) if sufficient_sizes else None,
        "pivotal_concept_counts": dict(pivotal),
    }


def audit_run(root: Path, dataset: str, seed: int, dependence: dict) -> dict:
    selected_path = root / SELECTED[dataset] / f"seed{seed}" / "test_predictions.json"
    hypbot_path = root / f"output/hypbot_{dataset}_paired/seed{seed}/test_predictions.json"
    candidate = load(selected_path)
    baseline = load(hypbot_path)
    a, b = candidate["manifest"], baseline["manifest"]
    if (a["dataset"] != dataset or b["dataset"] != dataset
            or a["seed"] != seed or b["seed"] != seed
            or a["split_sha256"] != b["split_sha256"]
            or a["split_sha256"] != dependence["split_sha256"]):
        raise ValueError("paired outcome split does not match dependence audit")
    if sha256_file(root / SELECTED[dataset] / f"seed{seed}" /
                   ("evidence_cards.jsonl.gz" if dataset == "mgtab"
                    else "concept_evidence_cards.jsonl.gz")) != dependence["cards_sha256"]:
        raise ValueError("dependence audit refers to different source-to-decision cards")
    candidate_rows = {str(row["account_id"]): row for row in candidate["predictions"]}
    baseline_rows = {str(row["account_id"]): row for row in baseline["predictions"]}
    dependence_rows = {str(row["account_id"]): row for row in dependence["accounts"]}
    if (len(candidate_rows) != len(candidate["predictions"])
            or len(baseline_rows) != len(baseline["predictions"])
            or len(dependence_rows) != len(dependence["accounts"])
            or not (set(candidate_rows) == set(baseline_rows) == set(dependence_rows))):
        raise ValueError("paired populations differ or contain repeated accounts")
    groups: dict[str, list[dict]] = {
        "veribot_only_correct": [], "hypbot_only_correct": [],
        "both_correct": [], "both_wrong": [],
    }
    for account_id, candidate_row in candidate_rows.items():
        baseline_row = baseline_rows[account_id]
        decision_row = dependence_rows[account_id]
        label = int(candidate_row["label"])
        if int(baseline_row["label"]) != label:
            raise ValueError("paired labels differ")
        candidate_pred = int(float(candidate_row["probability"]) >= a["threshold"])
        baseline_pred = int(float(baseline_row["probability"]) >= b["threshold"])
        if candidate_pred != decision_row["prediction"]:
            raise ValueError("dependence audit changes fixed VeriBot prediction")
        candidate_right = candidate_pred == label
        baseline_right = baseline_pred == label
        category = ("both_correct" if candidate_right and baseline_right else
                    "veribot_only_correct" if candidate_right else
                    "hypbot_only_correct" if baseline_right else "both_wrong")
        groups[category].append(decision_row)
    return {
        "seed": seed, "split_sha256": a["split_sha256"],
        "veribot_predictions_sha256": sha256_file(selected_path),
        "hypbot_predictions_sha256": sha256_file(hypbot_path),
        "dependence_cards_sha256": dependence["cards_sha256"],
        "test_accounts": len(candidate_rows),
        "outcomes": {name: _summary(dataset, rows) for name, rows in groups.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dependence_path = args.project_root / DEPENDENCE
    verified = load(dependence_path)
    datasets = []
    for dataset, directory in SELECTED.items():
        dep_dataset = next(row for row in verified["datasets"]
                           if row["dataset"] == dataset)
        dep_runs = {row["seed"]: row for row in dep_dataset["runs"]}
        runs = [audit_run(args.project_root, dataset, seed, dep_runs[seed])
                for seed in (0, 1, 2)]
        datasets.append({"dataset": dataset, "runs": runs})
    result = {
        "scope": "fixed_same_account_paired_outcomes_vs_exact_evidence_dependence",
        "dependence_audit_sha256": sha256_file(dependence_path),
        "meaning": "fixed-model evidence-input dependence; no causal source intervention",
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for dataset in datasets:
        print(dataset["dataset"], [
            {"seed": row["seed"],
             "veribot_only_correct": row["outcomes"]["veribot_only_correct"]["accounts"],
             "hypbot_only_correct": row["outcomes"]["hypbot_only_correct"]["accounts"],
             "veribot_wins_with_evidence_class_change":
                 row["outcomes"]["veribot_only_correct"]["evidence_changes_intercept_class"],
             "veribot_wins_with_pivotal_concept":
                 row["outcomes"]["veribot_only_correct"]["at_least_one_individually_pivotal_concept"]}
            for row in dataset["runs"]])


if __name__ == "__main__":
    main()
