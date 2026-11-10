"""Paired official-test comparison of locked sparse evidence and prior runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    cm = candidate["manifest"]
    current = candidate["predictions"]
    if cm["dataset"] != "twibot20" or cm["official_split"] is not True:
        raise ValueError("candidate is not official TwiBot-20 test")
    labels = np.asarray([row["label"] for row in current], dtype=int)
    candidate_correct = (
        np.asarray([row["probability"] for row in current]) >= cm["threshold"]) == labels
    deltas = []
    runs = []
    for seed in (0, 1, 2):
        baseline = json.loads((args.baseline_dir / f"seed{seed}" /
                               "test_predictions.json").read_text(encoding="utf-8"))
        bm = baseline["manifest"]
        rows = baseline["predictions"]
        if (bm["dataset"] != cm["dataset"]
                or bm["official_split"] is not True
                or bm.get("facts_sha256", bm.get("base_facts_sha256")) !=
                cm["base_facts_sha256"]
                or bm["source_archive_sha256"] != cm["source_archive_sha256"]
                or len(rows) != len(current)
                or any(a["account_id"] != b["account_id"] or
                       a["label"] != b["label"]
                       for a, b in zip(rows, current))):
            raise ValueError("baseline and candidate test population differ")
        old_correct = (
            np.asarray([row["probability"] for row in rows]) >= bm["threshold"]) == labels
        delta = candidate_correct.astype(int) - old_correct.astype(int)
        deltas.append(delta)
        runs.append({"seed": seed, "baseline_acc": float(old_correct.mean()),
                     "candidate_acc": float(candidate_correct.mean()),
                     "paired_acc_difference": float(delta.mean()),
                     "candidate_correct_baseline_wrong": int(np.sum(
                         candidate_correct & ~old_correct)),
                     "baseline_correct_candidate_wrong": int(np.sum(
                         old_correct & ~candidate_correct))})
    matrix = np.stack(deltas)
    rng = np.random.default_rng(20261102)
    draws = np.asarray([matrix[:, rng.integers(0, matrix.shape[1],
                                               matrix.shape[1])].mean()
                        for _ in range(10_000)])
    report = {
        "dataset": "twibot20", "test_accounts": len(current),
        "comparison_scope": "same official test accounts, base facts and source archive; candidate has source-backed unigram evidence",
        "baseline_dir": str(args.baseline_dir),
        "candidate": str(args.candidate),
        "runs": runs,
        "mean_paired_acc_difference": float(matrix.mean()),
        "account_cluster_bootstrap_95pct_ci": [float(x) for x in
                                               np.quantile(draws, [0.025, 0.975])],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
