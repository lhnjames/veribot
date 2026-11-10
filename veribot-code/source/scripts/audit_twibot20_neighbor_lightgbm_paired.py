"""Pair the frozen source-evidence candidate with the selected TwiBot test model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from train_v3_twibot20 import sha256


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    candidate_root = root / "output/v3_twibot20_neighbor_unigram/selected"
    candidate_path = candidate_root / "test_predictions.npz"
    candidate_report = json.loads((candidate_root / "report.json")
                                  .read_text(encoding="utf-8"))
    previous_path = root / "output/v3_twibot20_sparse_text_evidence/selected/test_predictions.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if (candidate_report["test_predictions_sha256"] != sha256(candidate_path)
            or previous["manifest"]["source_archive_sha256"] !=
            candidate_report["source_archive_sha256"]):
        raise ValueError("the two fixed tests have different source bindings")
    with np.load(candidate_path, allow_pickle=False) as candidate:
        ids = candidate["ids"].astype(str)
        labels = candidate["labels"].astype(int)
        probability = candidate["probability"].astype(float)
    old_rows = previous["predictions"]
    if (len(old_rows) != len(ids)
            or any(row["account_id"] != account_id or row["label"] != label
                   for row, account_id, label in zip(old_rows, ids, labels))):
        raise ValueError("fixed test account IDs, labels, or row order differ")
    old_prediction = np.asarray([
        row["probability"] >= previous["manifest"]["threshold"]
        for row in old_rows])
    new_prediction = probability >= candidate_report["threshold"]
    old_correct = old_prediction == labels
    new_correct = new_prediction == labels
    both_correct = int(np.count_nonzero(old_correct & new_correct))
    new_only = int(np.count_nonzero(~old_correct & new_correct))
    old_only = int(np.count_nonzero(old_correct & ~new_correct))
    both_wrong = int(np.count_nonzero(~old_correct & ~new_correct))
    if (both_correct + new_only + old_only + both_wrong != len(ids)
            or int(np.count_nonzero(new_correct)) !=
            candidate_report["test_neighbor_group_mask"]["full_correct"]):
        raise ValueError("paired count does not reconstruct candidate test")
    difference = new_correct.astype(int) - old_correct.astype(int)
    random = np.random.default_rng(20261103)
    bootstrap = np.mean(difference[random.integers(0, len(ids),
                                                  size=(10000, len(ids)))], axis=1)
    interval = np.quantile(bootstrap, (0.025, 0.975))
    result = {
        "dataset": "twibot20", "population": "same_official_test_1183_accounts",
        "old_test_predictions_sha256": sha256(previous_path),
        "candidate_test_predictions_sha256": sha256(candidate_path),
        "both_correct": both_correct, "candidate_only_correct": new_only,
        "old_only_correct": old_only, "both_wrong": both_wrong,
        "old_correct": int(np.count_nonzero(old_correct)),
        "candidate_correct": int(np.count_nonzero(new_correct)),
        "paired_accuracy_difference": float(np.mean(difference)),
        "paired_account_bootstrap_95_percent_interval": interval.tolist(),
        "discordant_exact_two_sided_p": float(binomtest(
            new_only, new_only + old_only, p=0.5).pvalue),
        "same_test_population_previously_used_for_other_candidates": True,
    }
    destination = candidate_root / "paired_vs_selected.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
