"""Replay every fixed-test decision from the label-free best-model batch entry."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate0_shallow_tree import sha256_file
from predict_best_evidence_batch import MODELS, predict


ROOT = Path(__file__).resolve().parents[2]
SAVED = {
    "botsim": "output/v3_botsim_observed_means_12",
    "fox8": "output/v3_fox8_neighbor_atomic_evidence_capacity",
    "mgtab": "output/v3_mgtab_ab_group_additive_tree",
}
OUT = ROOT / "research/certificates/best_evidence_batch_observed_means_full_test.json"


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    original_text = Path.read_text
    original_open = Path.open
    original_csv = pd.read_csv
    original_torch = torch.load
    attempts: list[str] = []

    def forbidden(path: object) -> bool:
        name = Path(path).name.lower()
        return ("test_predictions" in name or "test_evidence" in name or
                name in {"report.json", "fixed_test_report.json", "labels.csv",
                         "label.csv", "labels_bot.pt", "labels.pt"})

    def checked_text(path: Path, *args, **kwargs):
        if forbidden(path):
            attempts.append(str(path))
            raise AssertionError("batch inference read a result or label")
        return original_text(path, *args, **kwargs)

    def checked_open(path: Path, *args, **kwargs):
        if forbidden(path):
            attempts.append(str(path))
            raise AssertionError("batch inference opened a result or label")
        return original_open(path, *args, **kwargs)

    def checked_csv(path, *args, **kwargs):
        if forbidden(path):
            attempts.append(str(path))
            raise AssertionError("batch inference read a label table")
        return original_csv(path, *args, **kwargs)

    def checked_torch(path, *args, **kwargs):
        if forbidden(path):
            attempts.append(str(path))
            raise AssertionError("batch inference loaded labels")
        return original_torch(path, *args, **kwargs)

    runs = []
    for dataset in MODELS:
        for seed in (0, 1, 2):
            path = ROOT / SAVED[dataset] / f"seed{seed}/test_predictions.json"
            recorded = json.loads(path.read_text(encoding="utf-8"))["predictions"]
            ids = [str(row["account_id"]) for row in recorded]
            if len(set(ids)) != len(ids):
                raise ValueError("fixed test account IDs repeat")
            before = len(attempts)
            with patch.object(Path, "read_text", checked_text), \
                    patch.object(Path, "open", checked_open), \
                    patch.object(pd, "read_csv", checked_csv), \
                    patch.object(torch, "load", checked_torch):
                observed = predict(ROOT, dataset, seed, ids)
            if len(attempts) != before:
                raise ValueError("batch inference attempted a forbidden read")
            actual = observed["predictions"]
            if ([item["account_id"] for item in actual] != ids or
                    observed["classification_labels_accessed"] is not False or
                    observed["caller_supplied_fact_matrix_accepted"] is not False):
                raise ValueError("batch inference population or contract differs")
            expected_prob = np.asarray([row["probability"] for row in recorded],
                                       dtype=float)
            actual_prob = np.asarray([row["probability"] for row in actual],
                                     dtype=float)
            maximum = float(np.max(np.abs(expected_prob - actual_prob)))
            expected_class = (expected_prob >=
                              float(observed["threshold"])).astype(int)
            actual_class = np.asarray([row["prediction"] for row in actual],
                                      dtype=int)
            mismatches = int(np.count_nonzero(expected_class != actual_class))
            if maximum > 1e-12 or mismatches:
                raise ValueError("source-bound batch decision differs from frozen test")
            runs.append({
                "dataset": dataset,
                "seed": seed,
                "test_accounts": len(ids),
                "evidence_terms_per_account":
                    observed["evidence_terms_per_account"],
                "reference_version": observed["reference_version"],
                "test_predictions_sha256": sha256_file(path),
                "max_probability_difference": maximum,
                "class_mismatches": mismatches,
                "forbidden_read_attempts": len(attempts) - before,
            })
            print({key: runs[-1][key] for key in (
                "dataset", "seed", "test_accounts",
                "max_probability_difference", "class_mismatches")},
                flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "all_nine_best_evidence_models_source_bound_batch_replayed",
        "scope": "full_fixed_test_population_for_each_dataset_seed",
        "batch_entry": str(ROOT / "source/scripts/predict_best_evidence_batch.py"),
        "batch_entry_sha256": sha256_file(
            ROOT / "source/scripts/predict_best_evidence_batch.py"),
        "guarded_read_apis": ["Path.read_text", "Path.open",
                              "pandas.read_csv", "torch.load"],
        "guarded_label_or_test_result_read_attempts": len(attempts),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
