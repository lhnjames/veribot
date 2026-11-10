"""Evaluate dev-locked Fox8 single-fact trees on paired fixed tests."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from gate0_shallow_tree import evaluate, sha256_file
from lock_fox8_single_fact_baseline import OUT, SCREEN, column, population


ROOT = Path(__file__).resolve().parents[2]
LOCK = OUT / "dev_lock.json"
NEUTRAL = ROOT / "output/v3_fox8_twenty_neutral_anchor/fixed_test_report.json"
REPORT = OUT / "fixed_test_report.json"


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    strong = json.loads(NEUTRAL.read_text(encoding="utf-8"))
    if (lock["scope"] != "Fox8_best_single_atomic_fact_selected_on_dev_before_test" or
            lock["dev_screen_sha256"] != sha256_file(SCREEN) or
            lock["test_predictions_or_metrics_used_for_selection"] is not False):
        raise ValueError("Single-fact baseline selection is not dev-locked")
    runs = []
    for seed in (0, 1, 2):
        row = lock["runs"][seed]
        runtime, y, split, report = population(seed)
        if (report["manifest"]["split_sha256"] != row["split_sha256"] or
                row["split_sha256"] != strong["runs"][seed]["split_sha256"]):
            raise ValueError("Paired Fox8 fixed test differs")
        model_path = Path(row["tree_path"])
        if sha256_file(model_path) != row["tree_sha256"]:
            raise ValueError("Frozen single-fact tree changed")
        tree = joblib.load(model_path)
        test = split["test"]
        ids = np.asarray(runtime.ids)[test].astype(str)
        x = column(runtime, row["fact"], float(row["undefined_encoding"]))
        probability = tree.predict_proba(x[test])[:, 1]
        pred = probability >= .5
        metrics = evaluate(y[test], probability, .5)
        strong_path = Path(strong["runs"][seed]["prediction_matrix_path"])
        if sha256_file(strong_path) != strong["runs"][seed][
                "prediction_matrix_sha256"]:
            raise ValueError("Current model fixed predictions changed")
        with np.load(strong_path, allow_pickle=False) as z:
            if not np.array_equal(ids, z["account_ids"].astype(str)):
                raise ValueError("Single-fact and current model test IDs differ")
            current_pred = z["predicted_bot"].astype(bool)
        output_path = OUT / f"seed{seed}/fixed_test_predictions.npz"
        np.savez_compressed(output_path, account_ids=ids,
                            probability=probability, predicted_bot=pred,
                            threshold=np.asarray(.5))
        correct = int(np.count_nonzero(pred == y[test]))
        current_correct = int(np.count_nonzero(current_pred == y[test]))
        assert current_correct == strong["runs"][seed]["correct"]
        runs.append({
            "seed": seed, "test_accounts": len(test),
            "selected_fact": row["fact"],
            "selected_max_depth": row["max_depth"],
            "single_fact_correct": correct,
            "current_model_correct": current_correct,
            "hypbot_correct": strong["runs"][seed][
                "hypbot_correct_on_same_population"],
            "current_only_correct": int(np.count_nonzero(
                (current_pred == y[test]) & (pred != y[test]))),
            "single_only_correct": int(np.count_nonzero(
                (current_pred != y[test]) & (pred == y[test]))),
            "metrics": metrics,
            "split_sha256": row["split_sha256"],
            "tree_sha256": row["tree_sha256"],
            "fixed_prediction_matrix_sha256": sha256_file(output_path),
        })
        print({"seed": seed, "single": correct,
               "current": current_correct,
               "hypbot": runs[-1]["hypbot_correct"]}, flush=True)
    REPORT.write_text(json.dumps({
        "scope": "Fox8_dev_locked_single_fact_paired_fixed_test_control",
        "dev_lock_sha256": sha256_file(LOCK),
        "current_model_fixed_report_sha256": sha256_file(NEUTRAL),
        "test_population_previously_explored_by_other_candidates": True,
        "no_test_metric_used_to_choose_fact_or_depth": True,
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
