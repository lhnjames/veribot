"""Freeze the MGTAB positive-Human candidate from train/dev evidence only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_mgtab_ab_group_positive_human"


def main() -> None:
    dev_path = FOLDER / "dev_only.json"
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    base_lock_path = (ROOT /
        "output/v3_mgtab_ab_group_additive_tree/selection_lock.json")
    base = json.loads(base_lock_path.read_text(encoding="utf-8"))
    if (dev["scope"] !=
            "train_dev_only_high_performance_mgtab_positive_human_evidence"
            or dev["test_metrics_or_predictions_read"] is not False
            or dev["base_selection_lock_sha256"] != sha256_file(base_lock_path)
            or dev["anchor_ratio"] != 0.05
            or len(dev["runs"]) != 3):
        raise ValueError("positive Human validation screen differs")
    candidate = [r["validation"]["macro_f1"] for r in dev["runs"]]
    baseline = [r["baseline_validation"]["macro_f1"]
                for r in dev["runs"]]
    mean_delta = sum(candidate)/3 - sum(baseline)/3
    gate = (
        mean_delta >= -0.003 and
        all(r["zero_sufficient_90_percent_margin_accounts"] <=
            0.05*r["validation"]["n"] for r in dev["runs"]) and
        all(r["positive_human_only_mask_net_correct_loss"] > 0
            for r in dev["runs"]))
    if not gate:
        raise ValueError("positive Human dev gate was not met")
    payload = {
        "dataset": "mgtab",
        "scope": "dev_locked_ab_group_additive_positive_human_evidence",
        "selection_rule": (
            "same A/B group-additive tree with 9 source-present small-deviation "
            "Human channels and 5 percent balanced zero-input anchor; require "
            "mean dev Macro-F1 within 0.3 percentage points of baseline, "
            "at most 5 percent zero-sufficient dev accounts each seed, and "
            "positive net correctness from Human channels each seed"),
        "selected": True,
        "test_metric_or_prediction_used_to_create_lock": False,
        "prior_results_on_same_test_population_already_observed": True,
        "dev_result_sha256": sha256_file(dev_path),
        "base_selection_lock_sha256": sha256_file(base_lock_path),
        "source_features_sha256": dev["source_features_sha256"],
        "anchor_ratio": dev["anchor_ratio"],
        "model_config": base["selected_config"],
        "mean_dev_macro_f1_delta": mean_delta,
        "threshold_by_seed": {str(r["seed"]): r["threshold"]
                              for r in dev["runs"]},
        "split_sha256_by_seed": {str(r["seed"]): r["split_sha256"]
                                 for r in dev["runs"]},
        "reference_version_by_seed": {str(r["seed"]): r["reference_version"]
                                      for r in dev["runs"]},
    }
    path = FOLDER / "selection_lock.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(path, mean_delta)


if __name__ == "__main__":
    main()
