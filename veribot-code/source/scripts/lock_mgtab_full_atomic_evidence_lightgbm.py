"""Lock MGTAB full atomic-evidence LightGBM by mean official-dev Macro-F1."""

from __future__ import annotations

import json
from pathlib import Path

from gate0_shallow_tree import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dev_path = root / "output/v3_mgtab_full_atomic_evidence_lightgbm/dev_only.json"
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    if (dev["scope"] != "mgtab_official_train_dev_only_all_150_source_evidence_lightgbm"
            or dev["test_predictions_or_metrics_computed"] is not False
            or len(dev["configs"]) != 4 or len(dev["runs"]) != 3):
        raise ValueError("Unexpected MGTAB validation source")
    mean_f1 = [sum(run["alternatives"][j]["validation_at_fixed_half"]["macro_f1"]
                   for run in dev["runs"]) / 3 for j in range(4)]
    selected = max(range(4), key=lambda j: mean_f1[j])
    if selected != 3 or dev["configs"][selected] != [31, 400, 50, 10.0]:
        raise ValueError("Predefined validation selection changed")
    lock = {
        "dataset": "mgtab",
        "scope": "exploratory_full_atomic_evidence_fixed_test_after_dev_lock",
        "selection_rule": "largest mean official-dev Macro-F1 at prespecified 0.5 threshold among four predefined LightGBM capacities",
        "selected_config_index": selected,
        "selected_config": dev["configs"][selected],
        "mean_dev_macro_f1_by_config": mean_f1,
        "threshold": 0.5,
        "dev_result_sha256": sha256_file(dev_path),
        "source_features_sha256": dev["source_features_sha256"],
        "structural_facts_sha256": dev["structural_facts_sha256"],
        "neighbor_facts_sha256": dev["neighbor_facts_sha256"],
        "source_audit_sha256": dev["source_audit_sha256"],
        "split_sha256_by_seed": {str(run["seed"]): run["split_sha256"]
                                  for run in dev["runs"]},
        "reference_version_by_seed": {str(run["seed"]): run["reference_version"]
                                      for run in dev["runs"]},
        "projection_fingerprint_by_seed": {
            str(run["seed"]): run["projection_fingerprint"]
            for run in dev["runs"]},
        "prior_results_on_same_test_population_already_observed": True,
        "test_metric_or_prediction_used_to_create_lock": False,
    }
    path = root / "output/v3_mgtab_full_atomic_evidence_lightgbm/selection_lock.json"
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_config_index": selected,
                      "selected_config": lock["selected_config"],
                      "mean_dev_macro_f1": mean_f1[selected]}), flush=True)


if __name__ == "__main__":
    main()
