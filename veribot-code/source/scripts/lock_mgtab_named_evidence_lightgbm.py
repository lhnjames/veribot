"""Freeze the validation-selected named MGTAB LightGBM before fixed test."""

from __future__ import annotations

import json
from pathlib import Path

from gate0_shallow_tree import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dev_path = root / "output/v3_mgtab_named_evidence_lightgbm/dev_only.json"
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    if (dev["scope"] != "mgtab_train_dev_only_named_source_evidence_lightgbm"
            or dev["test_rows_accessed"] is not False
            or dev["published_content_projection_used"] is not False
            or len(dev["runs"]) != 3 or len(dev["configs"]) != 4):
        raise ValueError("validation source or candidate grid differs")
    averages = [sum(run["alternatives"][index]["validation"]["macro_f1"]
                    for run in dev["runs"]) / 3 for index in range(4)]
    selected = max(range(4), key=lambda index: averages[index])
    if selected != 3 or dev["configs"][selected] != [31, 400, 50, 10.0]:
        raise ValueError("predefined mean-dev-Macro-F1 winner differs")
    lock = {
        "dataset": "mgtab",
        "scope": "exploratory_named_source_lightgbm_test_after_dev_lock",
        "selection_rule": "largest mean official-dev Macro-F1 among four predefined evidence-only LightGBM capacities",
        "dev_result_sha256": sha256_file(dev_path),
        "selected_config_index": selected,
        "selected_config": dev["configs"][selected],
        "mean_dev_macro_f1_by_config": averages,
        "source_features_sha256": dev["source_features_sha256"],
        "structural_facts_sha256": dev["structural_facts_sha256"],
        "neighbor_facts_sha256": dev["neighbor_facts_sha256"],
        "split_sha256_by_seed": {str(run["seed"]): run["split_sha256"]
                                 for run in dev["runs"]},
        "reference_version_by_seed": {str(run["seed"]): run["reference_version"]
                                      for run in dev["runs"]},
        "threshold_by_seed": {str(run["seed"]):
                              run["alternatives"][selected]["threshold"]
                              for run in dev["runs"]},
        "prior_test_results_on_same_account_population_already_observed": True,
        "test_metric_or_prediction_used_to_create_lock": False,
    }
    destination = root / "output/v3_mgtab_named_evidence_lightgbm/selection_lock.json"
    destination.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(selected, dev["configs"][selected], averages[selected], flush=True)


if __name__ == "__main__":
    main()
