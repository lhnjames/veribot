"""Lock the group-additive MGTAB A/B monotone evidence tree from dev only."""

from __future__ import annotations

import json
from pathlib import Path

from gate0_shallow_tree import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_ab_group_additive_tree"
    dev_path = folder / "dev_only.json"
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    if (dev["scope"] !=
            "mgtab_official_train_dev_only_ab_monotone_additive_source_group_trees"
            or dev["test_predictions_or_metrics_computed"] is not False
            or len(dev["runs"]) != 3 or len(dev["configs"]) != 2):
        raise ValueError("Unexpected A/B group-additive validation source")
    means = [sum(run["candidates"][j]["validation_selected"]["macro_f1"]
                 for run in dev["runs"]) / 3 for j in range(2)]
    chosen = max(range(2), key=lambda j: means[j])
    if chosen != 1 or dev["configs"][chosen] != [31, 400, 50, 10.0]:
        raise ValueError("Predefined A/B group-additive choice differs")
    if any(not run["candidates"][chosen]["all_trees_use_one_source_group"]
           for run in dev["runs"]):
        raise ValueError("Selected source-group constraint failed")
    lock = {
        "dataset": "mgtab",
        "scope": "exploratory_ab_group_additive_monotone_tree_fixed_test_after_dev_lock",
        "selection_rule": "largest mean official-dev Macro-F1 across two predeclared group-additive monotone tree capacities; per-seed threshold selected on official dev",
        "dev_result_sha256": sha256_file(dev_path),
        "selected_config_index": chosen,
        "selected_config": dev["configs"][chosen],
        "mean_dev_macro_f1_by_config": means,
        "source_features_sha256": dev["source_features_sha256"],
        "split_sha256_by_seed": {str(run["seed"]): run["split_sha256"]
                                  for run in dev["runs"]},
        "reference_version_by_seed": {str(run["seed"]): run["reference_version"]
                                      for run in dev["runs"]},
        "threshold_by_seed": {
            str(run["seed"]): run["candidates"][chosen]["selected_threshold"]
            for run in dev["runs"]},
        "source_group_names_by_seed": {
            str(run["seed"]): run["source_group_names"]
            for run in dev["runs"]},
        "a_direction_and_b_fit_labels_disjoint": True,
        "prior_results_on_same_test_population_already_observed": True,
        "test_metric_or_prediction_used_to_create_lock": False,
    }
    path = folder / "selection_lock.json"
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"config": lock["selected_config"],
                      "mean_dev_macro_f1": means[chosen]}), flush=True)


if __name__ == "__main__":
    main()
