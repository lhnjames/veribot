"""Freeze the source-group additive Fox8 twenty-concept candidate using dev only."""

from __future__ import annotations

import json
from pathlib import Path

from gate0_shallow_tree import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_fox8_targeted_neighbor_20_concepts"
    dev_path = folder / "dev_only.json"
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    if (dev["scope"] != "fox8_official_train_dev_only_targeted_native_neighbor_twenty_concepts"
            or dev["test_predictions_or_metrics_computed"] is not False
            or len(dev["runs"]) != 3):
        raise ValueError("Unexpected validation source")
    ranked = []
    for removed in ("presentation_flags", "presentation_text"):
        for config_index in (0, 1):
            rows = [next(c for c in run["candidates"] if c["removed_group"] == removed)
                    ["ab_group_trees"][config_index] for run in dev["runs"]]
            ranked.append({"removed_group": removed, "config_index": config_index,
                           "mean_dev_macro_f1": sum(r["validation_selected"]["macro_f1"]
                                                    for r in rows) / 3,
                           "sum_dev_errors": sum(r["validation_selected"]["errors"]
                                                 for r in rows)})
    ranked.sort(key=lambda r: (-r["mean_dev_macro_f1"], r["sum_dev_errors"],
                               r["config_index"],
                               ("presentation_flags", "presentation_text").index(
                                   r["removed_group"])))
    chosen = ranked[0]
    if chosen != {"removed_group": "presentation_flags", "config_index": 0,
                  "mean_dev_macro_f1": chosen["mean_dev_macro_f1"],
                  "sum_dev_errors": 2}:
        raise ValueError("Dev selection changed; inspect candidate ranking")
    audits = (root / "research/certificates/fox8_neighbor_volume_full_raw_source.json",
              root / "research/certificates/fox8_native_neighbor_profile_raw_sample_audit.json")
    for path in audits:
        if not path.is_file():
            raise FileNotFoundError(path)
    lock = {
        "dataset": "Fox8-23",
        "scope": "exploratory_fixed_test_of_dev_locked_twenty_concept_ab_group_tree",
        "selection_rule": "largest mean official-dev Macro-F1 across the two predeclared replacement groups and two capacities; fewer errors, smaller capacity, then retain presentation text on ties",
        "selected": chosen,
        "ranking": ranked,
        "config": next(c for c in dev["runs"][0]["candidates"]
                       if c["removed_group"] == chosen["removed_group"])
                  ["ab_group_trees"][chosen["config_index"]]["config"],
        "threshold_by_seed": {
            str(run["seed"]): next(c for c in run["candidates"]
                                   if c["removed_group"] == chosen["removed_group"])
                              ["ab_group_trees"][chosen["config_index"]]["selected_threshold"]
            for run in dev["runs"]},
        "split_sha256_by_seed": {str(run["seed"]): run["split_sha256"]
                                 for run in dev["runs"]},
        "reference_version_by_seed": {str(run["seed"]): run["reference_version"]
                                      for run in dev["runs"]},
        "dev_result_sha256": sha256_file(dev_path),
        "neighbor_fact_sha256": dev["neighbor_fact_source_sha256"],
        "raw_source_audit_sha256": sha256_file(audits[0]),
        "neighbor_sample_audit_sha256": sha256_file(audits[1]),
        "prior_results_on_same_test_population_already_observed": True,
        "test_metric_or_prediction_used_to_create_lock": False,
    }
    path = folder / "selection_lock_ab_group_tree.json"
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": chosen, "threshold_by_seed": lock["threshold_by_seed"]}))


if __name__ == "__main__":
    main()
