"""Freeze the MGTAB twenty-concept candidate on official validation only."""

from __future__ import annotations

import json
from pathlib import Path

from gate0_shallow_tree import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_targeted_20_concept_ab_group_tree"
    path = folder / "dev_only.json"
    dev = json.loads(path.read_text(encoding="utf-8"))
    if (dev["scope"] != "mgtab_official_train_dev_only_targeted_twenty_concept_ab_group_tree"
            or dev["test_predictions_or_metrics_computed"] is not False
            or len(dev["runs"]) != 3):
        raise ValueError("Unexpected validation-only source")
    ranked = []
    for scheme in (candidate["scheme"] for candidate in dev["runs"][0]["candidates"]):
        for index in (0, 1):
            metrics = [next(c for c in run["candidates"] if c["scheme"] == scheme)
                       ["variants"][index]["validation"] for run in dev["runs"]]
            ranked.append({"scheme": scheme, "config_index": index,
                           "mean_dev_macro_f1": sum(row["macro_f1"] for row in metrics) / 3,
                           "sum_dev_errors": sum(row["errors"] for row in metrics)})
    ranked.sort(key=lambda row: (-row["mean_dev_macro_f1"], row["sum_dev_errors"],
                                 row["config_index"], row["scheme"]))
    selected = ranked[0]
    if (selected["scheme"] != "four_profile_two_relation_four_selected_content"
            or selected["config_index"] != 1):
        raise ValueError("Validation winner changed; inspect before fixed test")
    lock = {
        "dataset": "MGTAB", "selected": selected,
        "selection_rule": "highest mean official-dev Macro-F1 among four prespecified 20-concept partitions and two capacities; fewer total errors then smaller capacity on tie",
        "ranking": ranked,
        "config": dev["configs"][selected["config_index"]],
        "threshold_by_seed": {
            str(run["seed"]): next(c for c in run["candidates"]
                                   if c["scheme"] == selected["scheme"])
                              ["variants"][selected["config_index"]]["threshold"]
            for run in dev["runs"]},
        "selected_content_by_seed": {
            str(run["seed"]): next(c for c in run["candidates"]
                                   if c["scheme"] == selected["scheme"])
                              ["selected_content_on_A"]
            for run in dev["runs"]},
        "split_sha256_by_seed": {str(run["seed"]): run["split_sha256"]
                                 for run in dev["runs"]},
        "reference_version_by_seed": {str(run["seed"]): run["reference_version"]
                                      for run in dev["runs"]},
        "dev_result_sha256": sha256_file(path),
        "source_features_sha256": dev["source_features_sha256"],
        "prior_test_results_on_same_account_population_already_observed": True,
        "test_metric_or_prediction_used_to_create_lock": False,
        "content_provenance_level": "published_average_tweet_tensor",
    }
    output = folder / "selection_lock.json"
    output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
