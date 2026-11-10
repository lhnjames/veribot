"""Lock the official-dev selected 20-fact MGTAB evidence model before test."""

from __future__ import annotations

import json
from pathlib import Path

from gate0_shallow_tree import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "output/v3_mgtab_named_top20_lightgbm/dev_only.json"
    dev = json.loads(path.read_text(encoding="utf-8"))
    if (dev["scope"] !=
            "official_train_dev_only_top20_named_release_source_evidence_lightgbm"
            or dev["test_labels_accessed_for_selection_or_fit"] is not False
            or dev["published_content_projection_used"] is not False
            or len(dev["runs"]) != 3 or len(dev["configs"]) != 4):
        raise ValueError("compact validation-only candidate record differs")
    mean_f1 = [sum(run["alternatives"][index]["validation"]["macro_f1"]
                   for run in dev["runs"]) / 3 for index in range(4)]
    selected = max(range(4), key=lambda index: mean_f1[index])
    if selected != 3 or dev["configs"][selected] != [31, 400, 50, 10.0]:
        raise ValueError("mean-dev-Macro-F1 compact winner differs")
    lock = {
        "dataset": "mgtab",
        "scope": "exploratory_compact_named_release_source_evidence_test_after_dev_lock",
        "selection_rule": "largest mean official-dev Macro-F1 across four predefined capacities; each seed's top facts selected only from its own training model",
        "dev_result_sha256": sha256_file(path),
        "selected_config_index": selected,
        "selected_config": dev["configs"][selected],
        "mean_dev_macro_f1_by_config": mean_f1,
        "source_features_sha256": dev["source_features_sha256"],
        "structural_facts_sha256": dev["structural_facts_sha256"],
        "neighbor_facts_sha256": dev["neighbor_facts_sha256"],
        "split_sha256_by_seed": {str(run["seed"]): run["split_sha256"]
                                 for run in dev["runs"]},
        "selected_fact_indices_by_seed": {str(run["seed"]):
                                          run["selected_fact_indices"]
                                          for run in dev["runs"]},
        "selected_fact_names_by_seed": {str(run["seed"]):
                                        run["selected_fact_names"]
                                        for run in dev["runs"]},
        "parent_full_model_sha256_by_seed": {str(run["seed"]):
                                             run["full_model_sha256"]
                                             for run in dev["runs"]},
        "threshold_by_seed": {str(run["seed"]):
                              run["alternatives"][selected]["threshold"]
                              for run in dev["runs"]},
        "prior_test_results_on_same_account_population_already_observed": True,
        "test_metric_or_prediction_used_to_create_lock": False,
    }
    destination = root / "output/v3_mgtab_named_top20_lightgbm/selection_lock.json"
    destination.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(selected, dev["configs"][selected], mean_f1[selected], flush=True)


if __name__ == "__main__":
    main()
