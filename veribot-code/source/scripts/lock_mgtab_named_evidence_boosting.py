"""Freeze the dev-selected 86-fact evidence booster before test evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dev_path = root / "output/v3_mgtab_named_evidence_boosting/dev_only.json"
    raw = dev_path.read_bytes()
    report = json.loads(raw)
    if (report["scope"] != "mgtab_train_dev_only_named_source_evidence_boosting"
            or report["test_rows_accessed"] is not False
            or report["published_content_projection_used"] is not False
            or len(report["runs"]) != 3 or len(report["configs"]) != 4):
        raise ValueError("validation-only screening record differs")
    averages = [sum(run["alternatives"][index]["validation"]["macro_f1"]
                    for run in report["runs"]) / 3
                for index in range(4)]
    selected = max(range(4), key=lambda index: averages[index])
    if selected != 1 or report["configs"][selected] != [15, 300, 30, 10.0]:
        raise ValueError("preselected mean-Macro-F1 configuration differs")
    lock = {
        "dataset": "mgtab",
        "scope": "exploratory_named_source_evidence_test_after_dev_lock",
        "selection_rule": "largest mean official-dev Macro-F1 among four predefined evidence-only boosters",
        "dev_result_sha256": hashlib.sha256(raw).hexdigest(),
        "selected_config_index": selected,
        "selected_config": report["configs"][selected],
        "source_features_sha256": report["source_features_sha256"],
        "structural_facts_sha256": report["structural_facts_sha256"],
        "neighbor_facts_sha256": report["neighbor_facts_sha256"],
        "split_sha256_by_seed": {str(row["seed"]): row["split_sha256"]
                                 for row in report["runs"]},
        "threshold_by_seed": {str(row["seed"]):
                              row["alternatives"][selected]["threshold"]
                              for row in report["runs"]},
        "dev_macro_f1_by_seed": {str(row["seed"]):
                                 row["alternatives"][selected]["validation"]["macro_f1"]
                                 for row in report["runs"]},
        "prior_test_results_on_same_account_population_already_observed": True,
        "test_metric_or_prediction_used_to_create_lock": False,
    }
    destination = root / "output/v3_mgtab_named_evidence_boosting/selection_lock.json"
    destination.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
