"""Lock one Fox8 evidence-only neighbor classifier from official dev splits."""

from __future__ import annotations

import json
from pathlib import Path

from gate0_shallow_tree import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dev_path = root / "output/v3_fox8_neighbor_atomic_evidence_capacity/dev_only.json"
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    if (dev["scope"] != "fox8_train_dev_only_atomic_source_evidence_neighbor_capacity"
            or dev["new_test_predictions_or_metrics_computed"] is not False
            or len(dev["runs"]) != 3):
        raise ValueError("Unexpected validation source")
    ranked = []
    for variant in ("raw_neighbor_74", "hypbot_scaled_neighbor_74"):
        for config_index in range(2):
            scores = [run["variants"][variant]["model_results"][config_index]
                      ["validation_at_half"]["macro_f1"] for run in dev["runs"]]
            ranked.append({"variant": variant, "config_index": config_index,
                           "mean_dev_macro_f1_at_fixed_half": sum(scores) / len(scores)})
    ranked.sort(key=lambda item: (-item["mean_dev_macro_f1_at_fixed_half"],
                                  item["config_index"], item["variant"]))
    chosen = ranked[0]
    if chosen["variant"] != "raw_neighbor_74" or chosen["config_index"] != 0:
        raise ValueError("Predefined validation choice changed")
    lock = {
        "dataset": "fox8",
        "scope": "exploratory_evidence_only_native_neighbor_test_after_official_dev_lock",
        "selection_rule": "largest mean official-dev Macro-F1 at prespecified 0.5 threshold among two source-audited neighbor representations and two capacities; smaller capacity on tie",
        "selected": chosen,
        "ranking": ranked,
        "config": dev["configs"][0],
        "threshold": 0.5,
        "dev_result_sha256": sha256_file(dev_path),
        "raw_neighbor_sha256": dev["raw_neighbor_sha256"],
        "raw_neighbor_audit_sha256": dev["raw_neighbor_audit_sha256"],
        "split_sha256_by_seed": {str(run["seed"]): run["split_sha256"]
                                  for run in dev["runs"]},
        "reference_version_by_seed": {
            str(run["seed"]): run["variants"]["raw_neighbor_74"]["reference_version"]
            for run in dev["runs"]},
        "prior_results_on_same_test_population_already_observed": True,
        "test_metric_or_prediction_used_to_create_lock": False,
    }
    path = root / "output/v3_fox8_neighbor_atomic_evidence_capacity/selection_lock.json"
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(chosen, flush=True)


if __name__ == "__main__":
    main()
