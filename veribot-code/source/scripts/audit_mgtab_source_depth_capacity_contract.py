"""Join MGTAB named-source capacity and frozen content-dependence evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAPACITY = ROOT / "output/v3_mgtab_named_raw_capacity/dev_only.json"
DEPENDENCE = ROOT / "research/certificates/mgtab_named_source_decision_cohort.json"
SCHEMA18 = ROOT / "output/gate1_mgtab_schema18/dev_only.json"
OUT = ROOT / "research/certificates/mgtab_source_depth_capacity_contract.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    capacity = read(CAPACITY)
    dependence = read(DEPENDENCE)
    schema18 = read(SCHEMA18)
    if (capacity["dataset"] != "mgtab" or
            capacity["test_predictions_accessed"] is not False or
            schema18["test_predictions_or_metrics_accessed"] is not False or
            len(capacity["runs"]) != 3 or len(dependence["runs"]) != 3):
        raise ValueError("source capacity artifacts differ from the expected scope")
    rows = []
    for cap, dep in zip(capacity["runs"], dependence["runs"], strict=True):
        if cap["seed"] != dep["seed"]:
            raise ValueError("seed mismatch")
        control = cap["capacity_controls"]
        named = control["shallow_hist_gradient_boosting"]["validation"]
        real = control["shallow_hist_gradient_boosting_all_150"]["validation"]
        shuffled = control["shallow_hist_gradient_boosting_shuffled_64_content"]["validation"]
        if (named["n"] != real["n"] or named["n"] != shuffled["n"] or
                dep["test_accounts"] != 1020 or
                dep["projection_dependent_accounts"] +
                dep["class_invariant_without_projection_accounts"] != 1020):
            raise ValueError("population or frozen dependence count mismatch")
        rows.append({
            "seed": cap["seed"],
            "dev_accounts": named["n"],
            "same_shallow_tree_named_86_errors": named["errors"],
            "same_shallow_tree_named_86_plus_real_content_64_errors": real["errors"],
            "same_shallow_tree_named_86_plus_shuffled_content_64_errors": shuffled["errors"],
            "dev_error_reduction_real_content_vs_named": named["errors"] - real["errors"],
            "frozen_test_accounts": dep["test_accounts"],
            "frozen_projection_dependent_classes": dep["projection_dependent_accounts"],
            "frozen_full_correct": dep["full_correct"],
            "frozen_named_groups_only_correct": dep["named_groups_only_correct"],
            "frozen_correct_gain_with_projection":
                dep["full_correct"] - dep["named_groups_only_correct"],
        })
    if ([row["frozen_projection_dependent_classes"] for row in rows] !=
            [104, 120, 140] or
            [row["frozen_correct_gain_with_projection"] for row in rows] !=
            [44, 78, 70]):
        raise ValueError("frozen content dependency differs")
    result = {
        "scope": "MGTAB released-tensor source-depth versus named evidence capacity",
        "capacity_control_uses_same_train_dev_shallow_tree_family": True,
        "frozen_content_zeroing_is_model_input_stress_test_not_raw_source_correction": True,
        "published_mean_tweet_tensor_has_no_per_tweet_text_or_event_ids_here": True,
        "schema18_is_another_capacity_protocol_not_directly_paired_with_86_fact_tree": True,
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
                          for path in (CAPACITY, DEPENDENCE, SCHEMA18)},
        "rows": rows,
        "totals_across_three_fixed_splits_descriptive_not_independent_population": {
            "frozen_projection_dependent_class_events": sum(
                row["frozen_projection_dependent_classes"] for row in rows),
            "frozen_correct_gain_with_projection_events": sum(
                row["frozen_correct_gain_with_projection"] for row in rows),
        },
        "interpretation": (
            "Named profile and typed-relation facts alone lose information that "
            "the released mean-tweet tensor supplies. The frozen evidence model "
            "uses that tensor to determine some classes. Source certification "
            "can stop at the published tensor and cannot infer per-tweet semantic truth."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
