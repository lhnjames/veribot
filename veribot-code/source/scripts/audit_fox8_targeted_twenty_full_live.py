"""Replay all fixed-test Fox8 20-concept decisions from source-bound runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from verifiable_bot.fox8_targeted_twenty_runtime import Fox8TargetedTwentyRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/certificates/fox8_targeted_twenty_full_live.json"
FOLDER = ROOT / "output/v3_fox8_targeted_twenty_ab_group"


def run(seed: int) -> dict:
    saved_path = FOLDER / f"seed{seed}/test_predictions.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))["predictions"]
    account_ids = [entry["account_id"] for entry in saved]
    original_csv = pd.read_csv
    original_text = Path.read_text
    original_joblib_load = joblib.load

    def guarded_csv(path, *args, **kwargs):
        if Path(path).name == "labels.csv":
            raise AssertionError("runtime attempted to read class labels")
        return original_csv(path, *args, **kwargs)

    def guarded_text(path, *args, **kwargs):
        if Path(path).name == "test_predictions.json":
            raise AssertionError("runtime attempted to read saved test predictions")
        return original_text(path, *args, **kwargs)

    def guarded_joblib(path, *args, **kwargs):
        if Path(path).resolve() != (FOLDER / f"seed{seed}/model.joblib").resolve():
            raise AssertionError("runtime attempted to load another classifier")
        return original_joblib_load(path, *args, **kwargs)

    with patch.object(pd, "read_csv", guarded_csv), patch.object(
            Path, "read_text", guarded_text), patch.object(
                joblib, "load", guarded_joblib):
        runtime = Fox8TargetedTwentyRuntime(ROOT, seed)
        decisions = runtime.predict_many(account_ids)
        test_id_set = set(account_ids)
        non_test_id = next(account for account in runtime.ids
                           if account not in test_id_set)
        non_test_probe = runtime.predict(non_test_id)
    if len(decisions) != len(saved) or len(decisions) != 343:
        raise ValueError("fixed test population differs")
    if (non_test_probe["account_id"] != non_test_id or
            len(non_test_probe["concepts"]) != 20):
        raise ValueError("non-test source-bound probe differs")
    groups = [[2*k, 2*k+1, 20+2*k, 21+2*k] for k in range(10)]
    single_group, counts = all_trees_one_group(runtime.model, groups)
    if not single_group or {str(k): v for k, v in counts.items()} != runtime.record[
            "tree_count_by_group"]:
        raise ValueError("loaded model crossed evidence groups")
    max_probability = max_neighbor = max_additivity = 0.0
    source_bound = 0
    neighbor_changed = 0
    original_correct = masked_correct = rescued = harmed = 0
    for actual, expected in zip(decisions, saved, strict=True):
        if (actual["account_id"] != expected["account_id"] or
                actual["prediction"] != expected["prediction"] or
                len(actual["concepts"]) != 20 or
                len(actual["source_group_effects"]) != 10):
            raise ValueError("loaded runtime changes fixed decision")
        max_probability = max(max_probability,
                              abs(actual["probability"] -
                                  expected["probability"]))
        max_additivity = max(max_additivity,
                             actual["group_additivity_error"])
        neighbor = next(item for item in actual["source_group_effects"]
                        if item["group"] == "native_neighbor_volume")
        max_neighbor = max(max_neighbor,
                           abs(neighbor["probability_without_group"] -
                               expected["without_neighbor_probability"]))
        neighbor_changed += int(neighbor["class_changes_without_group"])
        full_right = actual["prediction"] == expected["label"]
        masked_right = int(neighbor["probability_without_group"] >=
                           runtime.threshold) == expected["label"]
        original_correct += int(full_right)
        masked_correct += int(masked_right)
        rescued += int(full_right and not masked_right)
        harmed += int(masked_right and not full_right)
        for term in actual["concepts"]:
            if term["available"]:
                if term["winning_source_fact"] is None or term[
                        "source_locator"] is None:
                    raise ValueError("available concept lacks source")
                source_bound += 1
    if max(max_probability, max_neighbor, max_additivity) > 1e-10:
        raise ValueError("frozen source-bound runtime differs")
    return {
        "seed": seed,
        "fixed_test_accounts": len(decisions),
        "concepts_per_account": 20,
        "source_bound_available_concepts": source_bound,
        "max_saved_probability_difference": max_probability,
        "max_saved_neighbor_mask_probability_difference": max_neighbor,
        "max_group_additivity_error": max_additivity,
        "neighbor_group_mask_changed_classes": neighbor_changed,
        "original_correct": original_correct,
        "without_neighbor_group_correct": masked_correct,
        "correct_recovered_by_neighbor_group": rescued,
        "correct_lost_due_to_neighbor_group": harmed,
        "non_test_probe_account_id": non_test_id,
        "non_test_probe_probability": non_test_probe["probability"],
        "class_label_file_access_guard_passed": True,
        "saved_test_prediction_access_guard_passed": True,
        "other_classifier_weight_access_guard_passed": True,
        "test_predictions_sha256": sha256_file(saved_path),
        "model_sha256": runtime.record["model_sha256"],
        "reference_version": runtime.reference.version,
    }


def main() -> None:
    rows = [run(seed) for seed in (0, 1, 2)]
    OUT.write_text(json.dumps({
        "status": "all_fox8_twenty_concept_source_bound_test_predictions_replayed",
        "source_fact_audit": "certificates/fox8_neighbor_atomic_evidence_full_replay.json",
        "selection_lock_sha256": sha256_file(FOLDER / "selection_lock.json"),
        "runs": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print({"accounts": sum(row["fixed_test_accounts"] for row in rows),
           "max_probability_difference": max(row[
               "max_saved_probability_difference"] for row in rows),
           "max_group_additivity_error": max(row[
               "max_group_additivity_error"] for row in rows),
           "neighbor_changes": [row["neighbor_group_mask_changed_classes"]
                                for row in rows]})
    print(OUT)


if __name__ == "__main__":
    main()
