"""Check contrary-evidence omissions among official fixed-test errors.

Official labels enter only this post hoc audit, never model inference, effect
selection, or construction of the balanced decision cards.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
BALANCE = ROOT / "research/certificates/best_reason_direction_balance.json"
SIDE = ROOT / "research/certificates/best_reason_direction_balance.jsonl.gz"
OUT = ROOT / "research/certificates/best_reason_direction_error_audit.json"
SAVED = {
    "botsim": ROOT / "output/v3_botsim_observed_means_12",
    "fox8": ROOT / "output/v3_fox8_neighbor_atomic_evidence_capacity",
    "mgtab": ROOT / "output/v3_mgtab_ab_group_additive_tree",
}
TOL = 1e-10


def main() -> None:
    balance = json.loads(BALANCE.read_text(encoding="utf-8"))
    if (balance["scope"] != "nine_fixed_test_runs_5400_account_decisions" or
            sha256_file(SIDE) != balance["account_sidecar_sha256"]):
        raise ValueError("balanced card population or hash differs")
    summary = []
    error_examples = []
    all_errors = []
    with gzip.open(SIDE, "rt", encoding="utf-8") as side_stream:
        for run in balance["runs"]:
            dataset, seed = run["dataset"], int(run["seed"])
            saved_path = SAVED[dataset] / f"seed{seed}/test_predictions.json"
            reason_path = (ROOT / "output/v3_best_decision_reasons" /
                           f"{dataset}_seed{seed}.jsonl.gz")
            source_path = (ROOT / "output/v3_best_verified_evidence_full_test" /
                           f"{dataset}_seed{seed}.jsonl.gz")
            if (sha256_file(reason_path) != run["reason_cards_sha256"] or
                    sha256_file(source_path) != run["source_cards_sha256"]):
                raise ValueError("reason or source cards changed")
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            predictions = saved["predictions"]
            if len(predictions) != run["accounts"]:
                raise ValueError("official fixed-test population differs")
            counts = Counter()
            error_by_reason = Counter()
            critical_groups = Counter()
            with gzip.open(reason_path, "rt", encoding="utf-8") as reason_stream, \
                    gzip.open(source_path, "rt", encoding="utf-8") as source_stream:
                for i, (official, reason_line, source_line) in enumerate(zip(
                        predictions, reason_stream, source_stream, strict=True)):
                    side = json.loads(next(side_stream))
                    reason = json.loads(reason_line)
                    source = json.loads(source_line)
                    if (side["dataset"] != dataset or
                            side["seed"] != seed or
                            side["account_id"] != str(official["account_id"]) or
                            reason["account_id"] != side["account_id"] or
                            source["account_id"] != side["account_id"] or
                            side["reason_card_sha256"] != run[
                                "reason_cards_sha256"] or
                            side["source_card_sha256"] != run[
                                "source_cards_sha256"] or
                            abs(float(official["probability"]) -
                                float(reason["probability"])) > TOL or
                            abs(float(source["probability"]) -
                                float(reason["probability"])) > TOL or
                            int(reason["prediction"]) != side["prediction"] or
                            int(source["prediction"]) != side["prediction"] or
                            float(source["threshold"]) != float(
                                reason["threshold"]) or
                            int(reason["prediction"]) != int(
                                reason["probability"] >= reason["threshold"])):
                        raise ValueError("official prediction/card alignment differs")
                    if ("prediction" in official and
                            int(official["prediction"]) != side["prediction"]):
                        raise ValueError("saved prediction disagrees with card")
                    label = int(official["label"])
                    wrong = label != side["prediction"]
                    omitted = side["opposing_omitted_from_display"]
                    dependency = source["exact_frozen_model_dependency"]
                    group_critical_despite_no_single_or_all_change = bool(
                        reason["reason_category"] ==
                        "no_class_change_from_all_evidence" and
                        any(effect["class_changes_without_group"]
                            for effect in dependency.get(
                                "source_group_masks", {}).values()))
                    if (group_critical_despite_no_single_or_all_change !=
                            side[
                                "group_critical_despite_no_single_or_all_change"]):
                        raise ValueError("source group effect disagrees with sidecar")
                    zero_probability = float(dependency[
                        "all_evidence_zero_probability"])
                    zero_prediction = int(zero_probability >=
                                          reason["threshold"])
                    if (bool(zero_prediction != side["prediction"]) !=
                            dependency["all_evidence_changes_class"] or
                            bool(reason["all_evidence_changes_class"]) !=
                            dependency["all_evidence_changes_class"]):
                        raise ValueError("all-evidence zero class mismatch")
                    counts["accounts"] += 1
                    counts["errors"] += wrong
                    counts["false_positive"] += wrong and label == 0
                    counts["false_negative"] += wrong and label == 1
                    counts["errors_same_as_zero_input_class"] += (
                        wrong and zero_prediction == side["prediction"])
                    counts["errors_changed_from_zero_input_class"] += (
                        wrong and zero_prediction != side["prediction"])
                    counts[
                        "errors_group_critical_despite_no_single_or_all_change"] += (
                            wrong and
                            group_critical_despite_no_single_or_all_change)
                    counts["opposition_present"] += side["opposing_count"] > 0
                    counts["errors_with_opposition"] += wrong and side[
                        "opposing_count"] > 0
                    counts["errors_with_old_omission"] += wrong and omitted
                    counts["correct_with_old_omission"] += not wrong and omitted
                    if wrong:
                        error_by_reason[reason["reason_category"]] += 1
                        groups = [name for name, effect in dependency.get(
                            "source_group_masks", {}).items()
                            if effect["class_changes_without_group"]]
                        critical_groups.update(groups)
                        all_errors.append({
                            "dataset": dataset, "seed": seed,
                            "account_id": side["account_id"],
                            "official_label": label,
                            "prediction": side["prediction"],
                            "probability": reason["probability"],
                            "threshold": reason["threshold"],
                            "zero_input_probability": zero_probability,
                            "zero_input_prediction": zero_prediction,
                            "reason_category": reason["reason_category"],
                            "single_input_class_critical_count": reason[
                                "class_critical_fact_count"],
                            "class_critical_source_groups": groups,
                            "group_critical_despite_no_single_or_all_change":
                                group_critical_despite_no_single_or_all_change,
                            "old_card_omitted_opposition": omitted,
                        })
                    if wrong and omitted:
                        extra = side[
                            "additional_opposing_item_for_balanced_card"]
                        if extra is None or extra["source_query"] is None:
                            raise ValueError("wrong decision lacks contrary source")
                        error_examples.append({
                            "dataset": dataset, "seed": seed,
                            "account_id": side["account_id"],
                            "official_label": label,
                            "prediction": side["prediction"],
                            "probability": reason["probability"],
                            "threshold": reason["threshold"],
                            "reason_category": side["reason_category"],
                            "five_critical_items_occupied_display":
                                side["class_critical_displayed_count"] == 5,
                            "strongest_omitted_contrary_input_effect": extra,
                        })
            if counts["accounts"] != run["accounts"] or (
                    counts["errors_with_old_omission"] +
                    counts["correct_with_old_omission"] !=
                    run["opposing_omitted_from_display"]):
                raise ValueError("run omission counts disagree")
            summary.append({
                "dataset": dataset, "seed": seed,
                **dict(counts),
                "official_predictions_sha256": sha256_file(saved_path),
                "reason_cards_sha256": run["reason_cards_sha256"],
                "source_cards_sha256": run["source_cards_sha256"],
                "error_by_reason_category": dict(error_by_reason),
                "class_critical_source_group_occurrences_among_errors": dict(
                    critical_groups),
            })
        if next(side_stream, None) is not None:
            raise ValueError("balanced cards exceed fixed predictions")
    OUT.write_text(json.dumps({
        "status": "all_fixed_test_errors_checked_for_contrary_display_omissions",
        "scope": "nine_fixed_test_runs_5400_account_decisions",
        "official_labels_used_only_after_balanced_cards_were_frozen": True,
        "effect_scope": "exact_frozen_encoded_input_withholding_not_raw_source_causality",
        "balance_summary_sha256": sha256_file(BALANCE),
        "balance_sidecar_sha256": sha256_file(SIDE),
        "runs": summary,
        "all_errors": all_errors,
        "errors_with_old_omission": error_examples,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)
    print({"accounts": sum(x["accounts"] for x in summary),
           "errors": sum(x["errors"] for x in summary),
           "errors_with_old_omission": len(error_examples)})


if __name__ == "__main__":
    main()
