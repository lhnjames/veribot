"""Recheck all fixed-test bounded-residual arithmetic and paired counts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from scipy.special import expit


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    directory = root / "output/v3_mgtab_stronger_bounded_residual/selected"
    summary_path = directory / "paired_summary.json"
    summary = read(summary_path)
    lock_path = root / "output/v3_mgtab_stronger_bounded_residual/selection_lock.json"
    if summary["selection_lock_sha256"] != digest(lock_path):
        raise ValueError("locked test summary differs")
    results = []
    for run in summary["runs"]:
        seed = run["seed"]
        candidate_path = directory / f"seed{seed}/test_predictions.json"
        hypbot_path = root / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        selected_path = root / f"output/v3_mgtab_ab_spline_content/direct_selected_20/seed{seed}/test_predictions.json"
        candidate = read(candidate_path)
        hypbot = read(hypbot_path)
        selected = read(selected_path)
        if (digest(candidate_path) != run["predictions_sha256"] or
                digest(hypbot_path) != run["hypbot_predictions_sha256"] or
                digest(selected_path) != run["selected_predictions_sha256"] or
                candidate["manifest"]["selection_lock_sha256"] != digest(lock_path) or
                len(candidate["predictions"]) != run["test_n"] or
                len(hypbot["predictions"]) != run["test_n"] or
                len(selected["predictions"]) != run["test_n"]):
            raise ValueError("paired frozen files differ")
        threshold = candidate["manifest"]["threshold"]
        h_threshold = hypbot["manifest"]["threshold"]
        s_threshold = selected["manifest"]["threshold"]
        correct = h_correct = s_correct = candidate_only = hypbot_only = 0
        override = 0
        max_probability_difference = 0.0
        max_residual = 0.0
        boundary = math.log(threshold / (1 - threshold))
        for item, other, chosen in zip(candidate["predictions"],
                                       hypbot["predictions"],
                                       selected["predictions"], strict=True):
            if (item["account_id"] != other["account_id"] or
                    item["account_id"] != chosen["account_id"] or
                    item["label"] != other["label"] or
                    item["label"] != chosen["label"]):
                raise ValueError("paired account or label differs")
            probability = float(expit(item["evidence_logit"] +
                                      item["residual_logit"]))
            max_probability_difference = max(max_probability_difference,
                                             abs(probability - item["probability"]))
            max_residual = max(max_residual, abs(item["residual_logit"]))
            if (item["evidence_logit"] >= boundary) != (
                    item["evidence_logit"] + item["residual_logit"] >= boundary):
                override += 1
            is_correct = int(item["probability"] >= threshold) == item["label"]
            h_is_correct = int(other["probability"] >= h_threshold) == item["label"]
            s_is_correct = int(chosen["probability"] >= s_threshold) == item["label"]
            correct += is_correct
            h_correct += h_is_correct
            s_correct += s_is_correct
            candidate_only += is_correct and not h_is_correct
            hypbot_only += h_is_correct and not is_correct
        if (max_probability_difference > 1e-12 or max_residual > 8.0 + 1e-12 or
                abs(correct / run["test_n"] - run["candidate"]["acc"]) > 1e-12 or
                abs(h_correct / run["test_n"] - run["hypbot_acc"]) > 1e-12 or
                abs(s_correct / run["test_n"] - run["selected_20_evidence_acc"]) > 1e-12 or
                candidate_only != run["candidate_correct_hypbot_wrong"] or
                hypbot_only != run["candidate_wrong_hypbot_correct"] or
                override != run[
                    "residual_changes_evidence_only_class_at_locked_threshold"]):
            raise ValueError("bounded residual arithmetic or paired counts differ")
        results.append({"seed": seed, "test_n": run["test_n"],
                        "correct_candidate": correct, "correct_hypbot": h_correct,
                        "correct_existing_20": s_correct,
                        "candidate_only_correct": candidate_only,
                        "hypbot_only_correct": hypbot_only,
                        "residual_changes_evidence_only_class": override,
                        "max_absolute_residual_logit": max_residual,
                        "max_probability_reconstruction_difference":
                            max_probability_difference})
    output = {"scope": "all_fixed_test_accounts_locked_residual_independent_arithmetic",
              "selection_lock_sha256": digest(lock_path),
              "paired_summary_sha256": digest(summary_path), "runs": results}
    destination = root / "research/mgtab_locked_residual_independent_arithmetic_audit_2026-11-03.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
