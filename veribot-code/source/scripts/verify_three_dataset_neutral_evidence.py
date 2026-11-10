"""Bind nine fixed-test numeric results to source-bound signed-evidence audits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research/certificates/three_dataset_observed_means_numeric_goal.json"
CERT = ROOT / "research/certificates/three_dataset_neutral_evidence_numeric_and_source.json"
SPECS = {
    "botsim": (
        ROOT / "output/v3_botsim_neutral_threshold/fixed_test_report.json",
        ROOT / "research/certificates/botsim_neutral_threshold_runtime.json",
        .994),
    "fox8": (
        ROOT / "output/v3_fox8_twenty_neutral_anchor/fixed_test_report.json",
        ROOT / "research/certificates/fox8_twenty_neutral_anchor_runtime.json",
        .994),
    "mgtab": (
        ROOT / "output/v3_mgtab_150_neutral_anchor/fixed_test_report.json",
        ROOT / "research/certificates/mgtab_150_neutral_anchor_runtime.json",
        .912),
}


def main() -> None:
    baseline = json.loads(BASE.read_text(encoding="utf-8"))
    out = []
    total = 0
    for dataset, (test_path, audit_path, minimum) in SPECS.items():
        test = json.loads(test_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        old = next(x for x in baseline["datasets"] if x["dataset"] == dataset)
        if (audit["fixed_test_report_sha256"] != sha256_file(test_path) or
                len(test["runs"]) != 3 or len(audit["runs"]) != 3):
            raise ValueError("Fixed-test report and source audit are not bound")
        runs = []
        for seed in (0, 1, 2):
            row, check, earlier = (test["runs"][seed], audit["runs"][seed],
                                   old["runs"][seed])
            account_count = int(row["accounts"])
            correct = int(row["correct"])
            hypbot = int(earlier["hypbot_correct"])
            observed_hypbot = int(row[
                "hypbot_correct_on_same_test_population" if dataset == "mgtab"
                else "hypbot_correct_on_same_population"])
            matrix_hash = row[
                "predictions_sha256" if dataset == "mgtab"
                else "prediction_matrix_sha256"]
            if (row["seed"] != seed or check["seed"] != seed or
                    account_count != check["accounts"] or
                    account_count != earlier["test_accounts"] or
                    row["split_sha256" if dataset != "botsim"
                        else "source_split_sha256"] != earlier["split_sha256"] or
                    observed_hypbot != hypbot or
                    matrix_hash != check["prediction_matrix_sha256"] or
                    abs(row["metrics"]["acc"]-
                        correct/account_count) > 1e-12 or
                    not correct > hypbot or
                    not row["metrics"]["acc"] > minimum or
                    check["maximum_probability_replay_error"] > 1e-12):
                raise ValueError(f"{dataset} seed{seed} numeric/source mismatch")
            signed_key = ("all_accounts_have_signed_movement_from_neutral"
                          if dataset == "mgtab" else
                          "all_accounts_have_signed_evidence_movement")
            if check[signed_key] is not True:
                raise ValueError("A decision lacks signed evidence movement")
            group_error = check.get(
                "maximum_source_group_logit_reconstruction_error",
                check.get("maximum_group_logit_additivity_error"))
            if group_error is not None and group_error > 1e-9:
                raise ValueError("Source group logit reconstruction differs")
            matrix_path = Path(row[
                "predictions_path" if dataset == "mgtab" else
                "prediction_matrix_path"])
            if sha256_file(matrix_path) != matrix_hash:
                raise ValueError("Saved per-account predictions changed")
            with np.load(matrix_path, allow_pickle=False) as matrix:
                if len(matrix["account_ids"]) != account_count:
                    raise ValueError("Per-account prediction count differs")
            runs.append({
                "seed": seed, "accounts": account_count,
                "correct": correct, "acc": row["metrics"]["acc"],
                "macro_f1": row["metrics"]["macro_f1"],
                "auroc": row["metrics"]["auroc"],
                "hypbot_correct": hypbot,
                "numeric_goal_passed": True,
                "signed_source_score_and_live_replay_passed": True,
                "fixed_test_prediction_sha256": matrix_hash,
            })
            total += account_count
        out.append({
            "dataset": dataset, "required_acc_strictly_above": minimum,
            "all_three_splits_pass": True,
            "fixed_test_report_sha256": sha256_file(test_path),
            "source_runtime_audit_sha256": sha256_file(audit_path),
            "runs": runs})
    if total != 5400:
        raise ValueError("Nine official test populations do not sum to 5400")
    CERT.write_text(json.dumps({
        "status": "nine_fixed_splits_pass_numeric_and_signed_source_runtime_checks",
        "total_account_predictions": total,
        "all_numeric_and_source_runtime_gates_pass": True,
        "baseline_certificate_sha256": sha256_file(BASE),
        "source_claim_scope": "released records/tensors to encoded evidence and actual frozen model score",
        "external_semantic_validity_or_natural_correction_claimed": False,
        "test_population_previously_explored_by_multiple_candidates": True,
        "datasets": out}, indent=2) + "\n", encoding="utf-8")
    print(CERT)


if __name__ == "__main__":
    main()
