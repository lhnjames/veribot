"""Evaluate label-blind source-correction orderings at fixed account budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if (source["dataset"] != "twibot20" or source["model"] !=
            "source_backed_unigram_l1_c1" or source["injection"]["uses_class_labels"]
            or source["max_source_correction_probability_difference"] != 0):
        raise ValueError("expected source-corrected frozen sparse model")
    rows = source["rows"]
    threshold = float(source["threshold"])
    ids = [str(row["account_id"]) for row in rows]
    if len(ids) != len(set(ids)) or len(ids) != source["selected_accounts"]:
        raise ValueError("selected account population differs")
    if any(float(row["source_value"]) != 1.0 - float(row["injected_value"])
           for row in rows):
        raise ValueError("counterfactual ranking requires the binary alternative")
    labels = np.asarray([int(row["label_for_evaluation"]) for row in rows])
    bad_probability = np.asarray([float(row["corrupted_probability"]) for row in rows])
    repaired_probability = np.asarray([float(row["corrected_probability"]) for row in rows])
    bad_error = (bad_probability >= threshold) != labels
    repaired_error = (repaired_probability >= threshold) != labels
    fixed_other_errors = int(source["corrupted"]["other_errors"])
    if (fixed_other_errors + int(bad_error.sum()) != source["corrupted"]["all_errors"]
            or fixed_other_errors + int(repaired_error.sum()) != source["source_corrected"]["all_errors"]):
        raise ValueError("per-account rows do not reproduce reported error totals")
    # In this binary injected-error experiment the alternate field value is
    # computable before source review; rankings never read Bot/Human labels.
    strategies = {
        "hash_random": sorted(range(len(rows)), key=lambda i: hashlib.sha256(
            ("veribot-source-correction-order-v1:" + ids[i]).encode()).digest()),
        "model_uncertainty": sorted(range(len(rows)), key=lambda i:
                                    (abs(bad_probability[i] - threshold), ids[i])),
        "counterfactual_score_change": sorted(range(len(rows)), key=lambda i:
                                              (-abs(repaired_probability[i] - bad_probability[i]), ids[i])),
    }
    budgets = [0, 10, 20, 50, 100, 150, len(rows)]
    curves = {}
    for name, ordering in strategies.items():
        points = []
        for budget in budgets:
            selected = np.asarray(ordering[:budget], dtype=int)
            corrected_errors = bad_error.copy()
            corrected_errors[selected] = repaired_error[selected]
            direct_errors = bad_error.copy()
            direct_errors[selected] = False
            points.append({
                "account_actions": budget,
                "source_correction_total_errors": fixed_other_errors + int(corrected_errors.sum()),
                "source_correction_net_errors_removed": int(bad_error.sum() - corrected_errors.sum()),
                "direct_account_label_total_errors": fixed_other_errors + int(direct_errors.sum()),
                "direct_account_label_net_errors_removed": int(bad_error.sum() - direct_errors.sum()),
            })
        curves[name] = points
    result = {
        "dataset": "twibot20", "model": source["model"],
        "source_correction_file": str(args.input),
        "injection": source["injection"],
        "ranking_uses_bot_human_labels": False,
        "counterfactual_rank_scope": "binary verified-field alternative under the frozen model; known injected-error setting",
        "comparison_unit": "account actions; human time and natural error prevalence not measured",
        "selected_accounts": len(rows),
        "corrupted_all_errors": int(source["corrupted"]["all_errors"]),
        "clean_all_errors": int(source["clean"]["all_errors"]),
        "curves": curves,
    }
    if any(points[-1]["source_correction_total_errors"] != source["clean"]["all_errors"]
           or points[-1]["direct_account_label_total_errors"] !=
           source["direct_account_label_same_account_count"]["all_errors"]
           for points in curves.values()):
        raise ValueError("full-budget endpoints differ from source correction experiment")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: points for name, points in curves.items()}, indent=2))


if __name__ == "__main__":
    main()
