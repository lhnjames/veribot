"""Apply the predeclared dev capacity gate without reading fixed test data."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCREEN = ROOT / "output/v3_mgtab_twenty_named_bounded_linear_residual/dev_only.json"
PARENT = ROOT / "output/v3_mgtab_primary_reference_mixture/dev_only.json"
OUT = ROOT / "research/certificates/mgtab_twenty_named_bounded_linear_residual_dev_gate.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, default=SCREEN)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    if (screen["test_predictions_or_metrics_accessed"] is not False
            or parent["test_predictions_or_metrics_accessed"] is not False
            or len(screen["runs"]) != 3 or len(parent["runs"]) != 3):
        raise ValueError("Dev-only scope differs")
    parent_f1 = np.asarray([r["original_all_train_validation"]["macro_f1"]
                            for r in parent["runs"]])
    parent_acc = np.asarray([r["original_all_train_validation"]["acc"]
                             for r in parent["runs"]])
    choices = []
    configurations = sorted({(x["residual_fit_half"], x["l2"])
                             for run in screen["runs"] for x in run["candidates"]})
    for half, l2 in configurations:
        rows = [next(x for x in run["candidates"]
                     if x["residual_fit_half"] == half and x["l2"] == l2)
                for run in screen["runs"]]
        f1 = np.asarray([x["validation"]["macro_f1"] for x in rows])
        acc = np.asarray([x["validation"]["acc"] for x in rows])
        passes = bool(f1.mean() >= parent_f1.mean() - .005 and
                      np.all(acc >= parent_acc - .01))
        choices.append({"residual_fit_half": half, "l2": l2,
                        "mean_dev_macro_f1": float(f1.mean()),
                        "dev_acc_by_seed": acc.tolist(),
                        "passes_predeclared_gate": passes})
    named_f1 = np.asarray([r["named_only_validation"]["macro_f1"]
                           for r in screen["runs"]])
    result = {
        "scope": "mgtab_twenty_named_source_concepts_plus_low_capacity_residual_dev_gate",
        "group_scheme": screen.get("group_scheme", "original_three_profile_seven_relation"),
        "test_predictions_or_metrics_accessed": False,
        "gate": "mean dev Macro-F1 within 0.5 percentage point and every seed dev ACC within 1 percentage point of frozen 150-fact parent",
        "parent_mean_dev_macro_f1": float(parent_f1.mean()),
        "parent_dev_acc_by_seed": parent_acc.tolist(),
        "named_only_mean_dev_macro_f1": float(named_f1.mean()),
        "named_only_gap_from_parent_macro_f1_percentage_points": float(
            100*(parent_f1.mean() - named_f1.mean())),
        "choices": choices,
        "any_candidate_passes": any(x["passes_predeclared_gate"] for x in choices),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print({"any_candidate_passes": result["any_candidate_passes"],
           "named_only_gap_pp": result[
               "named_only_gap_from_parent_macro_f1_percentage_points"],
           "best_mean_dev_macro_f1": max(x["mean_dev_macro_f1"]
                                         for x in choices)}, flush=True)
    print(args.output)


if __name__ == "__main__":
    main()
