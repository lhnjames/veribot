"""Freeze the three-seed Fox8 validation winner before test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    if screen["scope"] != "validation_only" or screen["test_predictions_accessed"]:
        raise ValueError("Fox8 screen must contain validation only")
    names = ("single_tail", "dual_tail", "monotone_spline")
    means = {name: {
        metric: statistics.mean(run[name]["validation"][metric]
                                for run in screen["runs"])
        for metric in ("acc", "macro_f1", "auroc")}
        for name in names}
    winner = max(names, key=lambda name: (means[name]["macro_f1"],
                                          means[name]["acc"]))
    lock = {
        "dataset": "fox8", "selected_model": winner,
        "selection_rule": screen["selection_rule"],
        "candidate_mean_validation_metrics": means,
        "screen_sha256": sha256_file(args.screen),
        "raw_feature_audit_sha256": screen["raw_feature_audit_sha256"],
        "feature_sha256": screen["feature_sha256"],
        "candidate_test_predictions_accessed": False,
        "prior_single_tail_test_was_viewed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
