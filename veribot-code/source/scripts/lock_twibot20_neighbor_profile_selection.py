"""Lock the official-dev winner before evaluating the neighbor candidate on test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.screen.read_text(encoding="utf-8"))
    if report["scope"] != "official_dev_only" or report["test_predictions_accessed"]:
        raise ValueError("candidate screen must use official dev only")
    labels = ("dual_tail_22", "dual_tail_32_neighbor")
    means = {label: {
        metric: statistics.mean(row[label]["validation"][metric]
                                for row in report["runs"])
        for metric in ("acc", "macro_f1", "auroc")}
        for label in labels}
    selected = max(labels, key=lambda label: (
        means[label]["macro_f1"], means[label]["acc"]))
    result = {
        "dataset": "twibot20", "selection_scope": "official_dev_only",
        "criterion": "maximum three-seed mean dev Macro-F1, tie by mean dev ACC",
        "selected_model": selected,
        "candidate_mean_dev_metrics": means,
        "screen_sha256": sha256(args.screen),
        "base_facts_sha256": report["base_facts_sha256"],
        "neighbor_facts_sha256": report["neighbor_facts_sha256"],
        "neighbor_source_audit_sha256": report["source_audit_sha256"],
        "candidate_test_predictions_accessed": False,
        "prior_22_fact_test_was_viewed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
