"""Lock one Fox8 native-evidence spline strength using validation only."""

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
    if (screen["dataset"] != "fox8"
            or screen["scope"] != "validation_only_native_ab_spline_l2"
            or screen["test_predictions_accessed"] is not False):
        raise ValueError("selection screen is not the Fox8 validation-only screen")
    candidates = [str(value) for value in screen["l2_candidates"]]
    means = {
        candidate: {
            metric: statistics.mean(run["alternatives"][candidate]["validation"][metric]
                                    for run in screen["runs"])
            for metric in ("macro_f1", "acc", "auroc")
        }
        for candidate in candidates
    }
    selected = max(candidates, key=lambda candidate: (
        means[candidate]["macro_f1"], means[candidate]["acc"],
        means[candidate]["auroc"], float(candidate)))
    lock = {
        "dataset": "fox8",
        "selected_model": "spline_43_native",
        "selected_l2": float(selected),
        "selection_rule": "maximum mean validation Macro-F1, then ACC, AUROC, then stronger L2",
        "candidate_mean_validation_metrics": means,
        "screen_sha256": sha256_file(args.screen),
        "base_feature_sha256": screen["base_feature_sha256"],
        "native_facts_sha256": screen["native_facts_sha256"],
        "native_source_audit_sha256": screen["native_source_audit_sha256"],
        "candidate_test_predictions_accessed": False,
        "prior_fox8_test_predictions_already_observed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
