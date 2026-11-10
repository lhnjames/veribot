"""Lock a validation-selected A/B monotone MGTAB candidate before its test run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gate0_shallow_tree import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screens", type=Path, nargs="+", required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = {}
    for path in args.screens:
        screen = json.loads(path.read_text(encoding="utf-8"))
        if (screen["dataset"] != "mgtab" or screen["test_predictions_accessed"]
                or screen["scope"] != "validation_only_ab_monotone_spline"):
            raise ValueError("expected untouched A/B validation screen")
        for strength in screen["l2_candidates"]:
            key = str(strength)
            if key in scores:
                raise ValueError("duplicate regularization candidate")
            scores[key] = sum(row["alternatives"][key]["validation"]["macro_f1"]
                              for row in screen["runs"]) / len(screen["runs"])
    selected = max(scores, key=lambda key: (scores[key], -float(key)))
    lock = {"dataset": "mgtab", "selected_reasoner": "ab_monotone_interval_spline",
            "selected_l2": float(selected),
            "selection_metric": "three_seed_mean_dev_macro_f1",
            "candidate_scores": scores,
            "candidate_test_predictions_accessed": False,
            "prior_other_architecture_test_predictions_already_observed": True,
            "representation_scope": "published tensor statistical content coordinates; not semantic tweet evidence",
            "screens_sha256": {str(path): sha256_file(path) for path in args.screens},
            "source_audit_sha256": sha256_file(args.source_audit),
            "structural_facts_sha256": sha256_file(args.structural_facts),
            "neighbor_facts_sha256": sha256_file(args.neighbor_facts),
            "source_features_sha256": sha256_file(args.source_features)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
