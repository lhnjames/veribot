"""Lock three-seed MGTAB sparse evidence choice from validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gate0_shallow_tree import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if (screen["scope"] != "validation_only"
            or screen["test_predictions_accessed"] is not False
            or audit["status"] != "verified_all_published_tensor_derived_fact_cells"
            or screen["source_audit_sha256"] != sha256_file(args.source_audit)
            or screen["structural_facts_sha256"] != sha256_file(args.structural_facts)
            or screen["neighbor_facts_sha256"] != sha256_file(args.neighbor_facts)
            or screen["source_features_sha256"] != sha256_file(args.features)):
        raise ValueError("dev screen, source audit or published tensor differs")
    candidates = ("l1_C0.1", "l1_C1", "l1_C10", "l2_C0.1", "l2_C1")
    scores = {name: sum(run[name]["validation"]["macro_f1"]
                        for run in screen["runs"]) / len(screen["runs"])
              for name in candidates}
    selected = max(candidates, key=lambda name: scores[name])
    if selected != "l1_C1":
        raise ValueError("selected reasoner has no fixed evaluator")
    report = {
        "dataset": "mgtab", "selected_model": selected,
        "selection_metric": "three_seed_mean_dev_macro_f1",
        "scores": scores,
        "candidate_test_predictions_accessed": False,
        "prior_86_and_projection_dual_tail_test_predictions_already_observed": True,
        "screen_sha256": sha256_file(args.screen),
        "source_audit_sha256": sha256_file(args.source_audit),
        "structural_facts_sha256": sha256_file(args.structural_facts),
        "neighbor_facts_sha256": sha256_file(args.neighbor_facts),
        "source_features_sha256": sha256_file(args.features),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
