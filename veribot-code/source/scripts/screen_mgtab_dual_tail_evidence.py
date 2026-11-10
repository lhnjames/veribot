"""Validation-only screen of the two-tail evidence bottleneck on MGTAB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load
from train_v3_computable import attach_mgtab_structural, groups_for, rows
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference, SignConstrainedReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--l2", type=float, default=0.01)
    args = parser.parse_args()
    runs = []
    for seed in args.seeds:
        x, y, ids, names, split, source = load("mgtab", args.data_dir, None, seed)
        x, names, source = attach_mgtab_structural(
            x, names, ids, source, args.data_dir, args.structural_facts)
        reference = LabelBlindReference(min_context_n=30).fit(x[split["train"]], names)
        evidence = reference.transform(x)
        a, b = train_test_split(split["train"], test_size=0.5,
                                stratify=y[split["train"]], random_state=seed)
        groups = groups_for("mgtab", names)
        result = {}
        for name, model in (("single_tail", SignConstrainedReasoner(groups, l2=args.l2)),
                            ("dual_tail", DualTailEvidenceReasoner(groups, l2=args.l2))):
            model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
            probability = model.predict_proba(rows(evidence, split["val"]))
            threshold = best_validation_threshold(probability, y[split["val"]])
            result[name] = {
                "validation": evaluate(y[split["val"]], probability, threshold),
                "threshold": threshold,
                "nonzero_coefficients": int(np.count_nonzero(model.coef > 0)),
                "reasoner": model.manifest(),
            }
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "reference_version": reference.version, **result})
    output = {
        "dataset": "mgtab", "scope": "validation_only",
        "test_predictions_accessed": False,
        "structural_facts": str(args.structural_facts.resolve()),
        "structural_facts_sha256": source["files_sha256"][str(args.structural_facts.resolve())],
        "l2": args.l2,
        "model_selection": "compare paired validation Macro-F1 and ACC across seeds; no test access",
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": output["scope"], "runs": [
        {"seed": row["seed"],
         "single_tail": row["single_tail"]["validation"],
         "dual_tail": row["dual_tail"]["validation"]}
        for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
