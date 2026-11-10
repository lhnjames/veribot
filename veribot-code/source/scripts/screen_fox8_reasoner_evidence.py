"""Validation-only comparison of three evidence-only reasoners on Fox8-23."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from train_v3_computable import groups_for, rows
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference, SignConstrainedReasoner
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if audit["dataset"] != "fox8" or audit["all_rows_match"] is not True:
        raise ValueError("Fox8 raw-record feature source audit is absent")
    if audit["files_sha256"][str((args.data_dir / "features.csv").resolve())] != \
            sha256_file(args.data_dir / "features.csv"):
        raise ValueError("Fox8 derived facts differ from raw-record source audit")
    runs = []
    for seed in (0, 1, 2):
        x, y, ids, names, split, source = load("fox8", args.data_dir, None, seed)
        reference = LabelBlindReference(min_context_n=30).fit(x[split["train"]], names)
        evidence = reference.transform(x)
        a, b = train_test_split(split["train"], test_size=0.5,
                                stratify=y[split["train"]], random_state=seed)
        groups = groups_for("fox8", names)
        result = {}
        for label, model in (
            ("single_tail", SignConstrainedReasoner(groups, l2=0.01)),
            ("dual_tail", DualTailEvidenceReasoner(groups, l2=0.01)),
            ("monotone_spline", MonotoneSplineReasoner(groups, l2=0.01)),
        ):
            model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
            probability = model.predict_proba(rows(evidence, split["val"]))
            threshold = best_validation_threshold(probability, y[split["val"]])
            result[label] = {
                "validation": evaluate(y[split["val"]], probability, threshold),
                "threshold": threshold,
                "reference_version": reference.version,
                "nonzero_coefficients": int(np.count_nonzero(model.coef > 0)),
            }
        runs.append({"seed": seed, "split_sha256": source["split_sha256"], **result})
    report = {
        "dataset": "fox8", "scope": "validation_only",
        "test_predictions_accessed": False,
        "raw_feature_audit_sha256": sha256_file(args.source_audit),
        "feature_sha256": sha256_file(args.data_dir / "features.csv"),
        "l2": 0.01,
        "selection_rule": "three-seed mean validation Macro-F1; tie by ACC",
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": report["scope"], "runs": [
        {"seed": row["seed"], **{name: row[name]["validation"]
                                  for name in ("single_tail", "dual_tail",
                                               "monotone_spline")}}
        for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
