"""Validation-only Fox8 spline screen with audited native interaction facts."""

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
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--native-facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    manifest = json.loads(args.native_facts.with_suffix(".json").read_text(
        encoding="utf-8"))
    if (audit["status"] != "verified_all_native_interaction_fact_cells_and_event_locators"
            or audit["fact_cache_sha256"] != sha256_file(args.native_facts)
            or manifest["facts_sha256"] != sha256_file(args.native_facts)
            or manifest["features_csv_sha256"] !=
            sha256_file(args.data_dir / "features.csv")):
        raise ValueError("Fox8 native interaction source audit differs")
    runs = []
    for seed in (0, 1, 2):
        x, y, ids, names, split, source = load("fox8", args.data_dir, None, seed)
        with np.load(args.native_facts, allow_pickle=False) as cache:
            if cache["ids"].astype(str).tolist() != ids:
                raise ValueError("native interaction account order differs")
            added = cache["values"].astype(float)
            added_names = cache["names"].astype(str).tolist()
        train = split["train"]
        a, b = train_test_split(train, test_size=0.5, stratify=y[train],
                                random_state=seed)
        outputs = {}
        for label, matrix, vocabulary, groups in (
            ("spline_31", x, names, groups_for("fox8", names)),
            ("spline_43_native", np.column_stack((x, added)), names + added_names,
             {**groups_for("fox8", names), "native_interaction": added_names}),
        ):
            reference = LabelBlindReference(min_context_n=30).fit(
                matrix[train], vocabulary)
            evidence = reference.transform(matrix)
            model = MonotoneSplineReasoner(groups, l2=0.01)
            model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
            probability = model.predict_proba(rows(evidence, split["val"]))
            threshold = best_validation_threshold(probability, y[split["val"]])
            outputs[label] = {
                "validation": evaluate(y[split["val"]], probability, threshold),
                "threshold": threshold,
                "reference_version": reference.version,
                "nonzero_coefficients": int(np.count_nonzero(model.coef > 0)),
            }
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     **outputs})
    result = {
        "dataset": "fox8", "scope": "validation_only",
        "test_predictions_accessed": False,
        "base_feature_sha256": sha256_file(args.data_dir / "features.csv"),
        "native_facts_sha256": sha256_file(args.native_facts),
        "native_source_audit_sha256": sha256_file(args.source_audit),
        "selection_rule": "three-seed mean validation Macro-F1; tie by ACC",
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": result["scope"], "runs": [
        {"seed": row["seed"], "spline_31": row["spline_31"]["validation"],
         "spline_43_native": row["spline_43_native"]["validation"]}
        for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
