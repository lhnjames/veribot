"""Evaluate the dev-locked Fox8 evidence spline on fixed test splits."""

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
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if (lock["selected_model"] != "monotone_spline"
            or lock["candidate_test_predictions_accessed"] is not False
            or lock["screen_sha256"] != sha256_file(args.screen)
            or lock["raw_feature_audit_sha256"] != sha256_file(args.source_audit)
            or lock["feature_sha256"] != sha256_file(args.data_dir / "features.csv")
            or screen["test_predictions_accessed"] is not False
            or audit["all_rows_match"] is not True):
        raise ValueError("Fox8 dev selection or raw-source audit differs")
    results = []
    for seed in (0, 1, 2):
        x, y, ids, names, split, source = load("fox8", args.data_dir, None, seed)
        if source["split_sha256"] != screen["runs"][seed]["split_sha256"]:
            raise ValueError("Fox8 split differs from dev screen")
        reference = LabelBlindReference(min_context_n=30).fit(x[split["train"]], names)
        evidence = reference.transform(x)
        a, b = train_test_split(split["train"], test_size=0.5,
                                stratify=y[split["train"]], random_state=seed)
        model = MonotoneSplineReasoner(groups_for("fox8", names), l2=0.01)
        model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
        dev_probability = model.predict_proba(rows(evidence, split["val"]))
        threshold = best_validation_threshold(dev_probability, y[split["val"]])
        dev_metric = evaluate(y[split["val"]], dev_probability, threshold)
        screened = screen["runs"][seed]["monotone_spline"]
        if (reference.version != screened["reference_version"]
                or not np.isclose(threshold, screened["threshold"], rtol=0, atol=1e-12)
                or any(not np.isclose(dev_metric[key], screened["validation"][key],
                                       rtol=0, atol=1e-12)
                       for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("frozen dev model differs from selection")
        test = split["test"]
        test_evidence = rows(evidence, test)
        probability = model.predict_proba(test_evidence)
        contributions = model.contributions(test_evidence)
        out = args.output / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        reference_path = out / "reference.json.gz"
        reference.save(reference_path)
        manifest = {
            **source,
            "model": "computable_evidence_monotone_spline",
            "raw_source_audit_sha256": sha256_file(args.source_audit),
            "screen_sha256": lock["screen_sha256"],
            "selection_lock_sha256": sha256_file(args.selection_lock),
            "reference_version": reference.version,
            "reference_used_bot_human_labels": False,
            "reference_artifact_sha256": sha256_file(reference_path),
            "direction_split_size": len(a),
            "coefficient_split_size": len(b),
            "threshold": threshold,
            "threshold_selected_on": "validation_macro_f1",
        }
        report = {"manifest": manifest, "reasoner": model.manifest(),
                  "validation": dev_metric,
                  "test": evaluate(y[test], probability, threshold)}
        (out / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (out / "test_predictions.json").write_text(json.dumps({
            "manifest": manifest,
            "predictions": [
                {"account_id": ids[int(i)], "label": int(y[i]),
                 "probability": float(probability[k]), "split": "test",
                 "group_contributions": {name: float(values[k])
                                         for name, values in contributions.items()}}
                for k, i in enumerate(test)]}, indent=2) + "\n", encoding="utf-8")
        results.append({"seed": seed, "validation": dev_metric,
                        "test": report["test"]})
    print(json.dumps({"dataset": "fox8", "selected_model": lock["selected_model"],
                      "runs": results}, indent=2))


if __name__ == "__main__":
    main()
