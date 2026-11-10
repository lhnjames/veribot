"""Evaluate validation-locked Fox8 spline strength using source-audited facts."""

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
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    native_manifest = json.loads(args.native_facts.with_suffix(".json").read_text(
        encoding="utf-8"))
    if (lock["selected_model"] != "spline_43_native"
            or lock["candidate_test_predictions_accessed"] is not False
            or lock["screen_sha256"] != sha256_file(args.screen)
            or lock["base_feature_sha256"] != sha256_file(args.data_dir / "features.csv")
            or lock["native_facts_sha256"] != sha256_file(args.native_facts)
            or lock["native_source_audit_sha256"] != sha256_file(args.source_audit)
            or screen["scope"] != "validation_only_native_ab_spline_l2"
            or screen["test_predictions_accessed"] is not False
            or audit["status"] != "verified_all_native_interaction_fact_cells_and_event_locators"
            or audit["fact_cache_sha256"] != native_manifest["facts_sha256"]):
        raise ValueError("Fox8 selection or native source audit differs")
    results = []
    for seed in (0, 1, 2):
        x, y, ids, names, split, source = load("fox8", args.data_dir, None, seed)
        with np.load(args.native_facts, allow_pickle=False) as cache:
            if cache["ids"].astype(str).tolist() != ids:
                raise ValueError("native interaction account order differs")
            added = cache["values"].astype(float)
            added_names = cache["names"].astype(str).tolist()
        if source["split_sha256"] != screen["runs"][seed]["split_sha256"]:
            raise ValueError("Fox8 split differs from dev selection")
        matrix = np.column_stack((x, added))
        vocabulary = names + added_names
        groups = {**groups_for("fox8", names), "native_interaction": added_names}
        train = split["train"]
        reference = LabelBlindReference(min_context_n=30).fit(
            matrix[train], vocabulary)
        evidence = reference.transform(matrix)
        a, b = train_test_split(train, test_size=0.5, stratify=y[train],
                                random_state=seed)
        selected_l2 = float(lock["selected_l2"])
        model = MonotoneSplineReasoner(groups, l2=selected_l2)
        model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
        dev_probability = model.predict_proba(rows(evidence, split["val"]))
        threshold = best_validation_threshold(dev_probability, y[split["val"]])
        dev_metric = evaluate(y[split["val"]], dev_probability, threshold)
        selected = screen["runs"][seed]["alternatives"][str(selected_l2)]
        if (reference.version != screen["runs"][seed]["reference_version"]
                or not np.isclose(threshold, selected["threshold"], rtol=0, atol=1e-12)
                or any(not np.isclose(dev_metric[key], selected["validation"][key],
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
            **source, "model": "computable_evidence_spline_with_native_interaction",
            "raw_source_sha256": native_manifest["raw_source_sha256"],
            "native_facts_sha256": sha256_file(args.native_facts),
            "native_event_index_sha256": native_manifest["event_index_sha256"],
            "native_source_audit_sha256": sha256_file(args.source_audit),
            "screen_sha256": lock["screen_sha256"],
            "selection_lock_sha256": sha256_file(args.selection_lock),
            "reference_version": reference.version,
            "selected_l2": selected_l2,
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
