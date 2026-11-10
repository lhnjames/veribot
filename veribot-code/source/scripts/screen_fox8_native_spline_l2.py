"""Dev-only regularization screen for Fox8 source-backed A/B spline evidence."""

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
    native_manifest = json.loads(args.native_facts.with_suffix(".json").read_text(
        encoding="utf-8"))
    if (audit["status"] != "verified_all_native_interaction_fact_cells_and_event_locators"
            or audit["fact_cache_sha256"] != sha256_file(args.native_facts)
            or native_manifest["facts_sha256"] != sha256_file(args.native_facts)
            or native_manifest["features_csv_sha256"] !=
            sha256_file(args.data_dir / "features.csv")):
        raise ValueError("Fox8 source audit or cached native facts differ")
    with np.load(args.native_facts, allow_pickle=False) as cache:
        cached_ids = cache["ids"].astype(str).tolist()
        native_values = cache["values"].astype(float)
        native_names = cache["names"].astype(str).tolist()
    strengths = (0.0001, 0.001, 0.01, 0.1)
    runs = []
    for seed in (0, 1, 2):
        x, y, ids, names, split, source = load("fox8", args.data_dir, None, seed)
        if ids != cached_ids:
            raise ValueError("native interaction account order differs")
        matrix = np.column_stack((x, native_values))
        vocabulary = names + native_names
        reference = LabelBlindReference(min_context_n=30).fit(
            matrix[split["train"]], vocabulary)
        evidence = reference.transform(matrix)
        a, b = train_test_split(split["train"], test_size=0.5,
                                stratify=y[split["train"]], random_state=seed)
        groups = {**groups_for("fox8", names), "native_interaction": native_names}
        options = {}
        for strength in strengths:
            model = MonotoneSplineReasoner(groups, l2=strength)
            model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
            p = model.predict_proba(rows(evidence, split["val"]))
            threshold = best_validation_threshold(p, y[split["val"]])
            options[str(strength)] = {
                "validation": evaluate(y[split["val"]], p, threshold),
                "threshold": threshold,
                "nonzero_coefficients": int(np.count_nonzero(model.coef > 0)),
            }
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "reference_version": reference.version,
                     "direction_split_size": len(a), "coefficient_split_size": len(b),
                     "alternatives": options})
    result = {"dataset": "fox8", "scope": "validation_only_native_ab_spline_l2",
              "test_predictions_accessed": False,
              "prior_other_architecture_test_predictions_already_observed": True,
              "source_level": "raw_profile_and_sampled_tweet_interactions",
              "base_feature_sha256": sha256_file(args.data_dir / "features.csv"),
              "native_facts_sha256": sha256_file(args.native_facts),
              "native_source_audit_sha256": sha256_file(args.source_audit),
              "l2_candidates": strengths, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{ "seed": row["seed"], "validation": {
        key: value["validation"] for key, value in row["alternatives"].items()}}
        for row in runs], indent=2))


if __name__ == "__main__":
    main()
