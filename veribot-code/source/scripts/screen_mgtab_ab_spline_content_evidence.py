"""Dev-only A/B sign-constrained spline over audited MGTAB facts and tensor projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_content_projection_evidence import fingerprint, transform_with_support
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural, groups_for, rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l2", type=float, nargs="+", default=(0.001, 0.01, 0.1))
    args = parser.parse_args()
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if audit["status"] != "verified_all_published_tensor_derived_fact_cells":
        raise ValueError("published source audit differs")
    published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    content = published[:, 20:]
    runs = []
    for seed in (0, 1, 2):
        x, y, ids, names, split, source = load("mgtab", args.data_dir, None, seed)
        x, names, source = attach_mgtab_structural(
            x, names, ids, source, args.data_dir, args.structural_facts)
        x, names, supports, source = attach_mgtab_neighbor(
            x, names, ids, source, args.data_dir, args.neighbor_facts)
        train, val = split["train"], split["val"]
        scaler = StandardScaler().fit(content[train])
        normalized = scaler.transform(content)
        pca = PCA(n_components=64, svd_solver="randomized", random_state=seed).fit(
            normalized[train])
        added = [f"published_tweet_projection_{j:03d}" for j in range(64)]
        matrix = np.column_stack((x, pca.transform(normalized)))
        vocabulary = names + added
        groups = {**groups_for("mgtab", names), "published_tweet_projections": added}
        reference = LabelBlindReference(min_context_n=30).fit(
            matrix[train], vocabulary)
        evidence = transform_with_support(reference, matrix, supports)
        a, b = train_test_split(train, test_size=0.5, stratify=y[train],
                                random_state=seed)
        alternatives = {}
        for strength in args.l2:
            model = MonotoneSplineReasoner(groups, l2=strength)
            model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
            p = model.predict_proba(rows(evidence, val))
            threshold = best_validation_threshold(p, y[val])
            alternatives[str(strength)] = {
                "nonnegative_coefficient_count": len(model.coef),
                "nonzero_coefficients": int(np.count_nonzero(model.coef > 0)),
                "validation": evaluate(y[val], p, threshold),
                "threshold": threshold}
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "reference_version": reference.version,
                     "projection_fingerprint": fingerprint(
                         scaler.mean_, scaler.scale_, pca.mean_, pca.components_),
                     "direction_split_size": len(a), "coefficient_split_size": len(b),
                     "alternatives": alternatives})
    result = {"dataset": "mgtab", "scope": "validation_only_ab_monotone_spline",
              "test_predictions_accessed": False,
              "source_features_sha256": sha256_file(args.data_dir / "features.pt"),
              "structural_facts_sha256": sha256_file(args.structural_facts),
              "neighbor_facts_sha256": sha256_file(args.neighbor_facts),
              "source_audit_sha256": sha256_file(args.source_audit),
              "reference_uses_labels": False,
              "direction_and_coefficient_train_splits_disjoint": True,
              "l2_candidates": args.l2,
              "representation_scope": "published tensor statistical content coordinates; not semantic tweet evidence",
              "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
