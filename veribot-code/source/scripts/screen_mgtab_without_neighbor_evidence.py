"""Dev-only control omitting MGTAB's 24 occasionally unavailable neighbor facts."""

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
from screen_mgtab_content_projection_evidence import fingerprint
from train_v3_computable import attach_mgtab_structural, groups_for, rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--selected-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if (audit.get("status") != "verified_all_published_tensor_derived_fact_cells"
            or audit["structural_facts_sha256"] != sha256_file(args.structural_facts)
            or any(audit["source_sha256"][name] != sha256_file(args.data_dir / name)
                   for name in ("features.pt", "edge_index.pt", "edge_type.pt"))):
        raise ValueError("MGTAB source audit differs")
    published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    if published.shape != (10199, 788) or not np.isfinite(published).all():
        raise ValueError("published content tensor differs")
    content = published[:, 20:]
    runs = []
    for seed in (0, 1, 2):
        selected = json.loads((args.selected_model_dir / f"seed{seed}" / "report.json")
                              .read_text(encoding="utf-8"))
        x, y, ids, names, split, source = load("mgtab", args.data_dir, None, seed)
        x, names, source = attach_mgtab_structural(
            x, names, ids, source, args.data_dir, args.structural_facts)
        if (len(names) != 62 or selected["manifest"]["split_sha256"] != source["split_sha256"]
                or selected["manifest"]["source_features_sha256"] !=
                sha256_file(args.data_dir / "features.pt")
                or selected["manifest"]["structural_facts_sha256"] !=
                sha256_file(args.structural_facts)
                or selected["manifest"]["source_audit_sha256"] !=
                sha256_file(args.source_audit)):
            raise ValueError("control and selected model source or split differ")
        train, dev = split["train"], split["val"]
        scaler = StandardScaler().fit(content[train])
        normalized = scaler.transform(content)
        pca = PCA(n_components=64, svd_solver="randomized", random_state=seed).fit(
            normalized[train])
        projection_fingerprint = fingerprint(
            scaler.mean_, scaler.scale_, pca.mean_, pca.components_)
        if projection_fingerprint != selected["manifest"]["projection_fingerprint"]:
            raise ValueError("train-only content projection differs")
        added = [f"published_tweet_projection_{j:03d}" for j in range(64)]
        matrix = np.column_stack((x, pca.transform(normalized)))
        vocabulary = names + added
        groups = {**groups_for("mgtab", names), "published_tweet_projections": added}
        reference = LabelBlindReference(min_context_n=30).fit(
            matrix[train], vocabulary)
        evidence = reference.transform(matrix)
        a, b = train_test_split(train, test_size=0.5, stratify=y[train],
                                random_state=seed)
        strength = float(selected["manifest"]["selected_l2"])
        model = MonotoneSplineReasoner(groups, l2=strength)
        model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
        probability = model.predict_proba(rows(evidence, dev))
        threshold = best_validation_threshold(probability, y[dev])
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "projection_fingerprint": projection_fingerprint,
            "l2": strength, "retained_fact_count": len(vocabulary),
            "omitted_neighbor_fact_count": 24,
            "without_neighbor_validation": evaluate(y[dev], probability, threshold),
            "with_neighbor_validation": selected["validation"],
        })
    result = {"dataset": "mgtab", "scope": "validation_only_neighbor_fact_ablation",
              "test_predictions_accessed": False,
              "source_audit_sha256": sha256_file(args.source_audit),
              "selected_model_dir": str(args.selected_model_dir), "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
