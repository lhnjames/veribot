"""Dev-only sparse reasoner over 86 audited facts and 64 tensor projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_content_projection_evidence import fingerprint, transform_with_support
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if source_audit["status"] != "verified_all_published_tensor_derived_fact_cells":
        raise ValueError("MGTAB published tensor audit differs")
    published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    if published.shape != (10199, 788):
        raise ValueError("MGTAB published feature tensor differs")
    content = published[:, 20:]
    runs = []
    for seed in (0, 1, 2):
        x, y, ids, names, split, source = load(
            "mgtab", args.data_dir, None, seed)
        x, names, source = attach_mgtab_structural(
            x, names, ids, source, args.data_dir, args.structural_facts)
        x, names, supports, source = attach_mgtab_neighbor(
            x, names, ids, source, args.data_dir, args.neighbor_facts)
        if x.shape != (10199, 86) or supports is None:
            raise ValueError("86 source-backed MGTAB facts differ")
        train, val = split["train"], split["val"]
        scaler = StandardScaler().fit(content[train])
        normalized = scaler.transform(content)
        pca = PCA(n_components=64, svd_solver="randomized",
                  random_state=seed).fit(normalized[train])
        projected = pca.transform(normalized)
        added_names = [f"published_tweet_projection_{j:03d}"
                       for j in range(64)]
        matrix = np.column_stack((x, projected))
        vocabulary = names + added_names
        reference = LabelBlindReference(min_context_n=30).fit(
            matrix[train], vocabulary)
        evidence = transform_with_support(reference, matrix, supports)
        design = np.column_stack((
            np.log1p(evidence.upper) * evidence.reliability,
            np.log1p(evidence.lower) * evidence.reliability))
        outputs = {}
        for penalty, strength in (("l1", 0.1), ("l1", 1.0),
                                  ("l1", 10.0), ("l2", 0.1),
                                  ("l2", 1.0)):
            model = LogisticRegression(C=strength, penalty=penalty,
                                       solver="liblinear", max_iter=2000,
                                       random_state=seed)
            model.fit(design[train], y[train])
            probability = model.predict_proba(design[val])[:, 1]
            threshold = best_validation_threshold(probability, y[val])
            outputs[f"{penalty}_C{strength:g}"] = {
                "validation": evaluate(y[val], probability, threshold),
                "threshold": threshold,
                "active_evidence_terms": int(np.count_nonzero(model.coef_)),
                "active_projection_terms": int(np.count_nonzero(
                    np.r_[model.coef_[0, 86:150], model.coef_[0, 236:300]])),
            }
        runs.append({"seed": seed,
                     "split_sha256": source["split_sha256"],
                     "reference_version": reference.version,
                     "projection_fingerprint": fingerprint(
                         scaler.mean_, scaler.scale_, pca.mean_, pca.components_),
                     **outputs})
    report = {
        "dataset": "mgtab", "scope": "validation_only",
        "test_predictions_accessed": False,
        "source_level": "published_feature_and_typed_edge_tensors",
        "source_audit_sha256": sha256_file(args.source_audit),
        "source_features_sha256": sha256_file(args.data_dir / "features.pt"),
        "structural_facts_sha256": sha256_file(args.structural_facts),
        "neighbor_facts_sha256": sha256_file(args.neighbor_facts),
        "fact_count": 150, "evidence_term_count": 300,
        "reference_used_labels": False,
        "projection_fit": "train-only label-blind standardization and PCA",
        "projection_semantics": "numeric coordinates traceable to published averaged-tweet tensor; no original tweet-level semantic claim",
        "model_input": "only upper/lower tail evidence; no raw tensor bypass",
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"scope": report["scope"], "runs": [
        {"seed": run["seed"], **{key: value["validation"] for key, value in
                                 run.items() if key.startswith(("l1_", "l2_"))}}
        for run in runs]}, indent=2))


if __name__ == "__main__":
    main()
