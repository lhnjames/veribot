"""Evaluate dev-locked sparse MGTAB evidence against official test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_content_projection_evidence import fingerprint, transform_with_support
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural, groups_for
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if (lock["selected_model"] != "l1_C1"
            or lock["candidate_test_predictions_accessed"] is not False
            or screen["test_predictions_accessed"] is not False
            or lock["screen_sha256"] != sha256_file(args.screen)
            or lock["source_audit_sha256"] != sha256_file(args.source_audit)
            or lock["structural_facts_sha256"] != sha256_file(args.structural_facts)
            or lock["neighbor_facts_sha256"] != sha256_file(args.neighbor_facts)
            or lock["source_features_sha256"] != sha256_file(
                args.data_dir / "features.pt")
            or audit["status"] != "verified_all_published_tensor_derived_fact_cells"):
        raise ValueError("dev lock or tensor source audit differs")
    published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    content = published[:, 20:]
    results = []
    for seed in (0, 1, 2):
        x, y, ids, names, split, source = load(
            "mgtab", args.data_dir, None, seed)
        x, names, source = attach_mgtab_structural(
            x, names, ids, source, args.data_dir, args.structural_facts)
        x, names, supports, source = attach_mgtab_neighbor(
            x, names, ids, source, args.data_dir, args.neighbor_facts)
        if x.shape != (10199, 86) or supports is None:
            raise ValueError("audited MGTAB fact table differs")
        train, val, test = split["train"], split["val"], split["test"]
        scaler = StandardScaler().fit(content[train])
        normalized = scaler.transform(content)
        pca = PCA(n_components=64, svd_solver="randomized",
                  random_state=seed).fit(normalized[train])
        projected = pca.transform(normalized)
        added = [f"published_tweet_projection_{j:03d}" for j in range(64)]
        matrix = np.column_stack((x, projected))
        vocabulary = names + added
        reference = LabelBlindReference(min_context_n=30).fit(
            matrix[train], vocabulary)
        evidence = transform_with_support(reference, matrix, supports)
        design = np.column_stack((
            np.log1p(evidence.upper) * evidence.reliability,
            np.log1p(evidence.lower) * evidence.reliability))
        model = LogisticRegression(C=1.0, penalty="l1",
                                   solver="liblinear", max_iter=2000,
                                   random_state=seed)
        model.fit(design[train], y[train])
        coefficient = model.coef_.ravel().astype(float)
        intercept = float(model.intercept_[0])
        dev_p = expit(intercept + design[val] @ coefficient)
        threshold = best_validation_threshold(dev_p, y[val])
        dev_metric = evaluate(y[val], dev_p, threshold)
        selected = screen["runs"][seed]["l1_C1"]
        projected_fingerprint = fingerprint(
            scaler.mean_, scaler.scale_, pca.mean_, pca.components_)
        if (source["split_sha256"] != screen["runs"][seed]["split_sha256"]
                or projected_fingerprint !=
                screen["runs"][seed]["projection_fingerprint"]
                or reference.version !=
                screen["runs"][seed]["reference_version"]
                or abs(threshold - selected["threshold"]) > 1e-10
                or any(abs(dev_metric[key] - selected["validation"][key]) > 1e-10
                       for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("fixed dev model differs from selection")
        out = args.output / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        reference_path = out / "reference.json.gz"
        reference.save(reference_path)
        artifact_path = out / "model.npz"
        np.savez_compressed(
            artifact_path, vocabulary=np.asarray(vocabulary),
            coefficients=coefficient, intercept=np.asarray([intercept]),
            threshold=np.asarray([threshold]), train_indices=train,
            scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
            pca_mean=pca.mean_, pca_components=pca.components_)
        test_design = design[test]
        test_logit = intercept + test_design @ coefficient
        probability = expit(test_logit)
        groups = {**groups_for("mgtab", names),
                  "published_tweet_projections": added}
        by_group = {}
        for group, members in groups.items():
            indices = np.asarray([vocabulary.index(name) for name in members])
            by_group[group] = (test_design[:, indices] @ coefficient[indices] +
                               test_design[:, indices + 150] @
                               coefficient[indices + 150])
        reconstructed = intercept + sum(by_group.values())
        if np.max(np.abs(reconstructed - test_logit)) > 1e-10:
            raise ValueError("evidence groups do not reconstruct test logits")
        manifest = {
            **source, "model": "sparse_additive_tensor_evidence",
            "source_features_sha256": sha256_file(args.data_dir / "features.pt"),
            "structural_facts_sha256": sha256_file(args.structural_facts),
            "neighbor_facts_sha256": sha256_file(args.neighbor_facts),
            "source_audit_sha256": sha256_file(args.source_audit),
            "selection_lock_sha256": sha256_file(args.selection_lock),
            "model_artifact_sha256": sha256_file(artifact_path),
            "reference_artifact_sha256": sha256_file(reference_path),
            "reference_version": reference.version,
            "reference_used_bot_human_labels": False,
            "projection_fingerprint": projected_fingerprint,
            "projection_fit": "train_only_label_blind",
            "projection_source_level": "published_averaged_tweet_tensor",
            "threshold": threshold,
            "threshold_selected_on": "validation_macro_f1",
            "active_evidence_terms": int(np.count_nonzero(coefficient)),
        }
        report = {"manifest": manifest, "dev": dev_metric,
                  "test": evaluate(y[test], probability, threshold),
                  "reasoner": {"type": "l1_sparse_additive_evidence_only",
                               "intercept": intercept, "groups": groups}}
        (out / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (out / "test_predictions.json").write_text(
            json.dumps({"manifest": manifest, "predictions": [
                {"account_id": ids[int(i)], "label": int(y[i]),
                 "probability": float(probability[k]),
                 "group_contributions": {name: float(values[k])
                                         for name, values in by_group.items()}}
                for k, i in enumerate(test)]}, indent=2) + "\n",
            encoding="utf-8")
        results.append({"seed": seed, "dev": dev_metric,
                        "test": report["test"]})
    print(json.dumps({"selected_model": lock["selected_model"],
                      "runs": results}, indent=2))


if __name__ == "__main__":
    main()
