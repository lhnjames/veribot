"""Evaluate the dev-locked A/B monotone MGTAB evidence model on fixed test."""

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
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--selected-screen", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    screen = json.loads(args.selected_screen.read_text(encoding="utf-8"))
    if (lock["selected_reasoner"] != "ab_monotone_interval_spline"
            or lock["selected_l2"] != 1e-5
            or lock["candidate_test_predictions_accessed"] is not False
            or lock["screens_sha256"][str(args.selected_screen)] != sha256_file(args.selected_screen)
            or screen["test_predictions_accessed"] is not False
            or lock["source_features_sha256"] != sha256_file(args.data_dir / "features.pt")
            or lock["structural_facts_sha256"] != sha256_file(args.structural_facts)
            or lock["neighbor_facts_sha256"] != sha256_file(args.neighbor_facts)
            or lock["source_audit_sha256"] != sha256_file(args.source_audit)):
        raise ValueError("MGTAB A/B dev lock or source facts differ")
    source_audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if source_audit["status"] != "verified_all_published_tensor_derived_fact_cells":
        raise ValueError("published tensor source audit differs")
    published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    if published.shape != (10199, 788):
        raise ValueError("MGTAB published feature shape differs")
    content = published[:, 20:]
    results = []
    for seed in (0, 1, 2):
        x, y, ids, names, split, source = load("mgtab", args.data_dir, None, seed)
        x, names, source = attach_mgtab_structural(
            x, names, ids, source, args.data_dir, args.structural_facts)
        x, names, supports, source = attach_mgtab_neighbor(
            x, names, ids, source, args.data_dir, args.neighbor_facts)
        if (x.shape != (10199, 86) or supports is None
                or source["split_sha256"] != screen["runs"][seed]["split_sha256"]):
            raise ValueError("MGTAB evidence population or fixed split differs")
        train = split["train"]
        scaler = StandardScaler().fit(content[train])
        normalized = scaler.transform(content)
        pca = PCA(n_components=64, svd_solver="randomized",
                  random_state=seed).fit(normalized[train])
        fingerprint_now = fingerprint(
            scaler.mean_, scaler.scale_, pca.mean_, pca.components_)
        screened_row = screen["runs"][seed]
        screened = screened_row["alternatives"][str(lock["selected_l2"])]
        if fingerprint_now != screened_row["projection_fingerprint"]:
            raise ValueError("train-only projection differs from dev selection")
        projected = pca.transform(normalized)
        added_names = [f"published_tweet_projection_{j:03d}" for j in range(64)]
        matrix = np.column_stack((x, projected))
        vocabulary = names + added_names
        groups = {**groups_for("mgtab", names),
                  "published_tweet_projections": added_names}
        reference = LabelBlindReference(min_context_n=30).fit(
            matrix[train], vocabulary)
        evidence = transform_with_support(reference, matrix, supports)
        a, b = train_test_split(train, test_size=0.5, stratify=y[train],
                                random_state=seed)
        model = MonotoneSplineReasoner(groups, l2=lock["selected_l2"])
        model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
        dev_probability = model.predict_proba(rows(evidence, split["val"]))
        threshold = best_validation_threshold(dev_probability, y[split["val"]])
        dev_metric = evaluate(y[split["val"]], dev_probability, threshold)
        if (reference.version != screened_row["reference_version"]
                or not np.isclose(threshold, screened["threshold"], rtol=0, atol=1e-12)
                or any(not np.isclose(dev_metric[key], screened["validation"][key],
                                       rtol=0, atol=1e-12)
                       for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("frozen dev reasoner differs from selection")
        test = split["test"]
        test_evidence = rows(evidence, test)
        probability = model.predict_proba(test_evidence)
        contributions = model.contributions(test_evidence)
        out = args.output / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        projection_path = out / "projection.npz"
        np.savez_compressed(projection_path, scaler_mean=scaler.mean_,
                            scaler_scale=scaler.scale_, pca_mean=pca.mean_,
                            pca_components=pca.components_,
                            train_indices=np.asarray(train),
                            feature_names=np.asarray(added_names))
        reference_path = out / "reference.json.gz"
        reference.save(reference_path)
        manifest = {
            **source, "model": "ab_monotone_spline_with_content_projection",
            "content_source_level": "published_averaged_tweet_vector_tensor",
            "source_features_sha256": lock["source_features_sha256"],
            "structural_facts_sha256": lock["structural_facts_sha256"],
            "neighbor_facts_sha256": lock["neighbor_facts_sha256"],
            "projection_artifact_sha256": sha256_file(projection_path),
            "projection_fingerprint": fingerprint_now,
            "projection_fit": "train-only, label-blind standardization and randomized PCA",
            "projection_semantics": "statistical content coordinates; no raw-tweet semantic ground truth",
            "screen_sha256": sha256_file(args.selected_screen),
            "source_audit_sha256": sha256_file(args.source_audit),
            "selection_lock_sha256": sha256_file(args.selection_lock),
            "reference_version": reference.version,
            "reference_used_bot_human_labels": False,
            "reference_artifact_sha256": sha256_file(reference_path),
            "direction_split_size": len(a),
            "coefficient_split_size": len(b),
            "direction_and_coefficient_splits_disjoint": True,
            "selected_l2": lock["selected_l2"],
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
    print(json.dumps({"selected_reasoner": lock["selected_reasoner"],
                      "runs": results}, indent=2))


if __name__ == "__main__":
    main()
