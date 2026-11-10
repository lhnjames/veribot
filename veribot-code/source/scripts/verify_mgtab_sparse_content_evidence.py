"""Replay every frozen MGTAB prediction from the published tensor and evidence artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from screen_mgtab_content_projection_evidence import fingerprint, transform_with_support
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural, groups_for
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def verify(args: argparse.Namespace, seed: int) -> dict:
    folder = args.model_dir / f"seed{seed}"
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    saved = json.loads((folder / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = report["manifest"]
    if manifest != saved["manifest"] or manifest["dataset"] != "mgtab" or manifest["seed"] != seed:
        raise ValueError("saved manifest differs")
    for path, key in ((args.data_dir / "features.pt", "source_features_sha256"),
                      (args.structural_facts, "structural_facts_sha256"),
                      (args.neighbor_facts, "neighbor_facts_sha256"),
                      (args.source_audit, "source_audit_sha256"),
                      (args.selection_lock, "selection_lock_sha256"),
                      (folder / "model.npz", "model_artifact_sha256"),
                      (folder / "reference.json.gz", "reference_artifact_sha256")):
        if sha256_file(path) != manifest[key]:
            raise ValueError(f"source or artifact hash differs: {path}")
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if audit["status"] != "verified_all_published_tensor_derived_fact_cells":
        raise ValueError("published tensor source audit differs")
    x, y, ids, names, split, current = load("mgtab", args.data_dir, None, seed)
    x, names, current = attach_mgtab_structural(
        x, names, ids, current, args.data_dir, args.structural_facts)
    x, names, supports, current = attach_mgtab_neighbor(
        x, names, ids, current, args.data_dir, args.neighbor_facts)
    if (current["files_sha256"] != manifest["files_sha256"]
            or current["split_sha256"] != manifest["split_sha256"]
            or len(names) != 86):
        raise ValueError("published source, fact table, or split differs")
    with np.load(folder / "model.npz", allow_pickle=False) as artifact:
        vocabulary = artifact["vocabulary"].astype(str).tolist()
        coefficients = artifact["coefficients"].astype(float)
        intercept = float(artifact["intercept"][0])
        threshold = float(artifact["threshold"][0])
        train_indices = artifact["train_indices"]
        parameters = tuple(artifact[key] for key in
                           ("scaler_mean", "scaler_scale", "pca_mean", "pca_components"))
    train, test = split["train"], split["test"]
    if (not np.array_equal(train, train_indices) or len(vocabulary) != 150
            or vocabulary[:86] != names
            or vocabulary[86:] != [f"published_tweet_projection_{j:03d}" for j in range(64)]
            or coefficients.shape != (300,) or threshold != manifest["threshold"]):
        raise ValueError("model vocabulary, split, or dimensions differ")
    published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    content = published[:, 20:]
    scaler = StandardScaler().fit(content[train])
    normalized = scaler.transform(content)
    pca = PCA(n_components=64, svd_solver="randomized", random_state=seed).fit(
        normalized[train])
    refit_parameters = (scaler.mean_, scaler.scale_, pca.mean_, pca.components_)
    if (fingerprint(*refit_parameters) != manifest["projection_fingerprint"]
            or any(not np.array_equal(a, b) for a, b in zip(parameters, refit_parameters))):
        raise ValueError("projection does not refit from training source")
    matrix = np.column_stack((x, pca.transform(normalized)))
    reference = LabelBlindReference.load(folder / "reference.json.gz")
    refit_reference = LabelBlindReference(min_context_n=30).fit(
        matrix[train], vocabulary)
    if (reference.version != refit_reference.version
            or reference.version != manifest["reference_version"]
            or reference.names != tuple(vocabulary)):
        raise ValueError("label-blind reference does not refit")
    evidence = transform_with_support(reference, matrix[test], supports[test])
    design = np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                              np.log1p(evidence.lower) * evidence.reliability))
    logits = intercept + design @ coefficients
    probabilities = expit(logits)
    rows = saved["predictions"]
    if len(rows) != len(test) or any(
            row["account_id"] != ids[int(index)] or row["label"] != int(y[index])
            for row, index in zip(rows, test)):
        raise ValueError("test account order or labels differ")
    probability_difference = float(np.max(np.abs(
        probabilities - np.asarray([row["probability"] for row in rows]))))
    if probability_difference > 1e-12:
        raise ValueError("saved probabilities do not replay")
    groups = {**groups_for("mgtab", names),
              "published_tweet_projections": vocabulary[86:]}
    group_values = {}
    for group, members in groups.items():
        indices = np.asarray([vocabulary.index(name) for name in members])
        group_values[group] = (design[:, indices] @ coefficients[indices]
                               + design[:, indices + 150] @ coefficients[indices + 150])
    group_difference = float(max(np.max(np.abs(values - np.asarray(
        [row["group_contributions"][name] for row in rows])))
        for name, values in group_values.items()))
    logit_difference = float(np.max(np.abs(
        intercept + sum(group_values.values()) - logits)))
    if group_difference > 1e-12 or logit_difference > 1e-12:
        raise ValueError("evidence groups do not reconstruct saved prediction")
    frozen_predictions = probabilities >= threshold
    group_ablation = {
        name: int(np.count_nonzero((expit(logits - values) >= threshold)
                                    != frozen_predictions))
        for name, values in group_values.items()
    }
    # With every evidence term masked, the only remaining logit is the intercept.
    if abs(intercept + np.zeros_like(design) @ coefficients - intercept).max() > 0:
        raise ValueError("all-evidence mask retained a bypass")
    return {"dataset": "mgtab", "seed": seed, "test_accounts": len(test),
            "status": "verified", "active_evidence_terms": int(np.count_nonzero(coefficients)),
            "max_probability_difference": probability_difference,
            "max_group_difference": group_difference,
            "max_logit_reconstruction_difference": logit_difference,
            "all_evidence_masked_logit": intercept,
            "group_mask_prediction_changes": group_ablation,
            "prediction_errors": int(np.count_nonzero(frozen_predictions != y[test]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"runs": [verify(args, seed) for seed in (0, 1, 2)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
