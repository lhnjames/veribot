"""Validation-only evidence bottleneck with label-blind content PCA facts."""

from __future__ import annotations

import argparse
import hashlib
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
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural, groups_for, rows
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.mgtab_neighbor_facts import SUPPORT_PRIOR
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference


def transform_with_support(reference: LabelBlindReference, matrix: np.ndarray,
                           supports: np.ndarray) -> EvidenceBatch:
    result = reference.transform(matrix)
    reliability = result.reliability.copy()
    reliability[:, 62:86] *= supports / (supports + SUPPORT_PRIOR)
    return EvidenceBatch(result.upper, result.lower, reliability, result.available,
                         result.reference_version, result.names)


def fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", type=int, nargs="+", default=(16, 64))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    if published.shape != (10199, 788):
        raise ValueError("MGTAB published feature tensor differs")
    content = published[:, 20:]
    runs = []
    for seed in args.seeds:
        x, y, ids, names, split, source = load("mgtab", args.data_dir, None, seed)
        x, names, source = attach_mgtab_structural(
            x, names, ids, source, args.data_dir, args.structural_facts)
        x, names, supports, source = attach_mgtab_neighbor(
            x, names, ids, source, args.data_dir, args.neighbor_facts)
        if x.shape != (10199, 86) or supports is None:
            raise ValueError("MGTAB 86-fact baseline differs")
        train = split["train"]
        a, b = train_test_split(train, test_size=0.5, stratify=y[train],
                                random_state=seed)
        outputs = {}
        for count in (0, *args.components):
            if count:
                scaler = StandardScaler().fit(content[train])
                normalized = scaler.transform(content)
                pca = PCA(n_components=count, svd_solver="randomized",
                          random_state=seed).fit(normalized[train])
                projected = pca.transform(normalized)
                names_added = [f"published_tweet_projection_{j:03d}"
                               for j in range(count)]
                matrix = np.column_stack((x, projected))
                vocabulary = names + names_added
                groups = {**groups_for("mgtab", names),
                          "published_tweet_projections": names_added}
                projection_fingerprint = fingerprint(
                    scaler.mean_, scaler.scale_, pca.mean_, pca.components_)
                retained_variance = float(sum(pca.explained_variance_ratio_))
            else:
                matrix = x
                vocabulary = names
                groups = groups_for("mgtab", names)
                projection_fingerprint = None
                retained_variance = None
            reference = LabelBlindReference(min_context_n=30).fit(
                matrix[train], vocabulary)
            evidence = transform_with_support(reference, matrix, supports)
            model = DualTailEvidenceReasoner(groups, l2=0.01)
            model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
            probability = model.predict_proba(rows(evidence, split["val"]))
            threshold = best_validation_threshold(probability, y[split["val"]])
            outputs[f"components_{count}"] = {
                "validation": evaluate(y[split["val"]], probability, threshold),
                "threshold": threshold,
                "reference_version": reference.version,
                "projection_fingerprint": projection_fingerprint,
                "retained_train_variance": retained_variance,
                "nonzero_coefficients": int(np.count_nonzero(model.coef > 0)),
            }
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     **outputs})
    report = {
        "dataset": "mgtab", "scope": "validation_only",
        "test_predictions_accessed": False,
        "source_features_sha256": sha256_file(args.data_dir / "features.pt"),
        "structural_facts_sha256": sha256_file(args.structural_facts),
        "neighbor_facts_sha256": sha256_file(args.neighbor_facts),
        "content_source_level": "published_averaged_tweet_vector_tensor",
        "projection_fit": "train-only, label-blind standardization and randomized PCA",
        "representation_scope": "statistical content projection, not independently validated tweet semantics",
        "selection_rule": "three-seed mean validation Macro-F1; tie by ACC",
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": report["scope"], "runs": [
        {"seed": row["seed"], **{f"components_{count}": row[f"components_{count}"]["validation"]
                                  for count in (0, *args.components)}}
        for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
