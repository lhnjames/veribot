"""Dev-only label-leakage diagnostic for MGTAB evidence references."""

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
from screen_mgtab_content_projection_evidence import transform_with_support
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def design(reference: LabelBlindReference, matrix: np.ndarray,
           supports: np.ndarray) -> np.ndarray:
    evidence = transform_with_support(reference, matrix, supports)
    return np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                            np.log1p(evidence.lower) * evidence.reliability))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if audit["status"] != "verified_all_published_tensor_derived_fact_cells":
        raise ValueError("published tensor fact audit differs")
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
        matrix = np.column_stack((x, pca.transform(normalized)))
        vocabulary = names + [f"published_tweet_projection_{j:03d}" for j in range(64)]
        human_train = train[y[train] == 0]
        rng = np.random.default_rng(20261103 + seed)
        random_same_size = rng.choice(train, size=len(human_train), replace=False)
        groups = {
            "all_train_label_blind": train,
            "random_train_same_size_label_blind": random_same_size,
            "human_only_label_using_diagnostic": human_train,
        }
        alternatives = {}
        for name, reference_rows in groups.items():
            reference = LabelBlindReference(min_context_n=30).fit(
                matrix[reference_rows], vocabulary)
            values = design(reference, matrix, supports)
            model = LogisticRegression(C=1.0, penalty="l1", solver="liblinear",
                                       max_iter=2000, random_state=seed).fit(
                values[train], y[train])
            p = expit(float(model.intercept_[0]) + values[val] @ model.coef_.ravel())
            threshold = best_validation_threshold(p, y[val])
            alternatives[name] = {
                "reference_accounts": int(len(reference_rows)),
                "reference_uses_bot_human_labels": name == "human_only_label_using_diagnostic",
                "reference_version": reference.version,
                "validation": evaluate(y[val], p, threshold),
                "threshold": threshold,
            }
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "alternatives": alternatives})
    result = {"dataset": "mgtab", "scope": "validation_only_leakage_diagnostic",
              "test_predictions_accessed": False,
              "source_features_sha256": sha256_file(args.data_dir / "features.pt"),
              "structural_facts_sha256": sha256_file(args.structural_facts),
              "neighbor_facts_sha256": sha256_file(args.neighbor_facts),
              "source_audit_sha256": sha256_file(args.source_audit),
              "selected_reference": "all_train_label_blind",
              "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{ "seed": row["seed"], "validation": {
        key: value["validation"] for key, value in row["alternatives"].items()}}
        for row in runs], indent=2))


if __name__ == "__main__":
    main()
