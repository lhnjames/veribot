"""Dev-only plan fallback: source-evidence spline plus low-capacity bounded content residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural, groups_for, rows
from verifiable_bot.mgtab_neighbor_facts import evidence_with_neighbor_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def fit_residual(base_logit: np.ndarray, residual: np.ndarray,
                 labels: np.ndarray, bound: float) -> np.ndarray:
    y = np.asarray(labels, dtype=float)

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = base_logit + residual @ weights
        loss = float(np.mean(np.logaddexp(0, logits) - y * logits)
                     + 0.005 * np.sum(weights ** 2))
        grad = residual.T @ (expit(logits) - y) / len(y) + 0.01 * weights
        return loss, grad

    result = minimize(objective, np.zeros(residual.shape[1]), method="L-BFGS-B",
                      jac=True, bounds=[(-bound, bound)] * residual.shape[1],
                      options={"maxiter": 2000, "ftol": 1e-11})
    if not result.success:
        raise RuntimeError(f"bounded residual optimization failed: {result.message}")
    return result.x


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
        reference = LabelBlindReference(min_context_n=30).fit(x[train], names)
        evidence = evidence_with_neighbor_support(reference, x, supports)
        a, b = train_test_split(train, test_size=0.5, stratify=y[train],
                                random_state=seed)
        model = MonotoneSplineReasoner(groups_for("mgtab", names), l2=1e-5)
        model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
        base_b = model.logits(rows(evidence, b))
        base_val = model.logits(rows(evidence, val))
        base_p = expit(base_val)
        base_threshold = best_validation_threshold(base_p, y[val])
        scaler = StandardScaler().fit(content[train])
        normalized = scaler.transform(content)
        pca = PCA(n_components=16, svd_solver="randomized", random_state=seed).fit(
            normalized[train])
        projected = pca.transform(normalized)
        projected /= np.maximum(np.sqrt(pca.explained_variance_), 1e-6)
        bounded = np.tanh(projected)
        candidates = {}
        for dimensions in (8, 16):
            for coefficient_bound in (0.125, 0.25):
                z = bounded[:, :dimensions]
                weights = fit_residual(base_b, z[b], y[b], coefficient_bound)
                residual_val = z[val] @ weights
                p = expit(base_val + residual_val)
                threshold = best_validation_threshold(p, y[val])
                key = f"dimensions_{dimensions}_bound_{coefficient_bound}"
                candidates[key] = {
                    "validation": evaluate(y[val], p, threshold),
                    "threshold": threshold,
                    "max_possible_absolute_residual_logit": dimensions * coefficient_bound,
                    "actual_max_absolute_residual_logit_dev": float(np.max(np.abs(residual_val))),
                    "mean_absolute_residual_logit_dev": float(np.mean(np.abs(residual_val))),
                    "mean_absolute_evidence_logit_dev": float(np.mean(np.abs(base_val - model.intercept))),
                    "coefficients": weights.tolist(),
                }
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "reference_version": reference.version,
                     "base_86_evidence_only": {"validation": evaluate(y[val], base_p, base_threshold),
                                               "threshold": base_threshold},
                     "candidates": candidates})
    result = {"dataset": "mgtab", "scope": "validation_only_bounded_residual_fallback",
              "test_predictions_accessed": False,
              "source_features_sha256": sha256_file(args.data_dir / "features.pt"),
              "structural_facts_sha256": sha256_file(args.structural_facts),
              "neighbor_facts_sha256": sha256_file(args.neighbor_facts),
              "source_audit_sha256": sha256_file(args.source_audit),
              "evidence_reasoner": "A/B fixed-direction nonnegative monotone spline on 86 named tensor facts",
              "residual": "8 or 16 train-only PCA coordinates, tanh bounded to [-1,1], per-coefficient absolute bound",
              "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{"seed": row["seed"],
                       "base": row["base_86_evidence_only"]["validation"],
                       "candidates": {key: value["validation"] for key, value in row["candidates"].items()}}
                      for row in runs], indent=2))


if __name__ == "__main__":
    main()
