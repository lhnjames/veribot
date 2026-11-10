"""Dev-only test of two prespecified, stronger explicit MGTAB residual caps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_bounded_content_residual import fit_residual
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural, groups_for, rows
from verifiable_bot.mgtab_neighbor_facts import evidence_with_neighbor_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    data = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    structural = root / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = root / "data/processed/mgtab_v3/neighbor_facts.npz"
    audit_path = root / "research/certificates/mgtab_full_source_audit.json"
    previous_path = root / "output/v3_mgtab_bounded_residual/validation_only.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit["status"] != "verified_all_published_tensor_derived_fact_cells"
            or previous["source_features_sha256"] != sha256_file(data / "features.pt")
            or previous["structural_facts_sha256"] != sha256_file(structural)
            or previous["neighbor_facts_sha256"] != sha256_file(neighbor)
            or previous["source_audit_sha256"] != sha256_file(audit_path)):
        raise ValueError("published sources or prior residual control differ")
    published = torch.load(data / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    content = published[:, 20:]
    runs = []
    for seed in (0, 1, 2):
        x, y, ids, names, split, source = load("mgtab", data, None, seed)
        x, names, source = attach_mgtab_structural(
            x, names, ids, source, data, structural)
        x, names, supports, source = attach_mgtab_neighbor(
            x, names, ids, source, data, neighbor)
        train, val = split["train"], split["val"]
        if source["split_sha256"] != previous["runs"][seed]["split_sha256"]:
            raise ValueError("fixed MGTAB split differs")
        reference = LabelBlindReference(min_context_n=30).fit(x[train], names)
        if reference.version != previous["runs"][seed]["reference_version"]:
            raise ValueError("label-blind reference differs")
        evidence = evidence_with_neighbor_support(reference, x, supports)
        a, b = train_test_split(train, test_size=0.5, stratify=y[train],
                                random_state=seed)
        model = MonotoneSplineReasoner(groups_for("mgtab", names), l2=1e-5)
        model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
        base_b = model.logits(rows(evidence, b))
        base_val = model.logits(rows(evidence, val))
        base_probability = expit(base_val)
        base_threshold = best_validation_threshold(base_probability, y[val])
        base_result = evaluate(y[val], base_probability, base_threshold)
        prior_base = previous["runs"][seed]["base_86_evidence_only"]["validation"]
        if any(abs(base_result[key] - prior_base[key]) > 1e-9
               for key in ("acc", "macro_f1", "auroc")):
            raise ValueError("86-fact A/B evidence control did not replay")
        scaler = StandardScaler().fit(content[train])
        normalized = scaler.transform(content)
        candidates = {}
        for dimensions, bound in ((16, 0.25), (16, 0.5), (32, 0.25)):
            if dimensions not in (16, 32):
                raise ValueError("unexpected residual dimension")
            # Refit on train only for each dimension; the 16-D setting must
            # exactly reproduce the previous bounded-residual experiment.
            pca = PCA(n_components=dimensions, svd_solver="randomized",
                      random_state=seed).fit(normalized[train])
            projected = pca.transform(normalized)
            projected /= np.maximum(np.sqrt(pca.explained_variance_), 1e-6)
            z = np.tanh(projected)
            weights = fit_residual(base_b, z[b], y[b], bound)
            residual_val = z[val] @ weights
            probability = expit(base_val + residual_val)
            threshold = best_validation_threshold(probability, y[val])
            key = f"dimensions_{dimensions}_bound_{bound}"
            result = evaluate(y[val], probability, threshold)
            candidates[key] = {
                "validation": result,
                "threshold": threshold,
                "theoretical_absolute_logit_cap": dimensions * bound,
                "observed_max_absolute_residual_logit_dev": float(
                    np.max(np.abs(residual_val))),
                "mean_absolute_residual_logit_dev": float(
                    np.mean(np.abs(residual_val))),
                "mean_absolute_evidence_logit_dev": float(
                    np.mean(np.abs(base_val - model.intercept))),
                "residual_changes_evidence_only_class_at_candidate_threshold": int(
                    np.count_nonzero((base_val + residual_val >=
                                      np.log(threshold / (1 - threshold))) !=
                                     (base_val >= np.log(threshold / (1 - threshold))))),
                "coefficients": weights.tolist(),
            }
            if (dimensions, bound) == (16, 0.25):
                old = previous["runs"][seed]["candidates"][key]
                if (abs(threshold - old["threshold"]) > 1e-9 or
                        any(abs(result[name] - old["validation"][name]) > 1e-9
                            for name in ("acc", "macro_f1", "auroc"))):
                    raise ValueError("prior 16-dimensional residual did not replay")
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "reference_version": reference.version,
                     "base_86_evidence_only": {"validation": base_result,
                                               "threshold": base_threshold},
                     "candidates": candidates})
        print(seed, {name: row["validation"]["acc"]
                     for name, row in candidates.items()}, flush=True)
    result = {
        "dataset": "mgtab", "scope": "official_train_dev_only_stronger_bounded_content_residual",
        "test_labels_loaded_by_full_dataset_adapter": True,
        "test_labels_used_for_fit_or_selection": False,
        "test_predictions_accessed": False,
        "source_features_sha256": sha256_file(data / "features.pt"),
        "structural_facts_sha256": sha256_file(structural),
        "neighbor_facts_sha256": sha256_file(neighbor),
        "source_audit_sha256": sha256_file(audit_path),
        "prior_control_sha256": sha256_file(previous_path),
        "residual_source": "published_average_tweet_tensor_train_only_unsupervised_pca",
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
