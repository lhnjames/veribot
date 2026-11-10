"""Train/dev screen of a low-dimensional, globally bounded content residual."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_bounded_content_residual import fit_residual
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural, groups_for, rows
from verifiable_bot.mgtab_neighbor_facts import evidence_with_neighbor_support
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_bounded_residual_cap/dev_only.json"
PREVIOUS = ROOT / "output/v3_mgtab_stronger_bounded_residual/dev_only.json"
GLOBAL_RESIDUAL_CAP = 3.0
CONFIGS = ((16, 0.5), (16, 0.75), (16, 1.0), (32, 0.5))


def main() -> None:
    data = resolve_mgtab_source_dir(ROOT)
    structural = ROOT / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = ROOT / "data/processed/mgtab_v3/neighbor_facts.npz"
    audit_path = ROOT / "research/certificates/mgtab_full_source_audit.json"
    prior = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit["status"] != "verified_all_published_tensor_derived_fact_cells" or
            prior["source_features_sha256"] != sha256_file(data / "features.pt") or
            prior["structural_facts_sha256"] != sha256_file(structural) or
            prior["neighbor_facts_sha256"] != sha256_file(neighbor)):
        raise ValueError("published source or earlier control changed")
    published = torch.load(data / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    content = published[:, 20:]
    output = []
    for seed in (0, 1, 2):
        facts, labels, ids, names, split, source = load(
            "mgtab", data, None, seed)
        facts, names, source = attach_mgtab_structural(
            facts, names, ids, source, data, structural)
        facts, names, supports, source = attach_mgtab_neighbor(
            facts, names, ids, source, data, neighbor)
        if source["split_sha256"] != prior["runs"][seed]["split_sha256"]:
            raise ValueError("official train/dev split changed")
        train, val = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(
            facts[train], names)
        if reference.version != prior["runs"][seed]["reference_version"]:
            raise ValueError("label-blind reference changed")
        evidence = evidence_with_neighbor_support(
            reference, facts, supports)
        a, b = train_test_split(train, test_size=0.5,
                                stratify=labels[train], random_state=seed)
        reasoner = MonotoneSplineReasoner(
            groups_for("mgtab", names), l2=1e-5)
        reasoner.fit(rows(evidence, a), labels[a],
                     rows(evidence, b), labels[b])
        base_b = reasoner.logits(rows(evidence, b))
        base_val = reasoner.logits(rows(evidence, val))
        base_p = expit(base_val)
        base_threshold = best_validation_threshold(base_p, labels[val])
        base_result = evaluate(labels[val], base_p, base_threshold)
        previous_base = prior["runs"][seed]["base_86_evidence_only"]["validation"]
        if any(abs(base_result[key] - previous_base[key]) > 1e-9
               for key in ("acc", "macro_f1", "auroc")):
            raise ValueError("named evidence control did not replay")
        scaler = StandardScaler().fit(content[train])
        normalized = scaler.transform(content)
        projections = {}
        for dimensions in (16, 32):
            pca = PCA(n_components=dimensions, svd_solver="randomized",
                      random_state=seed).fit(normalized[train])
            projected = pca.transform(normalized)
            projected /= np.maximum(np.sqrt(pca.explained_variance_), 1e-6)
            projections[dimensions] = np.tanh(projected)
        candidates = {}
        for dimensions, coefficient_bound in CONFIGS:
            z = projections[dimensions]
            weights = fit_residual(base_b, z[b], labels[b], coefficient_bound)
            raw = z[val] @ weights
            residual_val = np.clip(raw, -GLOBAL_RESIDUAL_CAP,
                                   GLOBAL_RESIDUAL_CAP)
            probability = expit(base_val + residual_val)
            threshold = best_validation_threshold(probability, labels[val])
            evaluation = evaluate(labels[val], probability, threshold)
            key = f"dimensions_{dimensions}_coefficient_bound_{coefficient_bound}"
            candidates[key] = {
                "validation": evaluation,
                "threshold": threshold,
                "global_absolute_logit_cap": GLOBAL_RESIDUAL_CAP,
                "mean_absolute_residual_logit_dev": float(
                    np.mean(np.abs(residual_val))),
                "max_absolute_residual_logit_dev": float(
                    np.max(np.abs(residual_val))),
                "fraction_dev_residual_clipped": float(
                    np.mean(np.abs(raw) > GLOBAL_RESIDUAL_CAP)),
                "residual_changes_named_only_class": int(np.count_nonzero(
                    (probability >= threshold) !=
                    (base_p >= threshold))),
                "coefficients": weights.tolist(),
            }
            if (dimensions, coefficient_bound) == (16, 0.5):
                earlier = prior["runs"][seed]["candidates"][
                    "dimensions_16_bound_0.5"]
                if (abs(threshold - earlier["threshold"]) > 1e-9 or
                        any(abs(evaluation[name] -
                                earlier["validation"][name]) > 1e-9
                            for name in ("acc", "macro_f1", "auroc"))):
                    raise ValueError("prior 16-D bounded residual did not replay")
        output.append({"seed": seed,
                       "split_sha256": source["split_sha256"],
                       "reference_version": reference.version,
                       "named_86_evidence_only": base_result,
                       "candidates": candidates})
        print({"seed": seed, "dev_macro_f1": {
            key: round(value["validation"]["macro_f1"], 5)
            for key, value in candidates.items()}}, flush=True)
    baseline_key = "dimensions_16_coefficient_bound_0.5"
    mean_baseline_f1 = float(np.mean([
        row["candidates"][baseline_key]["validation"]["macro_f1"]
        for row in output]))
    mean_f1 = {key: float(np.mean([
        row["candidates"][key]["validation"]["macro_f1"]
        for row in output])) for key in output[0]["candidates"]}
    result = {
        "scope": "mgtab_official_train_dev_only_global_cap_low_dimensional_residual",
        "test_predictions_or_metrics_accessed": False,
        "test_labels_loaded_by_full_dataset_adapter": True,
        "test_labels_used_for_fit_or_selection": False,
        "source_features_sha256": sha256_file(data / "features.pt"),
        "previous_dev_sha256": sha256_file(PREVIOUS),
        "residual_source": "published_average_tweet_tensor_train_only_pca",
        "global_absolute_logit_cap": GLOBAL_RESIDUAL_CAP,
        "selection_gate": "candidate mean dev Macro-F1 must exceed 16-D bound-0.5 control by at least 0.005 and every seed dev Macro-F1 may fall by no more than 0.002",
        "mean_dev_macro_f1": mean_f1,
        "baseline_mean_dev_macro_f1": mean_baseline_f1,
        "gate_passing_candidates": [key for key, average in mean_f1.items()
            if key != baseline_key and average >= mean_baseline_f1 + 0.005
            and all(row["candidates"][key]["validation"]["macro_f1"] >=
                    row["candidates"][baseline_key]["validation"]["macro_f1"]
                    - 0.002 for row in output)],
        "runs": output,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
