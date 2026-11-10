"""Audit whether the saved source-group subset games admit an additive logit story.

This reads immutable subset matrices; it does not fit or modify a model. An
interaction residual is descriptive of the specified encoded-group zeroing
operation, not a causal effect of deleting raw records.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logit


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "research/certificates/minimal_sufficient_source_groups.json"
OUTPUT = ROOT / "research/certificates/best_group_additivity_scope.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for item in manifest["runs"]:
        path = ROOT / "output/v3_minimal_sufficient_source_groups" / (
            f"{item['dataset']}_seed{item['seed']}.npz")
        if sha256(path) != item["subset_matrix_sha256"]:
            raise ValueError(f"subset matrix changed: {path}")
        with np.load(path, allow_pickle=False) as archive:
            probability = archive["subset_probability"].astype(float)
            group_names = archive["group_names"].astype(str).tolist()
            threshold = float(archive["threshold"])
            ids = archive["ids"].astype(str)
        if len(ids) != item["accounts"] or group_names != item["group_names"]:
            raise ValueError(f"subset population/group mismatch: {path}")
        if np.any((probability <= 0) | (probability >= 1)):
            raise ValueError(f"nonfinite logit: {path}")
        groups = len(group_names)
        if probability.shape != (len(ids), 1 << groups):
            raise ValueError(f"subset shape changed: {path}")
        score = logit(probability)
        full = score[:, -1]
        base = score[:, 0]
        singleton = np.column_stack([score[:, 1 << j] - base
                                     for j in range(groups)])
        additive = base + singleton.sum(axis=1)
        residual = full - additive
        abs_residual = np.abs(residual)
        additive_probability = 1 / (1 + np.exp(-additive))
        flip_disagreement = ((additive_probability >= threshold) !=
                             (probability[:, -1] >= threshold))
        def higher_order_terms(index: int) -> list[dict]:
            terms = []
            for mask in range(1, 1 << groups):
                if mask.bit_count() < 2:
                    continue
                value = sum(
                    (-1) ** (mask.bit_count() - subset.bit_count()) *
                    score[index, subset]
                    for subset in range(1 << groups)
                    if subset & mask == subset)
                terms.append({
                    "groups": [group_names[j] for j in range(groups)
                               if mask & (1 << j)],
                    "logit_interaction": float(value),
                })
            if abs(sum(x["logit_interaction"] for x in terms) - residual[index]) > 1e-12:
                raise ValueError("interaction reconstruction failed")
            return sorted(terms, key=lambda x: -abs(x["logit_interaction"]))
        # Shapley distributes the same subset-game interactions across groups.
        # This is an exact allocation identity, not an independent model cause.
        from math import factorial
        shapley = np.zeros((len(ids), groups), dtype=float)
        for j in range(groups):
            for mask in range(1 << groups):
                if mask & (1 << j):
                    continue
                k = mask.bit_count()
                weight = factorial(k) * factorial(groups-k-1) / factorial(groups)
                shapley[:, j] += weight * (
                    score[:, mask | (1 << j)] - score[:, mask])
        allocation_error = float(np.max(np.abs(base + shapley.sum(axis=1) - full)))
        if allocation_error > 1e-12:
            raise ValueError(f"subset allocation identity failed: {path}")
        rows.append({
            "dataset": item["dataset"], "seed": item["seed"],
            "model_variant": {
                "fox8": "raw_neighbor_74_lightgbm",
                "mgtab": "ab_group_additive",
                "botsim": "observed_mean_group_additive",
            }[item["dataset"]],
            "subset_matrix_origin": (
                "derived_from_saved_group_logit_terms_and_full_model_logit"
                if item["dataset"] == "botsim" else
                "direct_frozen_lightgbm_inference_for_every_group_subset"),
            "accounts": len(ids), "groups": group_names,
            "subset_matrix_sha256": item["subset_matrix_sha256"],
            "maximum_absolute_interaction_logit": float(abs_residual.max()),
            "median_absolute_interaction_logit": float(np.median(abs_residual)),
            "interaction_abs_gt_0_01": int(np.count_nonzero(abs_residual > .01)),
            "interaction_abs_gt_0_1": int(np.count_nonzero(abs_residual > .1)),
            "interaction_abs_gt_1": int(np.count_nonzero(abs_residual > 1)),
            "additive_approximation_class_disagreements": int(flip_disagreement.sum()),
            "additive_approximation_disagreement_cases": [
                {"id": str(ids[i]),
                 "full_probability": float(probability[i, -1]),
                 "additive_approximation_probability": float(additive_probability[i]),
                 "threshold": threshold,
                 "interaction_logit": float(residual[i]),
                 "higher_order_interactions": higher_order_terms(int(i))}
                for i in np.flatnonzero(flip_disagreement)],
            "shapley_subset_allocation_max_replay_error": allocation_error,
            "largest_interaction_account": {
                "id": str(ids[int(abs_residual.argmax())]),
                "full_probability": float(probability[abs_residual.argmax(), -1]),
                "additive_approximation_probability": float(
                    additive_probability[abs_residual.argmax()]),
                "interaction_logit": float(residual[abs_residual.argmax()]),
            },
        })
    result = {
        "status": "frozen_group_additivity_scope_audited",
        "definition": "full logit minus [zero-input logit plus sum of each singleton group's logit change]",
        "operation": "zero both encoded tails of omitted groups; raw source records are not edited",
        "semantic_limit": "exact subset score or Shapley allocation does not establish behavior truth or a unique causal reason",
        "fixed_test_labels_read": False,
        "manifest_sha256": sha256(MANIFEST),
        "runs": rows,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    for row in result["runs"]:
        print(row["dataset"], row["seed"], row["accounts"],
              row["maximum_absolute_interaction_logit"],
              row["additive_approximation_class_disagreements"])
