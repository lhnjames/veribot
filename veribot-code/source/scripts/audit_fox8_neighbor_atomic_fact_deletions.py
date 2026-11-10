"""Exact frozen-model ablations for every Fox8 atomic evidence input."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_fox8_neighbor_atomic_evidence_capacity"
    runs = []
    for seed in (0, 1, 2):
        run_dir = folder / f"seed{seed}"
        model = joblib.load(run_dir / "model.joblib")
        with np.load(run_dir / "test_evidence.npz", allow_pickle=False) as saved:
            x = saved["evidence_matrix"].copy()
            ids = saved["ids"].astype(str).tolist()
            names = saved["names"].astype(str).tolist()
            probability = saved["probability"].copy()
        if x.shape != (len(ids), 148) or len(names) != 74:
            raise ValueError("frozen evidence dimension differs")
        fresh = model.predict_proba(x)[:, 1]
        if np.max(np.abs(fresh - probability)) > 1e-14:
            raise ValueError("frozen original predictions differ")
        n, width = x.shape
        empty_probability = model.predict_proba(np.zeros_like(x))[:, 1]
        if np.max(empty_probability) - np.min(empty_probability) > 1e-14:
            raise ValueError("empty evidence has inconsistent frozen response")
        without_neighbors = x.copy()
        without_neighbors[:, 43:74] = 0.0
        without_neighbors[:, 117:148] = 0.0
        no_neighbor_probability = model.predict_proba(without_neighbors)[:, 1]
        variants = np.tile(x, (74, 1))
        for j in range(74):
            rows = slice(j * n, (j + 1) * n)
            variants[rows, j] = 0.0
            variants[rows, j + 74] = 0.0
        after = model.predict_proba(variants)[:, 1].reshape(74, n).T
        change = np.abs(after - probability[:, None])
        class_flip = (after >= 0.5) != (probability[:, None] >= 0.5)
        per_fact = []
        for j, name in enumerate(names):
            per_fact.append({
                "fact": name,
                "accounts_with_probability_change_gt_1e_10": int(np.count_nonzero(
                    change[:, j] > 1e-10)),
                "accounts_with_class_flip": int(np.count_nonzero(class_flip[:, j])),
                "max_probability_change": float(np.max(change[:, j])),
            })
        runs.append({
            "seed": seed, "test_accounts": n,
            "all_zero_evidence_probability": float(empty_probability[0]),
            "all_zero_evidence_class": int(empty_probability[0] >= 0.5),
            "accounts_whose_class_requires_some_evidence": int(np.count_nonzero(
                (empty_probability >= 0.5) != (probability >= 0.5))),
            "accounts_whose_class_requires_native_neighbor_group":
                int(np.count_nonzero((no_neighbor_probability >= 0.5) !=
                                     (probability >= 0.5))),
            "single_fact_masks_evaluated": n * 74,
            "accounts_with_at_least_one_probability_change_gt_1e_10":
                int(np.count_nonzero(np.any(change > 1e-10, axis=1))),
            "accounts_with_at_least_one_single_fact_class_flip":
                int(np.count_nonzero(np.any(class_flip, axis=1))),
            "native_neighbor_fact_mask_class_flips": int(np.count_nonzero(
                class_flip[:, 43:])),
            "top_by_single_fact_class_flip": sorted(
                per_fact, key=lambda item: item["accounts_with_class_flip"],
                reverse=True)[:10],
        })
        print(json.dumps({key: runs[-1][key] for key in (
            "seed", "accounts_with_at_least_one_probability_change_gt_1e_10",
            "accounts_with_at_least_one_single_fact_class_flip",
            "native_neighbor_fact_mask_class_flips")}), flush=True)
    output = {
        "scope": "fox8_frozen_evidence_only_model_all_test_single_fact_input_masks",
        "intervention": "set the two encoded tail columns for one fact to zero at frozen inference; do not alter the released source graph",
        "runs": runs,
    }
    path = root / "research/certificates/fox8_neighbor_atomic_single_fact_deletions.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
