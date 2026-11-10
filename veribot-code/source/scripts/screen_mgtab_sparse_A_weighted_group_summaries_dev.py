"""Train/dev screen of sparse A-selected weights in 18 MGTAB evidence groups."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load
from screen_mgtab_A_weighted_group_summaries_dev import learn_weights
from screen_mgtab_eighteen_group_summaries_dev import (
    MAX_MEAN_MACRO_F1_LOSS_PP, ROOT, candidate)
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


OUT = ROOT / "output/v3_mgtab_sparse_A_weighted_group_summaries/dev_only.json"
TOP_K = (8, 16)


def sparsify(weights: np.ndarray, top_k: int) -> np.ndarray:
    if weights.shape != (18, 300) or top_k not in TOP_K:
        raise ValueError("A-only source weight shape or budget differs")
    result = np.zeros_like(weights)
    for row in range(18):
        selected = np.argsort(-weights[row], kind="stable")[:top_k]
        result[row, selected] = weights[row, selected]
        total = result[row].sum()
        if total > 0:
            result[row] /= total
    if (not np.isfinite(result).all() or
            np.count_nonzero(result, axis=1).max() > top_k):
        raise ValueError("sparse source weights differ")
    return result


def main() -> None:
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load("mgtab", runtime.data_dir,
                                               None, seed)
        if ids != runtime.ids or source["split_sha256"] != runtime.lock[
                "split_sha256_by_seed"][str(seed)]:
            raise ValueError("official source population or split differs")
        with np.load(runtime.run_dir / "ab_direction_contract.npz",
                     allow_pickle=False) as contract:
            a = contract["a_indices"].astype(int)
            b = contract["b_indices"].astype(int)
        dev = split["val"]
        if (len(set(a) & set(b)) or
                not np.array_equal(np.sort(np.r_[a, b]),
                                   np.sort(split["train"]))):
            raise ValueError("A/B split differs")
        raw_a, raw_b, raw_dev = (evidence_design(runtime, rows)
                                 for rows in (a, b, dev))
        base = learn_weights(raw_a, labels[a],
                             list(runtime.group_fact_indices.values()))
        frozen = evaluate(labels[dev],
                          runtime.model.predict_proba(raw_dev)[:, 1],
                          runtime.threshold)
        candidates = []
        for top_k in TOP_K:
            weights = sparsify(base, top_k)
            pooled_a, pooled_b, pooled_dev = (matrix @ weights.T
                                               for matrix in (raw_a, raw_b, raw_dev))
            result = candidate(pooled_a, pooled_b, pooled_dev,
                               labels[a], labels[b], labels[dev],
                               f"A_only_signed_AUC_top_{top_k}")
            result["max_nonzero_fact_tail_slots_per_summary"] = top_k
            result["nonzero_fact_tail_slots"] = int(np.count_nonzero(weights))
            result["distinct_source_facts"] = int(len({
                int(column) % 150 for column in np.flatnonzero(
                    np.any(weights > 0, axis=0))}))
            result["source_weights"] = [
                [{"source_fact": runtime.names[j % 150],
                  "tail": "upper" if j < 150 else "lower",
                  "weight": float(weights[row, j])}
                 for j in np.flatnonzero(weights[row])]
                for row in range(18)]
            candidates.append(result)
        runs.append({"seed": seed,
                     "split_sha256": source["split_sha256"],
                     "reference_version": runtime.reference.version,
                     "frozen_150_fact_dev": frozen,
                     "candidates": candidates})
        print({"seed": seed, "frozen_macro_f1": frozen["macro_f1"],
               "sparse_macro_f1": [c["dev"]["macro_f1"]
                                   for c in candidates]}, flush=True)
    summary = []
    for index, top_k in enumerate(TOP_K):
        losses = [100.0*(run["frozen_150_fact_dev"]["macro_f1"] -
                         run["candidates"][index]["dev"]["macro_f1"])
                  for run in runs]
        summary.append({
            "top_k_per_group_direction": top_k,
            "loss_pp_by_seed": losses,
            "mean_macro_f1_loss_pp": float(np.mean(losses)),
            "passes_predeclared_mean_3pp_gate":
                bool(np.mean(losses) <= MAX_MEAN_MACRO_F1_LOSS_PP),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "A_only_sparse_fact_weights_B_only_fit_official_dev_only",
        "test_prediction_or_metric_accessed": False,
        "concept_count": 18,
        "top_k_grid_fixed_before_dev": list(TOP_K),
        "max_mean_macro_f1_loss_pp": MAX_MEAN_MACRO_F1_LOSS_PP,
        "runs": runs,
        "summary": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
