"""A-weighted, B-fitted 18-dimensional MGTAB source-group evidence screen."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load
from screen_mgtab_eighteen_group_summaries_dev import (
    MAX_MEAN_MACRO_F1_LOSS_PP, ROOT, candidate)
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


OUT = ROOT / "output/v3_mgtab_A_weighted_group_summaries/dev_only.json"


def learn_weights(a_x: np.ndarray, a_y: np.ndarray,
                  groups: list[list[int]]) -> np.ndarray:
    if a_x.shape[1] != 300 or len(groups) != 9:
        raise ValueError("A-only evidence input differs")
    auc = np.full(300, .5)
    for column in range(300):
        values = a_x[:, column]
        if np.unique(values).size > 1:
            auc[column] = roc_auc_score(a_y, values)
    weights = np.zeros((18, 300), dtype=float)
    for group_number, members in enumerate(groups):
        columns = np.r_[members, np.asarray(members) + 150]
        positive = np.maximum(auc[columns] - .5, 0.0)
        negative = np.maximum(.5 - auc[columns], 0.0)
        if positive.sum() > 0:
            weights[2*group_number, columns] = positive / positive.sum()
        if negative.sum() > 0:
            weights[2*group_number+1, columns] = negative / negative.sum()
    if not np.isfinite(weights).all():
        raise ValueError("A-only group evidence weights are nonfinite")
    return weights


def main() -> None:
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load("mgtab", runtime.data_dir,
                                               None, seed)
        if ids != runtime.ids or source["split_sha256"] != runtime.lock[
                "split_sha256_by_seed"][str(seed)]:
            raise ValueError("MGTAB split or source population differs")
        with np.load(runtime.run_dir / "ab_direction_contract.npz",
                     allow_pickle=False) as contract:
            a = contract["a_indices"].astype(int)
            b = contract["b_indices"].astype(int)
        dev = split["val"]
        if (len(set(a) & set(b)) or
                not np.array_equal(np.sort(np.r_[a, b]),
                                   np.sort(split["train"]))):
            raise ValueError("A/B population differs")
        groups = list(runtime.group_fact_indices.values())
        raw_a, raw_b, raw_dev = (evidence_design(runtime, rows)
                                 for rows in (a, b, dev))
        weights = learn_weights(raw_a, labels[a], groups)
        pooled_a, pooled_b, pooled_dev = (matrix @ weights.T
                                           for matrix in (raw_a, raw_b, raw_dev))
        result = candidate(pooled_a, pooled_b, pooled_dev,
                           labels[a], labels[b], labels[dev],
                           "A_only_signed_AUC_weights")
        frozen = evaluate(labels[dev],
                          runtime.model.predict_proba(raw_dev)[:, 1],
                          runtime.threshold)
        runs.append({
            "seed": seed,
            "split_sha256": source["split_sha256"],
            "reference_version": runtime.reference.version,
            "frozen_150_fact_dev": frozen,
            "weighted_18_summary_dev": result,
            "A_only_weights": [
                [{"source_fact": runtime.names[j % 150],
                  "tail": "upper" if j < 150 else "lower",
                  "weight": float(weights[row, j])}
                 for j in np.flatnonzero(weights[row])]
                for row in range(18)],
        })
        print({"seed": seed, "frozen_macro_f1": frozen["macro_f1"],
               "weighted_macro_f1": result["dev"]["macro_f1"]}, flush=True)
    losses = [100.0*(run["frozen_150_fact_dev"]["macro_f1"] -
                     run["weighted_18_summary_dev"]["dev"]["macro_f1"])
              for run in runs]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "A_only_fact_direction_weights_B_only_final_fit_official_dev_only",
        "test_prediction_or_metric_accessed": False,
        "concept_count": 18,
        "group_count": 9,
        "weight_formula": "within group and both tails: positive/negative signed A-half univariate AUC deviation from 0.5, normalized to sum 1",
        "all_weights_nonnegative": True,
        "max_mean_macro_f1_loss_pp": MAX_MEAN_MACRO_F1_LOSS_PP,
        "macro_f1_loss_pp_by_seed": losses,
        "mean_macro_f1_loss_pp": float(np.mean(losses)),
        "passes_predeclared_mean_3pp_gate":
            bool(np.mean(losses) <= MAX_MEAN_MACRO_F1_LOSS_PP),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
