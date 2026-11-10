"""Train/dev screen of 18 pooled MGTAB source-group tail summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_eighteen_group_summaries/dev_only.json"
POOLING = ("maximum", "top_three_mean")
CONTENT_CAP = 3.0
MAX_MEAN_MACRO_F1_LOSS_PP = 3.0


def pool(matrix: np.ndarray, groups: list[list[int]], method: str) -> np.ndarray:
    if matrix.shape[1] != 300 or method not in POOLING or len(groups) != 9:
        raise ValueError("MGTAB source-group pooling contract differs")
    values = []
    for members in groups:
        for offset in (0, 150):
            source = matrix[:, np.asarray(members) + offset]
            if method == "maximum":
                pooled = np.max(source, axis=1)
            else:
                pooled = np.mean(np.sort(source, axis=1)[:, -3:], axis=1)
            values.append(pooled)
    result = np.column_stack(values)
    if result.shape != (len(matrix), 18) or not np.isfinite(result).all():
        raise ValueError("pooled evidence is incomplete")
    return result


def candidate(a_x: np.ndarray, b_x: np.ndarray, dev_x: np.ndarray,
              a_y: np.ndarray, b_y: np.ndarray, dev_y: np.ndarray,
              method: str) -> dict:
    signs = directions(a_x, a_y).astype(int)
    groups = [list(range(2*k, 2*k+2)) for k in range(9)]
    model = LGBMClassifier(
        objective="binary", n_estimators=400, num_leaves=31,
        min_child_samples=50, reg_lambda=10.0, learning_rate=.05,
        n_jobs=4, random_state=0, deterministic=True,
        force_col_wise=True, verbosity=-1,
        monotone_constraints=signs.tolist(),
        monotone_constraints_method="advanced",
        interaction_constraints=groups)
    model.fit(b_x, b_y)
    single, tree_counts = all_trees_one_group(model, groups)
    if not single:
        raise ValueError("pooled model tree crossed source groups")
    full = np.asarray(model.booster_.predict(dev_x, raw_score=True))
    without_content = dev_x.copy()
    without_content[:, -2:] = 0.0
    named = np.asarray(model.booster_.predict(without_content, raw_score=True))
    content = full - named
    probability = expit(named + np.clip(content, -CONTENT_CAP, CONTENT_CAP))
    threshold = best_validation_threshold(probability, dev_y)
    return {
        "pooling": method,
        "monotone_signs_from_A": signs.tolist(),
        "tree_counts_by_source_group": tree_counts,
        "all_trees_one_source_group": single,
        "content_logit_cap": CONTENT_CAP,
        "content_cap_active_dev_accounts": int(np.count_nonzero(
            np.abs(content) > CONTENT_CAP)),
        "threshold": threshold,
        "dev": evaluate(dev_y, probability, threshold),
        "dev_at_half": evaluate(dev_y, probability, .5),
        "content_group_removed_class_changes": int(np.count_nonzero(
            (expit(named) >= threshold) != (probability >= threshold))),
    }


def main() -> None:
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load("mgtab", runtime.data_dir,
                                               None, seed)
        if ids != runtime.ids or source["split_sha256"] != runtime.lock[
                "split_sha256_by_seed"][str(seed)]:
            raise ValueError("MGTAB official split differs")
        with np.load(runtime.run_dir / "ab_direction_contract.npz",
                     allow_pickle=False) as contract:
            a = contract["a_indices"].astype(int)
            b = contract["b_indices"].astype(int)
        dev = split["val"]
        if (len(set(a) & set(b)) or
                not np.array_equal(np.sort(np.r_[a, b]),
                                   np.sort(split["train"]))):
            raise ValueError("A/B split differs from frozen training population")
        groups = list(runtime.group_fact_indices.values())
        raw_a, raw_b, raw_dev = (evidence_design(runtime, rows)
                                 for rows in (a, b, dev))
        frozen_prob = runtime.model.predict_proba(raw_dev)[:, 1]
        frozen = evaluate(labels[dev], frozen_prob, runtime.threshold)
        candidates = []
        for method in POOLING:
            pooled_a, pooled_b, pooled_dev = (
                pool(matrix, groups, method)
                for matrix in (raw_a, raw_b, raw_dev))
            candidates.append(candidate(pooled_a, pooled_b, pooled_dev,
                                        labels[a], labels[b], labels[dev],
                                        method))
        runs.append({"seed": seed,
                     "split_sha256": source["split_sha256"],
                     "reference_version": runtime.reference.version,
                     "frozen_150_fact_dev": frozen,
                     "candidates": candidates})
        print({"seed": seed, "frozen_macro_f1": frozen["macro_f1"],
               "pool_macro_f1": [x["dev"]["macro_f1"]
                                 for x in candidates]}, flush=True)
    summaries = []
    for index, method in enumerate(POOLING):
        losses = [100.0 * (run["frozen_150_fact_dev"]["macro_f1"] -
                           run["candidates"][index]["dev"]["macro_f1"])
                  for run in runs]
        summaries.append({
            "pooling": method,
            "macro_f1_loss_pp_by_seed": losses,
            "mean_macro_f1_loss_pp": float(np.mean(losses)),
            "passes_predeclared_mean_3pp_gate":
                bool(np.mean(losses) <= MAX_MEAN_MACRO_F1_LOSS_PP),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "train_A_direction_train_B_fit_official_dev_only",
        "test_prediction_or_metric_accessed": False,
        "source_groups": list(runtime.group_fact_indices),
        "facts_per_group": [len(group) for group in groups],
        "summaries_per_group": 2,
        "summary_formula": "max or mean of top 3 nonnegative reliability-weighted log1p upper/lower tail inputs",
        "pooling_grid_fixed_before_dev": list(POOLING),
        "max_mean_macro_f1_loss_pp": MAX_MEAN_MACRO_F1_LOSS_PP,
        "runs": runs,
        "summary": summaries,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
