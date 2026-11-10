"""Dev-only screen of source-traceable top-k pooling for 20 named concepts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_twenty_named_ratio_struct_neighbor_dev import fixed_groups
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_named_topk_pool/dev_only.json"
K_GRID = (1, 2, 3)


def matrix(runtime: MGTABFullAtomicRuntime, selected: np.ndarray,
           groups: dict[str, list[str]], k: int) -> np.ndarray:
    """Average the strongest k available atomic tails per named source group."""
    batch = runtime.reference.transform(runtime.facts[selected])
    reliability = batch.reliability.copy()
    reliability[:, 62:86] *= runtime.supports[selected] / (
        runtime.supports[selected] + 5.0)
    name_to_index = {name: i for i, name in enumerate(batch.names)}
    columns = []
    for members in groups.values():
        indices = [name_to_index[name] for name in members]
        for values in (batch.upper, batch.lower):
            signal = np.log1p(values[:, indices]) * reliability[:, indices]
            present = batch.available[:, indices]
            ordered = np.sort(np.where(present, signal, -np.inf), axis=1)[:, ::-1]
            selected_values = ordered[:, :k]
            valid = np.isfinite(selected_values)
            pooled = np.where(valid, selected_values, 0.0).sum(axis=1)
            pooled /= np.maximum(valid.sum(axis=1), 1)
            columns.append(pooled)
    return np.column_stack(columns)


def fit(x_a: np.ndarray, y_a: np.ndarray,
        x_b: np.ndarray, y_b: np.ndarray) -> LGBMClassifier:
    signs = directions(x_a, y_a)
    source_groups = [[2*j, 2*j+1] for j in range(10)]
    model = LGBMClassifier(
        objective="binary", n_estimators=400, num_leaves=31,
        min_child_samples=50, reg_lambda=10.0, learning_rate=.05,
        n_jobs=4, random_state=0, deterministic=True, force_col_wise=True,
        verbosity=-1, monotone_constraints=signs.tolist(),
        monotone_constraints_method="advanced",
        interaction_constraints=source_groups)
    model.fit(x_b, y_b)
    single, counts = all_trees_one_group(model, source_groups)
    if not single or sum(counts.values()) != 400:
        raise ValueError("tree crossed source groups")
    return model


def main() -> None:
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load(
            "mgtab", runtime.data_dir, None, seed)
        if (ids != runtime.ids or source["split_sha256"] !=
                runtime.lock["split_sha256_by_seed"][str(seed)]):
            raise ValueError("official MGTAB train/dev population differs")
        with np.load(runtime.run_dir / "ab_direction_contract.npz",
                     allow_pickle=False) as contract:
            a = contract["a_indices"].astype(int)
            b = contract["b_indices"].astype(int)
        if (len(np.intersect1d(a, b)) or
                not np.array_equal(np.sort(np.r_[a, b]),
                                   np.sort(split["train"]))):
            raise ValueError("A/B split differs")
        groups = fixed_groups(runtime)
        candidates = []
        for k in K_GRID:
            x_a = matrix(runtime, a, groups, k)
            x_b = matrix(runtime, b, groups, k)
            x_dev = matrix(runtime, split["val"], groups, k)
            for transform in (("direct",) if k > 1 else
                              ("direct", "double_log_duplicate")):
                if transform == "double_log_duplicate":
                    t = lambda x: np.column_stack((np.log1p(x), np.log1p(x)))
                    source_groups = [[2*j, 2*j+1, 20+2*j, 21+2*j]
                                     for j in range(10)]
                    signs = directions(t(x_a), labels[a])
                    model = LGBMClassifier(
                        objective="binary", n_estimators=400, num_leaves=31,
                        min_child_samples=50, reg_lambda=10.0,
                        learning_rate=.05, n_jobs=4, random_state=0,
                        deterministic=True, force_col_wise=True,
                        verbosity=-1, monotone_constraints=signs.tolist(),
                        monotone_constraints_method="advanced",
                        interaction_constraints=source_groups)
                    model.fit(t(x_b), labels[b])
                    single, counts = all_trees_one_group(model, source_groups)
                    if not single or sum(counts.values()) != 400:
                        raise ValueError("baseline trees crossed source groups")
                    design_dev = t(x_dev)
                else:
                    model = fit(x_a, labels[a], x_b, labels[b])
                    design_dev = x_dev
                probability = model.predict_proba(design_dev)[:, 1]
                threshold = best_validation_threshold(
                    probability, labels[split["val"]])
                candidates.append({
                    "top_k": k,
                    "transform": transform,
                    "validation": evaluate(labels[split["val"]],
                                           probability, threshold),
                    "threshold": threshold,
                })
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "reference_version": runtime.reference.version,
                     "named_groups": list(groups), "candidates": candidates})
        print({"seed": seed, "macro_f1": {
            f"{v['top_k']}:{v['transform']}":
                round(v["validation"]["macro_f1"], 6)
            for v in candidates}}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "mgtab_20_named_concepts_topk_source_pool_dev_only",
        "top_k_predeclared": list(K_GRID),
        "train_labels_used_only_for_direction_and_tree_fit": True,
        "pooling_uses_no_labels_or_predictions": True,
        "test_predictions_or_metrics_accessed": False,
        "prior_test_population_exploration": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
