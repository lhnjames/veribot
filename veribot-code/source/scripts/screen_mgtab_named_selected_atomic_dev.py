"""Dev-only 20-concept bottleneck with A-selected named atomic facts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load
from screen_mgtab_named_topk_pool_dev import fit
from screen_mgtab_twenty_named_ratio_struct_neighbor_dev import fixed_groups
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_named_selected_atomic/dev_only.json"
MODES = ("tail_specific", "same_fact_both_tails")


def named_atomic(runtime: MGTABFullAtomicRuntime,
                 selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch = runtime.reference.transform(runtime.facts[selected])
    reliability = batch.reliability.copy()
    reliability[:, 62:86] *= runtime.supports[selected] / (
        runtime.supports[selected] + 5.0)
    return (np.log1p(batch.upper[:, :86]) * reliability[:, :86],
            np.log1p(batch.lower[:, :86]) * reliability[:, :86])


def standardized_difference(values: np.ndarray,
                            labels: np.ndarray) -> np.ndarray:
    bot = values[labels == 1]
    human = values[labels == 0]
    return np.abs(bot.mean(axis=0) - human.mean(axis=0)) / np.sqrt(
        bot.var(axis=0) + human.var(axis=0) + 1e-12)


def select(group_indices: list[list[int]],
           high_a: np.ndarray, low_a: np.ndarray,
           labels_a: np.ndarray, mode: str) -> list[tuple[int, int]]:
    high_scores = standardized_difference(high_a, labels_a)
    low_scores = standardized_difference(low_a, labels_a)
    chosen = []
    for indices in group_indices:
        if mode == "tail_specific":
            high = max(indices, key=lambda j: (high_scores[j], -j))
            low = max(indices, key=lambda j: (low_scores[j], -j))
        elif mode == "same_fact_both_tails":
            selected = max(indices, key=lambda j: (
                max(high_scores[j], low_scores[j]), -j))
            high = low = selected
        else:
            raise ValueError("unknown selection mode")
        chosen.append((high, low))
    return chosen


def design(high: np.ndarray, low: np.ndarray,
           chosen: list[tuple[int, int]]) -> np.ndarray:
    return np.column_stack([values[:, j] for high_j, low_j in chosen
                            for values, j in ((high, high_j), (low, low_j))])


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
        index = {name: i for i, name in enumerate(runtime.names[:86])}
        group_indices = [[index[name] for name in members]
                         for members in groups.values()]
        high_a, low_a = named_atomic(runtime, a)
        high_b, low_b = named_atomic(runtime, b)
        high_dev, low_dev = named_atomic(runtime, split["val"])
        candidates = []
        for mode in MODES:
            chosen = select(group_indices, high_a, low_a, labels[a], mode)
            x_a = design(high_a, low_a, chosen)
            x_b = design(high_b, low_b, chosen)
            x_dev = design(high_dev, low_dev, chosen)
            if x_a.shape[1] != 20 or x_b.shape[1] != 20:
                raise ValueError("twenty-concept bottleneck differs")
            model = fit(x_a, labels[a], x_b, labels[b])
            probability = model.predict_proba(x_dev)[:, 1]
            threshold = best_validation_threshold(
                probability, labels[split["val"]])
            candidates.append({
                "mode": mode,
                "selected_named_facts": [{
                    "group": group,
                    "high": runtime.names[chosen[j][0]],
                    "low": runtime.names[chosen[j][1]],
                } for j, group in enumerate(groups)],
                "distinct_atomic_facts": len(set(j for pair in chosen for j in pair)),
                "validation": evaluate(labels[split["val"]], probability,
                                       threshold),
                "threshold": threshold,
            })
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "reference_version": runtime.reference.version,
                     "candidates": candidates})
        print({"seed": seed, "macro_f1": {
            row["mode"]: round(row["validation"]["macro_f1"], 6)
            for row in candidates}}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "mgtab_20_A_selected_named_atomic_concepts_B_fitted_tree_dev_only",
        "selection_modes_predeclared": list(MODES),
        "selection_and_direction_use_train_A_labels_only": True,
        "tree_fit_uses_disjoint_train_B_labels_only": True,
        "test_predictions_or_metrics_accessed": False,
        "prior_test_population_exploration": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
