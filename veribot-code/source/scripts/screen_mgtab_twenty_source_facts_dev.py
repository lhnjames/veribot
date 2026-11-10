"""Dev-only screen of <=20 MGTAB source facts with bounded content."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_twenty_source_facts/dev_only.json"
QUOTAS = ((4, 8, 8), (6, 6, 8), (8, 4, 8))
CAP = 3.0


def rank_facts(x_a: np.ndarray, labels_a: np.ndarray) -> np.ndarray:
    scores = np.zeros(86)
    for j in range(86):
        for column in (j, j + 150):
            values = x_a[:, column]
            if len(np.unique(values)) > 1:
                scores[j] = max(scores[j],
                                abs(roc_auc_score(labels_a, values) - .5))
    return scores


def choose(runtime: MGTABFullAtomicRuntime, scores: np.ndarray,
           profile_n: int, relation_n: int, content_n: int) -> np.ndarray:
    profile = sorted(range(20), key=lambda j: (-scores[j], j))[:profile_n]
    relation = sorted(range(20, 86), key=lambda j: (-scores[j], j))[:relation_n]
    selected = np.asarray(sorted(profile + relation) + list(range(86, 86 + content_n)))
    if (len(selected) != profile_n + relation_n + content_n or
            len(selected) > 20 or len(set(selected)) != len(selected)):
        raise ValueError("compact fact budget differs")
    return selected


def candidate(runtime: MGTABFullAtomicRuntime, x_a: np.ndarray,
              x_b: np.ndarray, x_dev: np.ndarray, labels: np.ndarray,
              a: np.ndarray, b: np.ndarray, dev: np.ndarray,
              selected: np.ndarray, quota: tuple[int, int, int]) -> dict:
    d = len(selected)
    columns = np.r_[selected, selected + 150]
    a_design, b_design, dev_design = (
        matrix[:, columns] for matrix in (x_a, x_b, x_dev))
    signs = directions(a_design, labels[a]).astype(int)
    groups = []
    group_names = []
    for name, members in runtime.group_fact_indices.items():
        local = [j for j, source_j in enumerate(selected) if source_j in members]
        if local:
            groups.append(local + [j + d for j in local])
            group_names.append(name)
    if sorted(j for members in groups for j in members) != list(range(2*d)):
        raise ValueError("source group partition differs")
    model = LGBMClassifier(
        objective="binary", n_estimators=400, num_leaves=31,
        min_child_samples=50, reg_lambda=10.0, learning_rate=.05,
        n_jobs=4, random_state=0, deterministic=True,
        force_col_wise=True, verbosity=-1,
        monotone_constraints=signs.tolist(),
        monotone_constraints_method="advanced",
        interaction_constraints=groups)
    model.fit(b_design, labels[b])
    single, tree_counts = all_trees_one_group(model, groups)
    if not single:
        raise ValueError("compact model crossed source groups")
    full_logit = np.asarray(model.booster_.predict(dev_design, raw_score=True))
    named = dev_design.copy()
    content_start = quota[0] + quota[1]
    named[:, content_start:d] = 0.0
    named[:, d + content_start:2*d] = 0.0
    named_logit = np.asarray(model.booster_.predict(named, raw_score=True))
    content_effect = full_logit - named_logit
    bounded = expit(named_logit + np.clip(content_effect, -CAP, CAP))
    threshold = best_validation_threshold(bounded, labels[dev])
    return {
        "quota_profile_relation_content": list(quota),
        "selected_fact_indices": selected.astype(int).tolist(),
        "selected_fact_names": [runtime.names[int(j)] for j in selected],
        "source_groups": group_names,
        "signs": signs.tolist(),
        "all_trees_one_source_group": single,
        "tree_count_by_group_index": tree_counts,
        "bounded_content_logit_cap": CAP,
        "clipped_dev_accounts": int(np.count_nonzero(abs(content_effect) > CAP)),
        "threshold": threshold,
        "validation": evaluate(labels[dev], bounded, threshold),
        "validation_at_half": evaluate(labels[dev], bounded, .5),
        "content_removed_class_changes_at_selected": int(np.count_nonzero(
            (expit(named_logit) >= threshold) != (bounded >= threshold))),
    }


def main() -> None:
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load("mgtab", runtime.data_dir, None, seed)
        if ids != runtime.ids or source["split_sha256"] != runtime.lock[
                "split_sha256_by_seed"][str(seed)]:
            raise ValueError("official MGTAB population differs")
        with np.load(runtime.run_dir / "ab_direction_contract.npz",
                     allow_pickle=False) as contract:
            a, b = contract["a_indices"].astype(int), contract["b_indices"].astype(int)
        if (len(set(a) & set(b)) or
                not np.array_equal(np.sort(np.r_[a, b]), np.sort(split["train"]))):
            raise ValueError("A/B labels are not disjoint")
        dev = split["val"]
        x_a, x_b, x_dev = (evidence_design(runtime, rows)
                           for rows in (a, b, dev))
        scores = rank_facts(x_a, labels[a])
        candidates = [candidate(runtime, x_a, x_b, x_dev, labels,
                                a, b, dev, choose(runtime, scores, *quota), quota)
                      for quota in QUOTAS]
        frozen_probability = runtime.model.predict_proba(x_dev)[:, 1]
        frozen = evaluate(labels[dev], frozen_probability, runtime.threshold)
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "reference_version": runtime.reference.version,
            "model_sha256": sha256_file(runtime.run_dir / "model.joblib"),
            "direction_contract_sha256": sha256_file(runtime.run_dir / "ab_direction_contract.npz"),
            "frozen_validation": frozen,
            "A_only_fact_auc_scores": scores.tolist(),
            "candidates": candidates,
        })
        print({"seed": seed, "frozen_dev_acc": frozen["acc"],
               "compact_dev_acc": [round(c["validation"]["acc"], 5)
                                   for c in candidates]}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "mgtab_named_20_fact_A_selection_B_fit_bounded_content_dev_only",
        "test_predictions_or_metrics_accessed": False,
        "prior_test_population_exploration": True,
        "source_selection_labels": "A only",
        "monotone_direction_labels": "A only",
        "tree_fit_labels": "B only",
        "quota_grid_profile_relation_content": [list(q) for q in QUOTAS],
        "content_projection_indices": list(range(86, 94)),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
