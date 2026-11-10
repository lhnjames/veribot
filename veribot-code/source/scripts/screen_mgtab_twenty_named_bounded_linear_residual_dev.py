"""Dev-only: twenty named source concepts plus a bounded 16-coefficient residual."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_ab_monotone_evidence_tree_dev import directions, evidence_matrix
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_supervised_bounded_linear_residual_dev import bounded, fit
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_twenty_named_bounded_linear_residual/dev_only.json"
L2_GRID = (0.0, .0001, .001, .01)


def fixed_groups(runtime: MGTABFullAtomicRuntime) -> dict[str, list[str]]:
    names = runtime.names
    groups = {
        "profile_audience_counts": [names[j] for j in (3, 5, 7)],
        "profile_audience_ratio": [names[14]],
        "profile_activity_and_presentation": [names[j] for j in range(20)
                                              if j not in (3, 5, 7, 14)],
    }
    for label, indices in runtime.group_fact_indices.items():
        if label.startswith("published_relation_"):
            groups[label] = [names[j] for j in indices]
    used = [name for members in groups.values() for name in members]
    if len(groups) != 10 or len(used) != 86 or len(set(used)) != 86:
        raise ValueError("Fixed named source groups do not partition 86 facts")
    return groups


def rows(runtime: MGTABFullAtomicRuntime,
         selected: np.ndarray,
         groups: dict[str, list[str]]) -> tuple[np.ndarray, np.ndarray]:
    batch = runtime.reference.transform(runtime.facts[selected])
    reliability = batch.reliability.copy()
    reliability[:, 62:86] *= runtime.supports[selected] / (
        runtime.supports[selected] + 5.0)
    atomic = EvidenceBatch(batch.upper[:, :86], batch.lower[:, :86],
                           reliability[:, :86], batch.available[:, :86],
                           batch.reference_version, batch.names[:86])
    concept, _ = collapse_group_tails(atomic, groups)
    if len(concept.names) != 20:
        raise ValueError("Expected exactly twenty named source concepts")
    content = np.column_stack((
        np.log1p(batch.upper[:, 86:94]) * reliability[:, 86:94],
        np.log1p(batch.lower[:, 86:94]) * reliability[:, 86:94]))
    return evidence_matrix(concept), content


def fit_named(runtime: MGTABFullAtomicRuntime,
              x_a: np.ndarray, y_a: np.ndarray,
              x_b: np.ndarray, y_b: np.ndarray) -> LGBMClassifier:
    signs = directions(x_a, y_a)
    source_groups = [[2*k, 2*k+1, 20+2*k, 21+2*k]
                     for k in range(10)]
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
        raise ValueError("Named model crossed source groups")
    return model


def main(group_builder=fixed_groups, output_path: Path = OUT,
         scheme_name: str = "three_profile_seven_relation") -> None:
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load(
            "mgtab", runtime.data_dir, None, seed)
        if ids != runtime.ids or source["split_sha256"] != (
                runtime.lock["split_sha256_by_seed"][str(seed)]):
            raise ValueError("MGTAB official population differs")
        groups = group_builder(runtime)
        with np.load(runtime.run_dir / "ab_direction_contract.npz",
                     allow_pickle=False) as contract:
            a = contract["a_indices"].astype(int)
            b = contract["b_indices"].astype(int)
        if len(np.intersect1d(a, b)) or not np.array_equal(
                np.sort(np.r_[a, b]), np.sort(split["train"])):
            raise ValueError("MGTAB A/B partition differs")
        x_a, content_a = rows(runtime, a, groups)
        x_b, content_b = rows(runtime, b, groups)
        dev = split["val"]
        x_dev, content_dev = rows(runtime, dev, groups)
        model = fit_named(runtime, x_a, labels[a], x_b, labels[b])
        named_a = np.asarray(model.booster_.predict(x_a, raw_score=True))
        named_b = np.asarray(model.booster_.predict(x_b, raw_score=True))
        named_dev = np.asarray(model.booster_.predict(x_dev, raw_score=True))
        named_probability = expit(named_dev)
        named_threshold = best_validation_threshold(named_probability, labels[dev])
        named_metric = evaluate(labels[dev], named_probability, named_threshold)
        candidates = []
        for fitting_half, x_fit, named_fit, fit_idx in (
                ("A_out_of_sample_named_score", content_a, named_a, a),
                ("B_in_sample_named_score", content_b, named_b, b)):
            for l2 in L2_GRID:
                weights, optimization = fit(x_fit, named_fit, labels[fit_idx], l2)
                residual = bounded(content_dev, weights)
                p = expit(named_dev + residual)
                threshold = best_validation_threshold(p, labels[dev])
                candidates.append({
                    "residual_fit_half": fitting_half,
                    "l2": l2,
                    "validation": evaluate(labels[dev], p, threshold),
                    "threshold": threshold,
                    "optimization": optimization,
                    "residual_changes_named_only_class": int(np.count_nonzero(
                        (p >= threshold) != (named_probability >= threshold))),
                    "max_absolute_residual_on_dev": float(np.max(np.abs(residual))),
                })
        run = {
            "seed": seed,
            "split_sha256": source["split_sha256"],
            "reference_version": runtime.reference.version,
            "named_concept_count": 20,
            "named_group_fact_counts": {key: len(value) for key, value in groups.items()},
            "numeric_residual_input_coordinates": 16,
            "numeric_residual_trainable_coefficients": 16,
            "numeric_residual_absolute_logit_bound": 3.0,
            "named_only_validation": named_metric,
            "named_only_threshold": named_threshold,
            "candidates": candidates,
        }
        runs.append(run)
        print({"seed": seed, "named_dev_f1": round(named_metric["macro_f1"], 5),
               "best_dev": max((row["validation"]["macro_f1"],
                                row["residual_fit_half"], row["l2"])
                               for row in candidates)}, flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "scope": "mgtab_twenty_named_concepts_plus_bounded_sixteen_coefficient_residual_dev_only",
        "group_scheme": scheme_name,
        "reference_uses_training_facts_without_labels": True,
        "fixed_named_vocabulary_uses_published_profile_and_typed_edges_only": True,
        "vocabulary_preselected_before_prior_label_exploration": False,
        "content_coordinates_are_explicit_numeric_residual_not_named_concepts": True,
        "A_and_B_named_direction_and_fit_disjoint": True,
        "test_predictions_or_metrics_accessed": False,
        "prior_test_population_exploration": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
