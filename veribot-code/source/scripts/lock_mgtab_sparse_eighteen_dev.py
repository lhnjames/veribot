"""Freeze the validation-selected sparse 18-summary MGTAB model before test."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_A_weighted_group_summaries_dev import learn_weights
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from screen_mgtab_eighteen_group_summaries_dev import CONTENT_CAP, ROOT
from screen_mgtab_sparse_A_weighted_group_summaries_dev import sparsify
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


SCREEN = ROOT / "output/v3_mgtab_sparse_A_weighted_group_summaries/dev_only.json"
FOLDER = ROOT / "output/v3_mgtab_sparse_A_weighted_group_summaries/selected"
OUT = FOLDER / "dev_lock.json"
TOP_K = 8


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if (screen["test_prediction_or_metric_accessed"] or
            len(screen["runs"]) != 3 or
            not screen["summary"][0]["passes_predeclared_mean_3pp_gate"] or
            screen["summary"][1]["passes_predeclared_mean_3pp_gate"] or
            screen["top_k_grid_fixed_before_dev"] != [8, 16]):
        raise ValueError("validation gate or candidate grid differs")
    FOLDER.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load("mgtab", runtime.data_dir,
                                               None, seed)
        prior = screen["runs"][seed]["candidates"][0]
        if ids != runtime.ids or source["split_sha256"] != screen[
                "runs"][seed]["split_sha256"]:
            raise ValueError("source or official split differs")
        with np.load(runtime.run_dir / "ab_direction_contract.npz",
                     allow_pickle=False) as contract:
            a = contract["a_indices"].astype(int)
            b = contract["b_indices"].astype(int)
        dev = split["val"]
        if (len(set(a) & set(b)) or
                not np.array_equal(np.sort(np.r_[a, b]),
                                   np.sort(split["train"]))):
            raise ValueError("A/B population differs")
        raw_a, raw_b, raw_dev = (evidence_design(runtime, rows)
                                 for rows in (a, b, dev))
        weights = sparsify(learn_weights(raw_a, labels[a], list(
            runtime.group_fact_indices.values())), TOP_K)
        if (int(np.count_nonzero(weights)) != prior[
                "nonzero_fact_tail_slots"] or
                int(len({int(column) % 150 for column in np.flatnonzero(
                    np.any(weights > 0, axis=0))})) != prior[
                        "distinct_source_facts"]):
            raise ValueError("A-only source weights differ from screen")
        a_x, b_x, dev_x = (matrix @ weights.T
                           for matrix in (raw_a, raw_b, raw_dev))
        signs = directions(a_x, labels[a]).astype(int)
        groups = [list(range(2*k, 2*k+2)) for k in range(9)]
        if signs.tolist() != prior["monotone_signs_from_A"]:
            raise ValueError("A-only monotone directions differ")
        model = LGBMClassifier(
            objective="binary", n_estimators=400, num_leaves=31,
            min_child_samples=50, reg_lambda=10.0, learning_rate=.05,
            n_jobs=4, random_state=0, deterministic=True,
            force_col_wise=True, verbosity=-1,
            monotone_constraints=signs.tolist(),
            monotone_constraints_method="advanced",
            interaction_constraints=groups)
        model.fit(b_x, labels[b])
        single, counts = all_trees_one_group(model, groups)
        if not single or counts != {int(key): value for key, value in prior[
                "tree_counts_by_source_group"].items()}:
            raise ValueError("B-only source group trees differ")
        full = np.asarray(model.booster_.predict(dev_x, raw_score=True))
        no_content = dev_x.copy()
        no_content[:, -2:] = 0.0
        named = np.asarray(model.booster_.predict(no_content, raw_score=True))
        probability = expit(named + np.clip(full-named, -CONTENT_CAP, CONTENT_CAP))
        threshold = best_validation_threshold(probability, labels[dev])
        metrics = evaluate(labels[dev], probability, threshold)
        if (metrics != prior["dev"] or threshold != prior["threshold"]):
            raise ValueError("locked model does not replay validation screen")
        folder = FOLDER / f"seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / "model.joblib"
        weights_path = folder / "source_weights.npz"
        joblib.dump(model, model_path)
        np.savez_compressed(weights_path, weights=weights,
                            fact_names=np.asarray(runtime.names))
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "reference_version": runtime.reference.version,
            "model_sha256": sha256_file(model_path),
            "weights_sha256": sha256_file(weights_path),
            "original_reference_sha256": sha256_file(
                runtime.run_dir / "label_blind_reference.json.gz"),
            "source_features_sha256": runtime.lock["source_features_sha256"],
            "source_group_names": list(runtime.group_fact_indices),
            "monotone_constraints": signs.tolist(),
            "threshold": threshold,
            "validation": metrics,
            "nonzero_fact_tail_slots": int(np.count_nonzero(weights)),
            "distinct_source_facts": prior["distinct_source_facts"],
        })
        print({"seed": seed, "dev_acc": metrics["acc"],
               "dev_macro_f1": metrics["macro_f1"]}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "sparse_18_source_group_A_selection_B_fit_dev_lock",
        "selected_top_k_per_group_direction": TOP_K,
        "selection_rule": "only candidate meeting predeclared mean 3pp Macro-F1 dev gate",
        "screen_sha256": sha256_file(SCREEN),
        "content_logit_cap": CONTENT_CAP,
        "test_prediction_or_metric_accessed": False,
        "prior_test_population_exploration": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
