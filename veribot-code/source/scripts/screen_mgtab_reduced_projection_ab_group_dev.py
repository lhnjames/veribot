"""Dev-only screen of fewer MGTAB projection inputs in the A/B group tree."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_reduced_projection_ab_group/dev_only.json"
LOCK = ROOT / "output/v3_mgtab_ab_group_additive_tree/selection_lock.json"
DIMENSIONS = (4, 8, 16, 32, 64)
CAP = 3.0


def candidate(runtime: MGTABFullAtomicRuntime, seed: int, dimension: int,
              b: np.ndarray, val: np.ndarray, labels: np.ndarray,
              signs_full: np.ndarray) -> dict:
    d = 86 + dimension
    fact_indices = np.r_[np.arange(86), np.arange(86, 86 + dimension)]
    encoded_indices = np.r_[fact_indices, fact_indices + 150]
    x_b = evidence_design(runtime, b)[:, encoded_indices]
    x_val = evidence_design(runtime, val)[:, encoded_indices]
    signs = signs_full[encoded_indices].astype(int)
    groups = []
    for name, members in runtime.group_fact_indices.items():
        selected = [j for j in members if j < d]
        if not selected:
            continue
        groups.append(selected + [j + d for j in selected])
    if (len(groups) != 9 or
            sorted(j for group in groups for j in group) != list(range(2*d))):
        raise ValueError("reduced evidence groups do not partition inputs")
    model = LGBMClassifier(
        objective="binary", n_estimators=400, num_leaves=31,
        min_child_samples=50, reg_lambda=10.0, learning_rate=.05,
        n_jobs=4, random_state=0, deterministic=True,
        force_col_wise=True, verbosity=-1,
        monotone_constraints=signs.tolist(),
        monotone_constraints_method="advanced",
        interaction_constraints=groups)
    model.fit(x_b, labels[b])
    single_group, tree_counts = all_trees_one_group(model, groups)
    if not single_group:
        raise ValueError("reduced tree crossed source groups")
    full_logit = np.asarray(model.booster_.predict(
        x_val, raw_score=True), dtype=float)
    named_x = x_val.copy()
    named_x[:, 86:d] = 0.0
    named_x[:, d+86:] = 0.0
    named_logit = np.asarray(model.booster_.predict(
        named_x, raw_score=True), dtype=float)
    residual = full_logit - named_logit
    bounded_logit = named_logit + np.clip(residual, -CAP, CAP)
    full_probability = expit(full_logit)
    bounded_probability = expit(bounded_logit)
    full_threshold = best_validation_threshold(full_probability, labels[val])
    bounded_threshold = best_validation_threshold(
        bounded_probability, labels[val])
    return {
        "projection_dimensions": dimension,
        "input_fact_count": d,
        "encoded_input_count": 2*d,
        "model_capacity": "31 leaves, 400 rounds, trees restricted to one source group",
        "all_trees_one_source_group": single_group,
        "tree_count_by_group": tree_counts,
        "uncapped_validation": evaluate(labels[val], full_probability,
                                         full_threshold),
        "uncapped_threshold": full_threshold,
        "bounded_validation": evaluate(labels[val], bounded_probability,
                                        bounded_threshold),
        "bounded_threshold": bounded_threshold,
        "accounts_clipped_on_val": int(np.count_nonzero(
            np.abs(residual) > CAP)),
        "projection_changes_named_only_class_on_val": int(
            np.count_nonzero((bounded_probability >= bounded_threshold) !=
                             (expit(named_logit) >= bounded_threshold))),
        "max_absolute_projection_group_logit_on_val": float(np.max(
            np.abs(residual))),
        "mean_absolute_projection_group_logit_on_val": float(np.mean(
            np.abs(residual))),
    }


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["selected_config"] != [31, 400, 50, 10.0]:
        raise ValueError("MGTAB capacity lock differs")
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed,
                                         variant="ab_group_additive")
        _, labels, ids, _, split, source = load(
            "mgtab", runtime.data_dir, None, seed)
        if (source["split_sha256"] != lock["split_sha256_by_seed"][str(seed)]
                or runtime.reference.version !=
                lock["reference_version_by_seed"][str(seed)]):
            raise ValueError("fixed reference or official split differs")
        with np.load(runtime.run_dir / "ab_direction_contract.npz",
                     allow_pickle=False) as contract:
            a = contract["a_indices"].astype(int)
            b = contract["b_indices"].astype(int)
            signs = contract["monotone_constraints"].astype(int)
        if (set(a) & set(b) or sorted(np.r_[a, b].tolist()) !=
                sorted(split["train"].tolist()) or len(signs) != 300):
            raise ValueError("A/B training contract differs")
        val = split["val"]
        original_matrix = evidence_design(runtime, val)
        original_logit = np.asarray(runtime.model.booster_.predict(
            original_matrix, raw_score=True), dtype=float)
        original_probability = expit(original_logit)
        original = evaluate(labels[val], original_probability,
                            runtime.threshold)
        choices = [candidate(runtime, seed, k, b, val, labels, signs)
                   for k in DIMENSIONS]
        full = choices[-1]
        if (abs(full["uncapped_threshold"] - runtime.threshold) > 1e-12
                or any(abs(full["uncapped_validation"][key] - original[key])
                       > 1e-12 for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("64-coordinate control did not replay frozen model")
        runs.append({
            "seed": seed,
            "split_sha256": source["split_sha256"],
            "reference_version": runtime.reference.version,
            "model_sha256": sha256_file(runtime.run_dir / "model.joblib"),
            "direction_contract_sha256": sha256_file(
                runtime.run_dir / "ab_direction_contract.npz"),
            "original_frozen_validation": original,
            "candidates": choices,
        })
        print({"seed": seed,
               "dev_macro_f1_by_k": {row["projection_dimensions"]:
                   round(row["bounded_validation"]["macro_f1"], 5)
                   for row in choices}}, flush=True)
    full_mean = float(np.mean([run["candidates"][-1][
        "bounded_validation"]["macro_f1"] for run in runs]))
    passing = []
    for k in DIMENSIONS[:-1]:
        rows = [next(item for item in run["candidates"]
                     if item["projection_dimensions"] == k) for run in runs]
        mean = float(np.mean([row["bounded_validation"]["macro_f1"]
                              for row in rows]))
        if (k <= 16 and mean >= full_mean - .005 and all(
                row["bounded_validation"]["acc"] >=
                run["candidates"][-1]["bounded_validation"]["acc"] - .01
                for row, run in zip(rows, runs, strict=True))):
            passing.append(k)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "validation_only_reduced_projection_group_tree_screen",
        "test_predictions_or_metrics_computed": False,
        "test_labels_loaded_for_official_split": True,
        "test_labels_used_for_selection": False,
        "dimension_grid": DIMENSIONS,
        "projection_output_absolute_logit_cap": CAP,
        "gate": "k<=16; mean dev Macro-F1 within 0.5 point of 64-coordinate capped control; every seed dev ACC within 1 point",
        "gate_passing_dimensions": passing,
        "full_150_model_selection_lock_sha256": sha256_file(LOCK),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print({"gate_passing_dimensions": passing})
    print(OUT)


if __name__ == "__main__":
    main()
