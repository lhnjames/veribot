"""Dev-lock and fixed-test evaluation of the smallest passing MGTAB projection group."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_reduced_projection_ab_group"
SCREEN = OUT / "dev_only.json"
LOCK = OUT / "selection_lock.json"
CAP = 3.0


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def lock() -> None:
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if (screen["status"] != "validation_only_reduced_projection_group_tree_screen"
            or screen["test_predictions_or_metrics_computed"] is not False
            or screen["test_labels_used_for_selection"] is not False
            or screen["projection_output_absolute_logit_cap"] != CAP
            or len(screen["runs"]) != 3):
        raise ValueError("Unexpected validation-only screen")
    passing = [int(k) for k in screen["gate_passing_dimensions"]]
    if not passing or not set(passing) <= {4, 8, 16}:
        raise ValueError("No reduced projection dimension passed the predeclared gate")
    dimension = min(passing)
    selected = [next(row for row in run["candidates"]
                     if row["projection_dimensions"] == dimension)
                for run in screen["runs"]]
    if any(not row["all_trees_one_source_group"] for row in selected):
        raise ValueError("Selected model crosses source groups")
    save_json(LOCK, {
        "scope": "mgtab_reduced_projection_dev_selection_before_fixed_test",
        "selection_rule": "minimum dimension satisfying predeclared validation gate",
        "screen_sha256": sha256_file(SCREEN),
        "parent_selection_lock_sha256": screen["full_150_model_selection_lock_sha256"],
        "selected_projection_dimensions": dimension,
        "selected_model_capacity": [31, 400, 50, 10.0],
        "projection_output_absolute_logit_cap": CAP,
        "threshold_by_seed": {str(run["seed"]): row["bounded_threshold"]
                              for run, row in zip(screen["runs"], selected, strict=True)},
        "split_sha256_by_seed": {str(run["seed"]): run["split_sha256"]
                                 for run in screen["runs"]},
        "reference_version_by_seed": {str(run["seed"]): run["reference_version"]
                                      for run in screen["runs"]},
        "a_b_contract_sha256_by_seed": {
            str(run["seed"]): run["direction_contract_sha256"]
            for run in screen["runs"]},
        "test_metric_or_prediction_used_to_create_lock": False,
        "prior_results_on_same_fixed_test_population_already_observed": True,
    })
    print({"selected_projection_dimensions": dimension,
           "selection_lock": str(LOCK)}, flush=True)


def bounded_scores(model: LGBMClassifier, x: np.ndarray, dimension: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = 86 + dimension
    full = np.asarray(model.booster_.predict(x, raw_score=True), dtype=float)
    named = x.copy()
    named[:, 86:d] = 0.0
    named[:, d + 86:] = 0.0
    named_logit = np.asarray(model.booster_.predict(named, raw_score=True), dtype=float)
    residual = full - named_logit
    logit = named_logit + np.clip(residual, -CAP, CAP)
    return expit(logit), logit, residual


def test() -> None:
    selection = json.loads(LOCK.read_text(encoding="utf-8"))
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if (selection["scope"] != "mgtab_reduced_projection_dev_selection_before_fixed_test"
            or selection["screen_sha256"] != sha256_file(SCREEN)
            or selection["test_metric_or_prediction_used_to_create_lock"] is not False):
        raise ValueError("Validation selection lock differs")
    dimension = int(selection["selected_projection_dimensions"])
    if dimension != min(screen["gate_passing_dimensions"]):
        raise ValueError("Selected dimension differs from dev rule")
    d = 86 + dimension
    indices = np.r_[np.arange(d), np.arange(d) + 150]
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load("mgtab", runtime.data_dir, None, seed)
        run_screen = screen["runs"][seed]
        if (run_screen["seed"] != seed
                or source["split_sha256"] != selection["split_sha256_by_seed"][str(seed)]
                or runtime.reference.version != selection["reference_version_by_seed"][str(seed)]):
            raise ValueError("Published source, split or evidence reference differs")
        with np.load(runtime.run_dir / "ab_direction_contract.npz", allow_pickle=False) as contract:
            a = contract["a_indices"].astype(int)
            b = contract["b_indices"].astype(int)
            signs = contract["monotone_constraints"].astype(int)[indices]
        if (sha256_file(runtime.run_dir / "ab_direction_contract.npz") !=
                selection["a_b_contract_sha256_by_seed"][str(seed)]
                or len(np.intersect1d(a, b))
                or not np.array_equal(np.sort(np.r_[a, b]), np.sort(split["train"]))):
            raise ValueError("A/B training split differs")
        groups = []
        for members in runtime.group_fact_indices.values():
            chosen = [j for j in members if j < d]
            if chosen:
                groups.append(chosen + [j + d for j in chosen])
        if len(groups) != 9 or sorted(j for group in groups for j in group) != list(range(2*d)):
            raise ValueError("Source groups do not partition reduced inputs")
        model = LGBMClassifier(
            objective="binary", n_estimators=400, num_leaves=31,
            min_child_samples=50, reg_lambda=10.0, learning_rate=.05,
            n_jobs=4, random_state=0, deterministic=True,
            force_col_wise=True, verbosity=-1,
            monotone_constraints=signs.tolist(),
            monotone_constraints_method="advanced",
            interaction_constraints=groups)
        x_b = evidence_design(runtime, b)[:, indices]
        model.fit(x_b, labels[b])
        one_group, counts = all_trees_one_group(model, groups)
        if not one_group:
            raise ValueError("Fitted tree crossed source groups")
        val = split["val"]
        val_probability, _, val_residual = bounded_scores(
            model, evidence_design(runtime, val)[:, indices], dimension)
        threshold = best_validation_threshold(val_probability, labels[val])
        val_metric = evaluate(labels[val], val_probability, threshold)
        expected = next(item for item in run_screen["candidates"]
                        if item["projection_dimensions"] == dimension)
        if (abs(threshold - selection["threshold_by_seed"][str(seed)]) > 1e-12
                or any(abs(val_metric[key] - expected["bounded_validation"][key]) > 1e-12
                       for key in ("acc", "macro_f1", "auroc"))
                or int(np.count_nonzero(np.abs(val_residual) > CAP)) !=
                expected["accounts_clipped_on_val"]):
            raise ValueError("Selected validation result failed exact replay")
        test_idx = split["test"]
        probability, logit, residual = bounded_scores(
            model, evidence_design(runtime, test_idx)[:, indices], dimension)
        test_metric = evaluate(labels[test_idx], probability, threshold)
        run_dir = OUT / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.joblib"
        joblib.dump(model, model_path)
        predicted = probability >= threshold
        hyp_path = ROOT / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        hyp = json.loads(hyp_path.read_text(encoding="utf-8"))
        if (hyp["manifest"]["split_sha256"] != source["split_sha256"]
                or [item["account_id"] for item in hyp["predictions"]] !=
                [ids[int(i)] for i in test_idx]
                or [item["label"] for item in hyp["predictions"]] !=
                labels[test_idx].astype(int).tolist()):
            raise ValueError("HypBot paired account or label alignment differs")
        hyp_predicted = np.asarray([
            row["probability"] >= hyp["manifest"]["threshold"]
            for row in hyp["predictions"]], dtype=bool)
        correct = predicted == labels[test_idx]
        hyp_correct = hyp_predicted == labels[test_idx]
        save_json(run_dir / "test_predictions.json", {
            "manifest": {
                "dataset": "mgtab", "seed": seed,
                "selected_projection_dimensions": dimension,
                "selection_lock_sha256": sha256_file(LOCK),
                "model_sha256": sha256_file(model_path),
                "reference_version": runtime.reference.version,
                "split_sha256": source["split_sha256"],
                "threshold": threshold,
                "projection_output_absolute_logit_cap": CAP,
            },
            "predictions": [
                {"account_id": ids[int(i)], "label": int(labels[i]),
                 "probability": float(probability[j]), "logit": float(logit[j]),
                 "projection_group_raw_logit": float(residual[j])}
                for j, i in enumerate(test_idx)],
        })
        result = {
            "seed": seed,
            "test": test_metric,
            "dev": val_metric,
            "model_sha256": sha256_file(model_path),
            "test_predictions_sha256": sha256_file(run_dir / "test_predictions.json"),
            "hypbot_predictions_sha256": sha256_file(hyp_path),
            "hypbot_acc": float(hyp_correct.mean()),
            "only_reduced_model_correct": int(np.count_nonzero(correct & ~hyp_correct)),
            "only_hypbot_correct": int(np.count_nonzero(~correct & hyp_correct)),
            "projection_group_trees": counts.get(8, 0),
            "accounts_clipped_on_test": int(np.count_nonzero(np.abs(residual) > CAP)),
            "max_absolute_projection_group_logit_on_test": float(np.max(np.abs(residual))),
            "all_trees_one_source_group": one_group,
            "above_hypbot": bool(test_metric["acc"] > hyp_correct.mean()),
            "above_user_91_2_percent": bool(test_metric["acc"] > .912),
        }
        runs.append(result)
        print(result, flush=True)
    save_json(OUT / "fixed_test_report.json", {
        "scope": "mgtab_reduced_projection_group_tree_fixed_test_after_dev_lock",
        "selection_lock_sha256": sha256_file(LOCK),
        "selected_projection_dimensions": dimension,
        "all_seeds_above_hypbot_and_91_2_percent": all(
            row["above_hypbot"] and row["above_user_91_2_percent"] for row in runs),
        "prior_results_on_same_fixed_test_population_already_observed": True,
        "runs": runs,
    })


def export() -> None:
    selection = json.loads(LOCK.read_text(encoding="utf-8"))
    parent_path = ROOT / "output/v3_mgtab_ab_group_additive_tree/inference_manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (selection["parent_selection_lock_sha256"] !=
            parent["selection_lock_sha256"]
            or parent["status"] != "inference_only_metadata_no_test_results"):
        raise ValueError("Parent source metadata differs")
    runs = []
    for seed in (0, 1, 2):
        parent_run = parent["runs"][seed]
        model = OUT / f"seed{seed}/model.joblib"
        reference = ROOT / parent_run["reference_path"]
        metadata = ROOT / parent_run["source_metadata_path"]
        contract = ROOT / parent_run["ab_contract_path"]
        if (not model.exists()
                or sha256_file(reference) != parent_run["reference_sha256"]
                or sha256_file(metadata) != parent_run["source_metadata_sha256"]
                or sha256_file(contract) !=
                selection["a_b_contract_sha256_by_seed"][str(seed)]):
            raise ValueError("Fitted model or source artifacts differ")
        runs.append({
            "seed": seed,
            "model_path": str(model.relative_to(ROOT)).replace("\\", "/"),
            "model_sha256": sha256_file(model),
            "reference_path": parent_run["reference_path"],
            "reference_sha256": parent_run["reference_sha256"],
            "reference_version": parent_run["reference_version"],
            "source_metadata_path": parent_run["source_metadata_path"],
            "source_metadata_sha256": parent_run["source_metadata_sha256"],
            "ab_contract_path": parent_run["ab_contract_path"],
            "ab_contract_sha256": parent_run["ab_contract_sha256"],
            "split_sha256": parent_run["split_sha256"],
            "threshold": selection["threshold_by_seed"][str(seed)],
        })
    save_json(OUT / "inference_manifest.json", {
        "status": "inference_only_metadata_no_test_results",
        "dataset": "mgtab",
        "variant": "reduced_projection_ab_group_additive",
        "selection_lock_sha256": sha256_file(LOCK),
        "parent_source_inference_manifest_sha256": sha256_file(parent_path),
        "source_features_sha256": parent["source_features_sha256"],
        "source_audit_sha256": parent["source_audit_sha256"],
        "selected_projection_dimensions": selection["selected_projection_dimensions"],
        "projection_output_absolute_logit_cap": CAP,
        "runs": runs,
    })
    print(OUT / "inference_manifest.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lock", "test", "export"), required=True)
    args = parser.parse_args()
    if args.stage == "lock":
        lock()
    elif args.stage == "test":
        test()
    else:
        export()


if __name__ == "__main__":
    main()
