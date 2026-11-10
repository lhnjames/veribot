"""Evaluate the dev-locked source-present Human evidence tree on fixed MGTAB tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_ab_group_additive_monotone_tree_dev import (
    all_trees_one_group, source_groups)
from screen_mgtab_ab_group_positive_human_dev import (
    ANCHOR_RATIO, fit_rows, with_positive_human)
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.mgtab_full_atomic_runtime import AB_RELATIONS
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_mgtab_ab_group_positive_human"


def source_metadata(seed: int, lock: dict) -> dict:
    path = (ROOT /
        f"output/v3_current_compact_deployment/mgtab_source_seed{seed}.json")
    source = json.loads(path.read_text(encoding="utf-8"))
    names = source["source_fact_names"]
    groups = {
        "published_profile_coordinates": names[:20],
        **{f"published_relation_{k}": [
            name for name in names[20:86]
            if name.startswith(f"published_{relation}_")]
            for k, relation in enumerate(AB_RELATIONS)},
        "published_tweet_projections": names[86:],
    }
    return {"manifest": {
                "split_sha256": lock["split_sha256_by_seed"][str(seed)],
                "projection_fingerprint": source["manifest"][
                    "projection_fingerprint"]},
            "reasoner": {"directions": names, "groups": groups}}


def main() -> None:
    lock_path = FOLDER / "selection_lock.json"
    dev_path = FOLDER / "dev_only.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    data_dir = resolve_mgtab_source_dir(ROOT)
    if (lock["selected"] is not True or
            lock["test_metric_or_prediction_used_to_create_lock"] is not False or
            lock["dev_result_sha256"] != sha256_file(dev_path) or
            lock["anchor_ratio"] != ANCHOR_RATIO or
            lock["model_config"] != [31, 400, 50, 10.0] or
            lock["source_features_sha256"] != sha256_file(data_dir / "features.pt")):
        raise ValueError("positive Human model was not dev-locked")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=ROOT / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=ROOT / "output/v3_mgtab_ab_spline_content/selected")
    runs = []
    for seed in (0, 1, 2):
        prior = source_metadata(seed, lock)
        facts, labels, names, split = selected_facts(args, seed, prior)
        train, validation, test = (split[k] for k in ("train", "val", "test"))
        reference = LabelBlindReference(min_context_n=30).fit(
            facts[train], names)
        if reference.version != lock["reference_version_by_seed"][str(seed)]:
            raise ValueError("reference differs from dev lock")
        groups = source_groups(names, prior["reasoner"]["groups"])
        train_batch = transform_with_support(reference, facts[train],
                                             split["supports"][train])
        val_batch = transform_with_support(reference, facts[validation],
                                           split["supports"][validation])
        test_batch = transform_with_support(reference, facts[test],
                                            split["supports"][test])
        x_train = with_positive_human(train_batch, groups)
        x_val = with_positive_human(val_batch, groups)
        x_test = with_positive_human(test_batch, groups)
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=labels[train], random_state=seed)
        constraints = np.r_[directions(x_train[a, :300], labels[train][a]),
                            -np.ones(9, dtype=np.int8)]
        source_columns = [members + [300+k]
                          for k, members in enumerate(groups)]
        fit_x, fit_y, weights = fit_rows(x_train[b], labels[train][b])
        model = LGBMClassifier(
            objective="binary", n_estimators=400, num_leaves=31,
            min_child_samples=50, reg_lambda=10.0, learning_rate=0.05,
            n_jobs=4, random_state=0, deterministic=True, force_col_wise=True,
            monotone_constraints=constraints.tolist(),
            monotone_constraints_method="advanced",
            interaction_constraints=source_columns, verbosity=-1)
        model.fit(fit_x, fit_y, sample_weight=weights)
        one_group, tree_counts = all_trees_one_group(model, source_columns)
        if not one_group:
            raise ValueError("positive Human tree crossed a source group")
        val_p = model.predict_proba(x_val)[:, 1]
        threshold = best_validation_threshold(val_p, labels[validation])
        expected_val = dev["runs"][seed]["validation"]
        actual_val = evaluate(labels[validation], val_p, threshold)
        if (abs(threshold-lock["threshold_by_seed"][str(seed)]) > 1e-12 or
                any(abs(actual_val[key]-expected_val[key]) > 1e-12
                    for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("frozen training does not replay dev choice")
        p = model.predict_proba(x_test)[:, 1]
        raw = np.asarray(model.booster_.predict(x_test, raw_score=True))
        shap = np.asarray(model.booster_.predict(x_test, pred_contrib=True))
        if (shap.shape != (len(test), 310) or
                np.max(np.abs(shap.sum(axis=1)-raw)) > 1e-9 or
                np.max(np.abs(expit(raw)-p)) > 1e-12):
            raise ValueError("new evidence cannot reconstruct prediction")
        pred = p >= threshold
        without = x_test.copy()
        without[:, 300:] = 0
        without_p = model.predict_proba(without)[:, 1]
        without_pred = without_p >= threshold
        zero_p = float(model.predict_proba(np.zeros((1, 309)))[:, 1][0])
        full_margin = np.where(pred, p-threshold, threshold-p)
        zero_margin = np.where(pred, zero_p-threshold, threshold-zero_p)
        zero_sufficient = int(np.count_nonzero(
            zero_margin >= 0.9*full_margin-1e-12))
        baseline = json.loads((ROOT /
            f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json")
            .read_text(encoding="utf-8"))
        baseline_rows = baseline["predictions"]
        if (baseline["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)] or
                len(baseline_rows) != len(test) or
                any(row["account_id"] != str(int(i)) or
                    row["label"] != int(labels[i])
                    for row, i in zip(baseline_rows, test, strict=True))):
            raise ValueError("HypBot paired population differs")
        baseline_pred = np.array([
            row["probability"] >= baseline["manifest"]["threshold"]
            for row in baseline_rows])
        run_dir = FOLDER / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.joblib"
        ref_path = run_dir / "label_blind_reference.json.gz"
        contract_path = run_dir / "ab_direction_contract.npz"
        joblib.dump(model, model_path)
        reference.save(ref_path)
        np.savez_compressed(contract_path, a_indices=train[a], b_indices=train[b],
                            monotone_constraints=constraints,
                            source_group_names=np.asarray(
                                list(prior["reasoner"]["groups"]), dtype=str))
        predictions_path = run_dir / "test_predictions.json"
        predictions_path.write_text(json.dumps({
            "manifest": {
                "dataset": "mgtab", "seed": seed,
                "threshold": threshold,
                "split_sha256": lock["split_sha256_by_seed"][str(seed)],
                "selection_lock_sha256": sha256_file(lock_path),
                "reference_version": reference.version,
                "model_sha256": sha256_file(model_path),
                "reference_sha256": sha256_file(ref_path),
                "ab_contract_sha256": sha256_file(contract_path),
            },
            "predictions": [{"account_id": str(int(i)),
                             "label": int(labels[i]),
                             "probability": float(value)}
                            for i, value in zip(test, p, strict=True)],
        }, indent=2) + "\n", encoding="utf-8")
        row = {
            "seed": seed, "test_accounts": len(test),
            "test": evaluate(labels[test], p, threshold),
            "dev": actual_val,
            "hypbot_test_acc": float(np.mean(baseline_pred == labels[test])),
            "only_new_correct": int(np.count_nonzero(
                (pred == labels[test]) & (baseline_pred != labels[test]))),
            "only_hypbot_correct": int(np.count_nonzero(
                (pred != labels[test]) & (baseline_pred == labels[test]))),
            "positive_human_mask_changed_classes": int(
                np.count_nonzero(without_pred != pred)),
            "positive_human_mask_net_correct_loss": int(
                np.count_nonzero(pred == labels[test]) -
                np.count_nonzero(without_pred == labels[test])),
            "zero_evidence_probability": zero_p,
            "zero_sufficient_90_percent_margin_accounts": zero_sufficient,
            "all_trees_use_one_source_group": one_group,
            "tree_count_by_group": tree_counts,
            "max_logit_reconstruction_error": float(
                np.max(np.abs(shap.sum(axis=1)-raw))),
            "model_sha256": sha256_file(model_path),
            "reference_sha256": sha256_file(ref_path),
            "contract_sha256": sha256_file(contract_path),
            "predictions_sha256": sha256_file(predictions_path),
        }
        runs.append(row)
        print(json.dumps({"seed": seed, "test_acc": row["test"]["acc"],
                          "hypbot_acc": row["hypbot_test_acc"],
                          "human_mask_net_loss": row[
                              "positive_human_mask_net_correct_loss"],
                          "zero_sufficient": zero_sufficient}), flush=True)
    report = {
        "scope": "dev_locked_mgtab_ab_additive_positive_human_fixed_test",
        "selection_lock_sha256": sha256_file(lock_path),
        "prior_results_on_same_test_population_already_observed": True,
        "runs": runs,
        "mean_test_acc": float(np.mean([r["test"]["acc"] for r in runs])),
        "mean_hypbot_test_acc": float(np.mean([
            r["hypbot_test_acc"] for r in runs])),
    }
    path = FOLDER / "fixed_test_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
