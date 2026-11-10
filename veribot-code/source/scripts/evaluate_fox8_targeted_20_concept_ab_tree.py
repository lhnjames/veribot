"""Evaluate the dev-locked, source-group additive Fox8 evidence classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_20_hybrid_concepts import columns
from screen_fox8_neighbor_atomic_evidence_capacity_dev import evidence_matrix, load_cache
from screen_fox8_native_neighbor_profile_concepts_dev import weighted_atomic
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_fox8_targeted_neighbor_20_concepts"
    lock_path, dev_path = folder / "selection_lock_ab_group_tree.json", folder / "dev_only.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    raw_path = root / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
    raw_audit_path = root / "research/certificates/fox8_neighbor_volume_full_raw_source.json"
    raw_audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
    if (lock["dev_result_sha256"] != sha256_file(dev_path)
            or lock["neighbor_fact_sha256"] != sha256_file(raw_path)
            or lock["raw_source_audit_sha256"] != sha256_file(raw_audit_path)
            or raw_audit["fact_cache_sha256"] != sha256_file(raw_path)
            or raw_audit["status"] != "all_target_accounts_and_neighbor_raw_profiles_verified"
            or lock["selected"]["removed_group"] != "presentation_flags"
            or lock["selected"]["config_index"] != 0
            or lock["config"] != [15, 200, 30, 10.0]
            or lock["test_metric_or_prediction_used_to_create_lock"] is not False):
        raise ValueError("Model or full raw-source audit differs from dev lock")
    ids, new_names, new_values, supports = load_cache(raw_path)
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/fox8_23")
    if pd.read_csv(data_dir / "features.csv")["user_id"].astype(str).tolist() != ids:
        raise ValueError("Raw feature account order differs")
    groups_original = json.loads((root / "research/fox8_20_concept_groups.json")
                                 .read_text(encoding="utf-8"))
    removed = set(groups_original["presentation_flags"])
    groups = {name: members for name, members in groups_original.items()
              if name != "presentation_flags"}
    neighbor_fact = "native_interaction_neighbor_mean_statuses_count"
    groups["native_neighbor_volume"] = [neighbor_fact]
    group_names = list(groups)
    group_columns = [[2*k, 2*k+1, 20+2*k, 21+2*k] for k in range(10)]
    args = SimpleNamespace(
        dataset="fox8", data_dir=data_dir, account_file=None,
        native_graph_facts=None, structural_facts=None, neighbor_facts=None,
        fox8_native_facts=root / "data/processed/fox8_v3/native_interaction_facts.npz",
        selected_model_dir=root / "output/v3_fox8_native_spline_l2/selected")
    summary = []
    for seed in (0, 1, 2):
        prior = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                           .read_text(encoding="utf-8"))
        facts, labels, names, split = selected_facts(args, seed, prior)
        if (len(names) != 43 or prior["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]):
            raise ValueError("Source vocabulary or official split differs")
        full = np.column_stack((facts, new_values))
        all_names = names + new_names
        train, validation, test = split["train"], split["val"], split["test"]
        reference = LabelBlindReference(min_context_n=30).fit(full[train], all_names)
        if reference.version != lock["reference_version_by_seed"][str(seed)]:
            raise ValueError("Label-blind reference differs")
        kept = [j for j, name in enumerate(all_names)
                if (j < 43 and name not in removed) or name == neighbor_fact]
        batches = []
        winners = []
        for rows in (train, validation, test):
            atomic = columns(weighted_atomic(reference, full[rows], supports[rows]), kept)
            concept, winning = collapse_group_tails(atomic, groups)
            batches.append(concept)
            winners.append(winning)
        x_train, x_dev, x_test = (evidence_matrix(batch) for batch in batches)
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=labels[train], random_state=seed)
        signs = directions(x_train[a], labels[train][a])
        leaves, rounds, min_leaf, l2 = lock["config"]
        model = LGBMClassifier(
            objective="binary", n_estimators=rounds, num_leaves=leaves,
            min_child_samples=min_leaf, reg_lambda=l2, learning_rate=0.05,
            n_jobs=4, random_state=0, deterministic=True, force_col_wise=True,
            verbosity=-1, monotone_constraints=signs.tolist(),
            monotone_constraints_method="advanced",
            interaction_constraints=group_columns)
        model.fit(x_train[b], labels[train][b])
        one_group, tree_counts = all_trees_one_group(model, group_columns)
        if not one_group:
            raise ValueError("Tree crossed source groups")
        dev_probability = model.predict_proba(x_dev)[:, 1]
        expected = next(c for c in dev["runs"][seed]["candidates"]
                        if c["removed_group"] == "presentation_flags")
        expected = expected["ab_group_trees"][0]
        threshold = lock["threshold_by_seed"][str(seed)]
        if (abs(threshold - expected["selected_threshold"]) > 1e-14
                or any(abs(evaluate(labels[validation], dev_probability, threshold)[key] -
                           expected["validation_selected"][key]) > 1e-12
                       for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("Official dev replay differs before test")
        probability = model.predict_proba(x_test)[:, 1]
        logit = np.asarray(model.booster_.predict(x_test, raw_score=True))
        shap = np.asarray(model.booster_.predict(x_test, pred_contrib=True))
        if (shap.shape != (len(test), 41)
                or np.max(np.abs(shap.sum(axis=1) - logit)) > 1e-9
                or np.max(np.abs(expit(logit) - probability)) > 1e-12):
            raise ValueError("Evidence contributions fail exact score reconstruction")
        group_contribution = np.column_stack(
            [shap[:, cols].sum(axis=1) for cols in group_columns])
        if np.max(np.abs(group_contribution.sum(axis=1) + shap[:, -1] - logit)) > 1e-9:
            raise ValueError("Source-group contributions do not reconstruct score")
        neighbor_index = group_names.index("native_neighbor_volume")
        masked = x_test.copy()
        masked[:, group_columns[neighbor_index]] = 0.0
        masked_probability = model.predict_proba(masked)[:, 1]
        all_mask_probability = model.predict_proba(np.zeros_like(x_test))[:, 1]
        prediction = probability >= threshold
        baseline_path = (root /
            f"output/hypbot_fox8_native_graph_paired/seed{seed}/test_predictions.json")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if (baseline["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]
                or len(baseline["predictions"]) != len(test)
                or any(row["account_id"] != ids[int(i)] or row["label"] != int(labels[i])
                       for row, i in zip(baseline["predictions"], test))):
            raise ValueError("Paired native-graph HypBot population differs")
        baseline_prediction = np.asarray([
            row["probability"] >= baseline["manifest"]["threshold"]
            for row in baseline["predictions"]])
        run_dir = folder / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "ab_group_tree.joblib"
        reference_path = run_dir / "label_blind_reference.json.gz"
        evidence_path = run_dir / "ab_group_tree_test_evidence.npz"
        joblib.dump(model, model_path)
        reference.save(reference_path)
        np.savez_compressed(
            evidence_path, ids=np.asarray([ids[int(i)] for i in test], dtype=str),
            concept_names=np.asarray(batches[2].names, dtype=str),
            retained_fact_names=np.asarray([all_names[j] for j in kept], dtype=str),
            retained_raw_values=full[test][:, kept],
            concept_scores=batches[2].upper, model_input=x_test,
            winning_fact_index=np.column_stack([winners[2][name]
                for name in batches[2].names]),
            direction_signs=signs, group_contribution=group_contribution,
            shap=shap, logit=logit, probability=probability,
            neighbor_mask_probability=masked_probability,
            all_mask_probability=all_mask_probability,
            label=labels[test].astype(np.int8))
        predictions_path = run_dir / "ab_group_tree_test_predictions.json"
        predictions_path.write_text(json.dumps({
            "manifest": {"dataset": "fox8", "seed": seed,
                         "threshold": threshold,
                         "split_sha256": lock["split_sha256_by_seed"][str(seed)],
                         "selection_lock_sha256": sha256_file(lock_path),
                         "model_sha256": sha256_file(model_path),
                         "reference_sha256": sha256_file(reference_path),
                         "evidence_sha256": sha256_file(evidence_path)},
            "predictions": [{"account_id": ids[int(i)], "label": int(labels[i]),
                             "probability": float(p)}
                            for i, p in zip(test, probability)]}, indent=2) + "\n",
            encoding="utf-8")
        result = {
            "seed": seed, "test_accounts": len(test),
            "test": evaluate(labels[test], probability, threshold),
            "dev": evaluate(labels[validation], dev_probability, threshold),
            "same_native_graph_hypbot_acc": float(np.mean(
                baseline_prediction == labels[test])),
            "paired_only_veribot_correct": int(np.count_nonzero(
                (prediction == labels[test]) & (baseline_prediction != labels[test]))),
            "paired_only_hypbot_correct": int(np.count_nonzero(
                (prediction != labels[test]) & (baseline_prediction == labels[test]))),
            "neighbor_group_mask_changed_classes": int(np.count_nonzero(
                prediction != (masked_probability >= threshold))),
            "neighbor_group_mask_correct": int(np.count_nonzero(
                (masked_probability >= threshold) == labels[test])),
            "all_evidence_mask_changed_classes": int(np.count_nonzero(
                prediction != (all_mask_probability >= threshold))),
            "all_evidence_mask_correct": int(np.count_nonzero(
                (all_mask_probability >= threshold) == labels[test])),
            "tree_count_by_group": tree_counts,
            "max_logit_reconstruction_error": float(np.max(np.abs(
                shap.sum(axis=1) - logit))),
            "model_sha256": sha256_file(model_path),
            "reference_sha256": sha256_file(reference_path),
            "evidence_sha256": sha256_file(evidence_path),
            "predictions_sha256": sha256_file(predictions_path),
        }
        summary.append(result)
        print(json.dumps({"seed": seed, "test": result["test"],
                          "neighbor_mask_changed": result[
                              "neighbor_group_mask_changed_classes"]}), flush=True)
    report = {
        "scope": "Fox8_exploratory_fixed_test_dev_locked_20_concept_ab_group_tree",
        "prior_test_results_on_same_account_population_already_observed": True,
        "selection_lock_sha256": sha256_file(lock_path),
        "full_raw_source_audit_sha256": sha256_file(raw_audit_path),
        "concept_count": 20, "source_group_count": 10,
        "runs": summary,
        "mean_test_acc": float(np.mean([r["test"]["acc"] for r in summary])),
        "mean_native_hypbot_acc": float(np.mean([
            r["same_native_graph_hypbot_acc"] for r in summary])),
    }
    (folder / "fixed_test_ab_group_tree.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
