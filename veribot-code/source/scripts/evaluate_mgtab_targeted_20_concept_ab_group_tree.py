"""Fixed-test evaluation of the dev-locked compact MGTAB evidence model."""

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
from gate0_shallow_tree import evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_20_hybrid_concepts import columns
from screen_mgtab_20_train_selected_concepts import rank_content
from screen_mgtab_ab_monotone_evidence_tree_dev import directions, evidence_matrix
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_content_projection_evidence import transform_with_support
from screen_mgtab_targeted_20_concept_ab_group_tree_dev import grouped
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_targeted_20_concept_ab_group_tree"
    lock_path, dev_path = folder / "selection_lock.json", folder / "dev_only.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    if (lock["dev_result_sha256"] != sha256_file(dev_path)
            or lock["source_features_sha256"] != sha256_file(data_dir / "features.pt")
            or lock["selected"]["scheme"] !=
            "four_profile_two_relation_four_selected_content"
            or lock["selected"]["config_index"] != 1
            or lock["config"] != [31, 400, 50, 10.0]
            or lock["test_metric_or_prediction_used_to_create_lock"] is not False):
        raise ValueError("Compact MGTAB candidate is not dev locked")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=root / "output/v3_mgtab_ab_spline_content/selected")
    summary = []
    for seed in (0, 1, 2):
        old = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        if (x.shape != (10199, 150)
                or old["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]):
            raise ValueError("Source population or official split differs")
        train, validation, test = split["train"], split["val"], split["test"]
        reference = LabelBlindReference(min_context_n=30).fit(x[train], names)
        if reference.version != lock["reference_version_by_seed"][str(seed)]:
            raise ValueError("Train-only reference differs")
        atomic_train = transform_with_support(reference, x[train],
                                              split["supports"][train])
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=y[train], random_state=seed)
        content = rank_content(atomic_train, y[train], a, count=4)
        if [names[j] for j in content] != lock["selected_content_by_seed"][str(seed)]:
            raise ValueError("Training-A content selection differs")
        kept, groups = grouped(names, lock["selected"]["scheme"], content)
        indices = [[2*k, 2*k+1, 20+2*k, 21+2*k] for k in range(10)]
        batches = []
        winners = []
        for rows in (train, validation, test):
            atomic = transform_with_support(reference, x[rows],
                                            split["supports"][rows])
            concept, winning = collapse_group_tails(columns(atomic, kept), groups)
            batches.append(concept)
            winners.append(winning)
        x_train, x_val, x_test = (evidence_matrix(batch) for batch in batches)
        signs = directions(x_train[a], y[train][a])
        leaves, rounds, min_leaf, l2 = lock["config"]
        model = LGBMClassifier(
            objective="binary", n_estimators=rounds, num_leaves=leaves,
            min_child_samples=min_leaf, reg_lambda=l2, learning_rate=0.05,
            n_jobs=4, random_state=0, deterministic=True, force_col_wise=True,
            monotone_constraints=signs.tolist(),
            monotone_constraints_method="advanced",
            interaction_constraints=indices, verbosity=-1)
        model.fit(x_train[b], y[train][b])
        one_group, tree_counts = all_trees_one_group(model, indices)
        if not one_group:
            raise ValueError("MGTAB tree crossed concept source groups")
        val_probability = model.predict_proba(x_val)[:, 1]
        expected = next(c for c in dev["runs"][seed]["candidates"]
                        if c["scheme"] == lock["selected"]["scheme"])
        expected = expected["variants"][lock["selected"]["config_index"]]
        threshold = lock["threshold_by_seed"][str(seed)]
        if (abs(threshold - expected["threshold"]) > 1e-12
                or any(abs(evaluate(y[validation], val_probability, threshold)[key] -
                           expected["validation"][key]) > 1e-12
                       for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("Official validation result failed exact replay")
        probability = model.predict_proba(x_test)[:, 1]
        logit = np.asarray(model.booster_.predict(x_test, raw_score=True))
        shap = np.asarray(model.booster_.predict(x_test, pred_contrib=True))
        if (shap.shape != (len(test), 41)
                or np.max(np.abs(shap.sum(axis=1) - logit)) > 1e-9
                or np.max(np.abs(expit(logit) - probability)) > 1e-12):
            raise ValueError("Evidence contributions fail score replay")
        zero_logit = np.asarray(model.booster_.predict(np.zeros_like(x_test),
                                                       raw_score=True))
        zero_probability = model.predict_proba(np.zeros_like(x_test))[:, 1]
        mask_probability = np.empty((len(test), 10))
        exact_group_delta = np.empty((len(test), 10))
        group_shap = np.column_stack([shap[:, cols].sum(axis=1) for cols in indices])
        for k, cols in enumerate(indices):
            masked = x_test.copy()
            masked[:, cols] = 0
            mask_probability[:, k] = model.predict_proba(masked)[:, 1]
            masked_logit = np.asarray(model.booster_.predict(masked, raw_score=True))
            exact_group_delta[:, k] = logit - masked_logit
        group_additivity = float(np.max(np.abs(
            logit - zero_logit - exact_group_delta.sum(axis=1))))
        if group_additivity > 1e-9:
            raise ValueError("Source-group deletion effects are not additive")
        baseline_path = root / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if (baseline["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]
                or len(baseline["predictions"]) != len(test)
                or any(item["account_id"] != str(int(i)) or
                       item["label"] != int(y[i])
                       for item, i in zip(baseline["predictions"], test))):
            raise ValueError("Paired HypBot account population differs")
        baseline_prediction = np.asarray([
            item["probability"] >= baseline["manifest"]["threshold"]
            for item in baseline["predictions"]])
        prediction = probability >= threshold
        run_dir = folder / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.joblib"
        ref_path = run_dir / "reference.json.gz"
        evidence_path = run_dir / "test_evidence.npz"
        joblib.dump(model, model_path)
        reference.save(ref_path)
        np.savez_compressed(
            evidence_path, ids=np.asarray([str(int(i)) for i in test], dtype=str),
            retained_fact_names=np.asarray([names[j] for j in kept], dtype=str),
            retained_raw_values=x[test][:, kept],
            neighbor_supports=split["supports"][test],
            concept_names=np.asarray(batches[2].names, dtype=str),
            concept_scores=batches[2].upper,
            concept_winners=np.column_stack([winners[2][name]
                for name in batches[2].names]),
            evidence_matrix=x_test, direction_signs=signs,
            probability=probability, logit=logit, shap=shap,
            group_shap=group_shap, exact_group_delta=exact_group_delta,
            group_mask_probability=mask_probability,
            all_mask_probability=zero_probability, label=y[test].astype(np.int8))
        prediction_path = run_dir / "test_predictions.json"
        prediction_path.write_text(json.dumps({
            "manifest": {"dataset": "mgtab", "seed": seed,
                         "threshold": threshold,
                         "split_sha256": lock["split_sha256_by_seed"][str(seed)],
                         "selection_lock_sha256": sha256_file(lock_path),
                         "model_sha256": sha256_file(model_path),
                         "reference_sha256": sha256_file(ref_path),
                         "evidence_sha256": sha256_file(evidence_path)},
            "predictions": [{"account_id": str(int(i)), "label": int(y[i]),
                             "probability": float(p)}
                            for i, p in zip(test, probability)]}, indent=2) + "\n",
            encoding="utf-8")
        group_names = list(groups)
        row = {
            "seed": seed, "test_accounts": len(test),
            "test": evaluate(y[test], probability, threshold),
            "dev": evaluate(y[validation], val_probability, threshold),
            "same_account_hypbot_acc": float(np.mean(
                baseline_prediction == y[test])),
            "paired_only_veribot_correct": int(np.count_nonzero(
                (prediction == y[test]) & (baseline_prediction != y[test]))),
            "paired_only_hypbot_correct": int(np.count_nonzero(
                (prediction != y[test]) & (baseline_prediction == y[test]))),
            "all_evidence_mask_changed_classes": int(np.count_nonzero(
                prediction != (zero_probability >= threshold))),
            "all_evidence_mask_correct": int(np.count_nonzero(
                (zero_probability >= threshold) == y[test])),
            "group_mask_changed_classes": {
                name: int(np.count_nonzero(prediction !=
                     (mask_probability[:, k] >= threshold)))
                for k, name in enumerate(group_names)},
            "group_mask_correct": {
                name: int(np.count_nonzero((mask_probability[:, k] >= threshold) == y[test]))
                for k, name in enumerate(group_names)},
            "retained_fact_count": len(kept), "concept_count": 20,
            "content_selection": lock["selected_content_by_seed"][str(seed)],
            "tree_count_by_group": tree_counts,
            "max_logit_additivity_error": float(np.max(np.abs(
                shap.sum(axis=1) - logit))),
            "max_group_deletion_additivity_error": group_additivity,
            "model_sha256": sha256_file(model_path),
            "reference_sha256": sha256_file(ref_path),
            "evidence_sha256": sha256_file(evidence_path),
            "predictions_sha256": sha256_file(prediction_path),
        }
        summary.append(row)
        print(json.dumps({"seed": seed, "test": row["test"],
                          "hypbot_acc": row["same_account_hypbot_acc"],
                          "group_masks": row["group_mask_changed_classes"]}), flush=True)
    report = {
        "scope": "mgtab_exploratory_fixed_test_dev_locked_twenty_concept_ab_group_tree",
        "prior_test_results_on_same_account_population_already_observed": True,
        "selection_lock_sha256": sha256_file(lock_path),
        "source_features_sha256": sha256_file(data_dir / "features.pt"),
        "concept_count": 20, "source_group_count": 10,
        "content_provenance_level": "published_average_tweet_tensor",
        "runs": summary,
        "mean_test_acc": float(np.mean([r["test"]["acc"] for r in summary])),
        "mean_hypbot_acc": float(np.mean([r["same_account_hypbot_acc"]
                                           for r in summary])),
    }
    (folder / "fixed_test_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
