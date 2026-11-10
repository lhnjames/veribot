"""Evaluate the dev-locked source-present Human evidence candidate once on test."""

from __future__ import annotations

import json
import sys
import warnings
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
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_content_projection_evidence import transform_with_support
from screen_mgtab_positive_human_evidence_dev import (
    neutral_training_rows, positive_human_matrix)
from screen_mgtab_targeted_20_concept_ab_group_tree_dev import grouped
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_mgtab_positive_human_evidence"
BASE = ROOT / "output/v3_mgtab_targeted_20_concept_ab_group_tree"
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")


def main() -> None:
    lock_path = FOLDER / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock["test_metric_or_prediction_used_to_create_lock"] is not False or
            lock["dev_screen_sha256"] != sha256_file(FOLDER / "dev_only.json") or
            lock["base_selection_lock_sha256"] != sha256_file(BASE / "selection_lock.json") or
            lock["dev_screen_script_sha256"] != sha256_file(
                ROOT / "source/scripts/screen_mgtab_positive_human_evidence_dev.py") or
            lock["selected"]["ratio"] != 0.05 or
            lock["scheme"] != "four_profile_two_relation_four_selected_content" or
            lock["model_config"] != [31, 400, 50, 10.0]):
        raise ValueError("validation-only selection lock differs")
    args = SimpleNamespace(
        dataset="mgtab",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB"),
        account_file=None, native_graph_facts=None,
        structural_facts=ROOT / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=ROOT / "output/v3_mgtab_ab_spline_content/selected")
    baseline_lock = json.loads((BASE / "selection_lock.json")
                               .read_text(encoding="utf-8"))
    if (lock["split_sha256_by_seed"] != baseline_lock["split_sha256_by_seed"] or
            lock["selected_content_by_seed"] != baseline_lock["selected_content_by_seed"]):
        raise ValueError("paired split or selected content differs")
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        facts, labels, names, split = selected_facts(args, seed, old)
        if old["manifest"]["split_sha256"] != lock["split_sha256_by_seed"][str(seed)]:
            raise ValueError("official population differs")
        train, dev, test = split["train"], split["val"], split["test"]
        reference = LabelBlindReference(min_context_n=30).fit(facts[train], names)
        if reference.version != lock["reference_version_by_seed"][str(seed)]:
            raise ValueError("train-only reference differs")
        atomic_train = transform_with_support(reference, facts[train],
                                              split["supports"][train])
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=labels[train], random_state=seed)
        content = rank_content(atomic_train, labels[train], a, count=4)
        if [names[j] for j in content] != lock["selected_content_by_seed"][str(seed)]:
            raise ValueError("train-A content selection differs")
        kept, groups = grouped(names, lock["scheme"], content)
        matrices = []
        for indices in (train, dev, test):
            atomic = transform_with_support(reference, facts[indices],
                                            split["supports"][indices])
            concepts, _ = collapse_group_tails(columns(atomic, kept), groups)
            matrices.append(positive_human_matrix(concepts))
        x_train, x_dev, x_test = matrices
        signs = np.r_[directions(x_train[a, :40], labels[train][a]),
                      -np.ones(10, dtype=np.int8)]
        source_cols = [[2*k, 2*k+1, 20+2*k, 21+2*k, 40+k]
                       for k in range(10)]
        fit_x, fit_y, weights = neutral_training_rows(
            x_train[b], labels[train][b], lock["selected"]["ratio"])
        model = LGBMClassifier(
            objective="binary", n_estimators=400, num_leaves=31,
            min_child_samples=50, reg_lambda=10.0, learning_rate=0.05,
            n_jobs=4, random_state=0, deterministic=True, force_col_wise=True,
            monotone_constraints=signs.tolist(),
            monotone_constraints_method="advanced",
            interaction_constraints=source_cols, verbosity=-1)
        model.fit(fit_x, fit_y, sample_weight=weights)
        one_group, tree_counts = all_trees_one_group(model, source_cols)
        if not one_group:
            raise ValueError("tree crossed source groups")
        dev_probability = model.predict_proba(x_dev)[:, 1]
        threshold = float(lock["selected"]["thresholds"][str(seed)])
        expected = next(v for v in json.loads((FOLDER / "dev_only.json")
            .read_text(encoding="utf-8"))["runs"][seed]["alternatives"]
            if v["neutral_anchor_weight_ratio"] == 0.05)
        dev_metrics = evaluate(labels[dev], dev_probability, threshold)
        if any(abs(dev_metrics[key]-expected["validation"][key]) > 1e-12
               for key in ("acc", "macro_f1", "auroc")):
            raise ValueError("dev selection did not replay")
        probability = model.predict_proba(x_test)[:, 1]
        raw = np.asarray(model.booster_.predict(x_test, raw_score=True))
        shap = np.asarray(model.booster_.predict(x_test, pred_contrib=True))
        if (shap.shape != (len(test), 51) or
                np.max(np.abs(shap.sum(axis=1)-raw)) > 1e-9 or
                np.max(np.abs(expit(raw)-probability)) > 1e-12):
            raise ValueError("source evidence does not reconstruct the score")
        zero_probability = float(model.predict_proba(
            np.zeros((1, x_test.shape[1])))[:, 1][0])
        full_prediction = probability >= threshold
        margin = np.where(full_prediction, probability-threshold,
                          threshold-probability)
        zero_margin = np.where(full_prediction, zero_probability-threshold,
                               threshold-zero_probability)
        zero_sufficient = int(np.count_nonzero(zero_margin >= .9*margin-1e-12))
        hypbot_path = ROOT / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        hypbot = json.loads(hypbot_path.read_text(encoding="utf-8"))
        if (hypbot["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)] or
                any(item["account_id"] != str(int(i)) or
                    item["label"] != int(labels[i])
                    for item, i in zip(hypbot["predictions"], test))):
            raise ValueError("HypBot paired population differs")
        hypbot_pred = np.array([item["probability"] >=
                                hypbot["manifest"]["threshold"]
                                for item in hypbot["predictions"]])
        group_mask_change = {}
        for k, group in enumerate(groups):
            masked = x_test.copy()
            masked[:, source_cols[k]] = 0.0
            masked_pred = model.predict_proba(masked)[:, 1] >= threshold
            group_mask_change[group] = int(np.count_nonzero(
                masked_pred != full_prediction))
        run_dir = FOLDER / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.joblib"
        evidence_path = run_dir / "test_evidence.npz"
        joblib.dump(model, model_path)
        np.savez_compressed(evidence_path,
                            ids=np.array([str(int(i)) for i in test]),
                            model_input=x_test,
                            label=labels[test].astype(np.int8),
                            probability=probability,
                            threshold=np.array(threshold),
                            source_group_names=np.array(list(groups)),
                            direction_signs=signs)
        run = {"seed": seed, "test_accounts": len(test),
               "split_sha256": lock["split_sha256_by_seed"][str(seed)],
               "selected_lock_sha256": sha256_file(lock_path),
               "model_sha256": sha256_file(model_path),
               "evidence_sha256": sha256_file(evidence_path),
               "test": evaluate(labels[test], probability, threshold),
               "dev": dev_metrics,
               "threshold": threshold,
               "zero_evidence_probability": zero_probability,
               "zero_evidence_90_percent_margin_accounts": zero_sufficient,
               "same_account_hypbot_acc": float(np.mean(hypbot_pred == labels[test])),
               "candidate_only_correct": int(np.count_nonzero(
                   (full_prediction == labels[test]) &
                   (hypbot_pred != labels[test]))),
               "hypbot_only_correct": int(np.count_nonzero(
                   (full_prediction != labels[test]) &
                   (hypbot_pred == labels[test]))),
               "group_mask_changed_classes": group_mask_change,
               "source_group_tree_counts": tree_counts,
               "max_shap_score_replay_error": float(np.max(np.abs(
                   shap.sum(axis=1)-raw)))}
        runs.append(run)
        print(json.dumps({"seed": seed, "acc": run["test"]["acc"],
                          "macro_f1": run["test"]["macro_f1"],
                          "hypbot_acc": run["same_account_hypbot_acc"],
                          "zero_sufficient": zero_sufficient}), flush=True)
    report = {"scope": "mgtab_dev_locked_positive_human_source_evidence_fixed_test",
              "prior_exploration_on_same_test_population": True,
              "test_used_for_model_or_ratio_selection": False,
              "source_limit": "published profile, typed relations, and mean tweet tensors",
              "selection_lock_sha256": sha256_file(lock_path),
              "runs": runs}
    path = FOLDER / "fixed_test_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
