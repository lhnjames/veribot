"""Dev-only screen of source-present Human evidence for the high-ACC MGTAB tree."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import lightgbm
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_ab_group_additive_monotone_tree_dev import (
    all_trees_one_group, source_groups)
from screen_mgtab_ab_monotone_evidence_tree_dev import directions, evidence_matrix
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir
from verifiable_bot.mgtab_full_atomic_runtime import AB_RELATIONS
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output/v3_mgtab_ab_group_additive_tree"
OUT = ROOT / "output/v3_mgtab_ab_group_positive_human/dev_only.json"
ANCHOR_RATIO = 0.05


def with_positive_human(batch, groups: list[list[int]]) -> np.ndarray:
    base = evidence_matrix(batch)
    if base.shape[1] != 300 or len(groups) != 9:
        raise ValueError("expected 150 source facts and nine source groups")
    normality = []
    for members in groups:
        facts = [j for j in members if j < 150]
        present = np.any(batch.available[:, facts], axis=1)
        deviation = np.max(base[:, members], axis=1)
        normality.append(np.where(present, np.exp(-deviation), 0.0))
    return np.column_stack((base, np.column_stack(normality)))


def fit_rows(x: np.ndarray, y: np.ndarray):
    anchor = np.zeros((100, x.shape[1]), dtype=float)
    labels = np.r_[np.zeros(50, dtype=y.dtype),
                   np.ones(50, dtype=y.dtype)]
    weights = np.r_[np.ones(len(x)),
                    np.full(100, ANCHOR_RATIO*len(x)/100)]
    return np.vstack((x, anchor)), np.r_[y, labels], weights


def main() -> None:
    data_dir = resolve_mgtab_source_dir(ROOT)
    lock_path = BASE / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    baseline_path = BASE / "dev_only.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (lock["test_metric_or_prediction_used_to_create_lock"] is not False or
            lock["selected_config"] != [31, 400, 50, 10.0] or
            lock["source_features_sha256"] != sha256_file(data_dir / "features.pt") or
            lock["dev_result_sha256"] != sha256_file(baseline_path)):
        raise ValueError("MGTAB A/B train/dev selection differs")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=ROOT / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=ROOT / "output/v3_mgtab_ab_spline_content/selected")
    runs = []
    for seed in (0, 1, 2):
        source_meta_path = (ROOT /
            f"output/v3_current_compact_deployment/mgtab_source_seed{seed}.json")
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        vocabulary = source_meta["source_fact_names"]
        if len(vocabulary) != 150:
            raise ValueError("published source vocabulary differs")
        group_members = {
            "published_profile_coordinates": vocabulary[:20],
            **{f"published_relation_{k}": [
                name for name in vocabulary[20:86]
                if name.startswith(f"published_{relation}_")]
                for k, relation in enumerate(AB_RELATIONS)},
            "published_tweet_projections": vocabulary[86:],
        }
        prior = {
            "manifest": {
                "split_sha256": lock["split_sha256_by_seed"][str(seed)],
                "projection_fingerprint": source_meta["manifest"][
                    "projection_fingerprint"],
            },
            "reasoner": {"directions": vocabulary, "groups": group_members},
        }
        facts, labels, names, split = selected_facts(args, seed, prior)
        if names != vocabulary:
            raise ValueError("source vocabulary differs from sealed metadata")
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(
            facts[train], names)
        if reference.version != lock["reference_version_by_seed"][str(seed)]:
            raise ValueError("train reference differs")
        groups = source_groups(names, prior["reasoner"]["groups"])
        train_batch = transform_with_support(reference, facts[train],
                                             split["supports"][train])
        dev_batch = transform_with_support(reference, facts[dev],
                                           split["supports"][dev])
        x_train = with_positive_human(train_batch, groups)
        x_dev = with_positive_human(dev_batch, groups)
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=labels[train], random_state=seed)
        signs = np.r_[directions(x_train[a, :300], labels[train][a]),
                      -np.ones(9, dtype=np.int8)]
        source_columns = [members + [300+k]
                          for k, members in enumerate(groups)]
        fit_x, fit_y, weights = fit_rows(x_train[b], labels[train][b])
        model = LGBMClassifier(
            objective="binary", n_estimators=400, num_leaves=31,
            min_child_samples=50, reg_lambda=10.0, learning_rate=0.05,
            n_jobs=4, random_state=0, deterministic=True, force_col_wise=True,
            monotone_constraints=signs.tolist(),
            monotone_constraints_method="advanced",
            interaction_constraints=source_columns, verbosity=-1)
        model.fit(fit_x, fit_y, sample_weight=weights)
        single_group, tree_counts = all_trees_one_group(model, source_columns)
        if not single_group:
            raise ValueError("new MGTAB tree crossed source groups")
        p = model.predict_proba(x_dev)[:, 1]
        threshold = best_validation_threshold(p, labels[dev])
        zero_p = float(model.predict_proba(np.zeros((1, 309)))[:, 1][0])
        pred = p >= threshold
        margin = np.where(pred, p-threshold, threshold-p)
        zero_margin = np.where(pred, zero_p-threshold, threshold-zero_p)
        zero_sufficient = int(np.count_nonzero(
            zero_margin >= 0.9*margin-1e-12))
        without_human = x_dev.copy()
        without_human[:, 300:] = 0
        without_p = model.predict_proba(without_human)[:, 1]
        changed = (without_p >= threshold) != pred
        full_correct = pred == labels[dev]
        without_correct = (without_p >= threshold) == labels[dev]
        baseline_metric = baseline["runs"][seed]["candidates"][1][
            "validation_selected"]
        row = {
            "seed": seed,
            "split_sha256": lock["split_sha256_by_seed"][str(seed)],
            "reference_version": reference.version,
            "validation": evaluate(labels[dev], p, threshold),
            "baseline_validation": baseline_metric,
            "threshold": float(threshold),
            "zero_evidence_probability": zero_p,
            "zero_sufficient_90_percent_margin_accounts": zero_sufficient,
            "positive_human_only_mask_changes": int(np.count_nonzero(changed)),
            "positive_human_only_mask_net_correct_loss": int(
                np.count_nonzero(full_correct)-np.count_nonzero(without_correct)),
            "positive_human_tree_count_by_group": tree_counts,
            "a_direction_count": len(a), "b_fit_count": len(b),
            "normality_channel_count": 9,
        }
        runs.append(row)
        print(json.dumps({"seed": seed,
                          "candidate_acc": row["validation"]["acc"],
                          "baseline_acc": baseline_metric["acc"],
                          "candidate_macro_f1": row["validation"]["macro_f1"],
                          "baseline_macro_f1": baseline_metric["macro_f1"],
                          "zero_sufficient": zero_sufficient,
                          "human_mask_changes": row["positive_human_only_mask_changes"]}),
              flush=True)
    payload = {
        "scope": "train_dev_only_high_performance_mgtab_positive_human_evidence",
        "test_metrics_or_predictions_read": False,
        "source_features_sha256": sha256_file(data_dir / "features.pt"),
        "base_selection_lock_sha256": sha256_file(lock_path),
        "anchor_ratio": ANCHOR_RATIO,
        "lightgbm_version": lightgbm.__version__,
        "runs": runs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
