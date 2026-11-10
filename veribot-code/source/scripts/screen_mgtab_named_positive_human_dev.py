"""Train/dev-only screen of named MGTAB facts plus affirmative Human evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_content_projection_evidence import transform_with_support
from screen_mgtab_positive_human_evidence_dev import neutral_training_rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_mgtab_named_positive_human"
BASE_LOCK = ROOT / "output/v3_mgtab_named_evidence_lightgbm/selection_lock.json"


def named_matrix(batch) -> np.ndarray:
    n = len(batch.names)
    if n != 86:
        raise ValueError("expected 86 named release facts")
    high = np.log1p(batch.upper)*batch.reliability
    low = np.log1p(batch.lower)*batch.reliability
    present = batch.available.astype(float)
    normality = present*np.exp(-np.maximum(high, low))
    return np.column_stack((high, low, normality))


def main() -> None:
    lock = json.loads(BASE_LOCK.read_text(encoding="utf-8"))
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=ROOT / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=ROOT / "output/v3_mgtab_ab_spline_content/selected")
    if (lock["source_features_sha256"] != sha256_file(data_dir / "features.pt") or
            lock["structural_facts_sha256"] != sha256_file(args.structural_facts) or
            lock["neighbor_facts_sha256"] != sha256_file(args.neighbor_facts)):
        raise ValueError("named published source differs")
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        facts, labels, names, split = selected_facts(args, seed, old)
        if old["manifest"]["split_sha256"] != lock["split_sha256_by_seed"][str(seed)]:
            raise ValueError("official split differs")
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(facts[train, :86],
                                                          names[:86])
        if reference.version != lock["reference_version_by_seed"][str(seed)]:
            raise ValueError("train-only named reference differs")
        train_batch = transform_with_support(reference, facts[train, :86],
                                             split["supports"][train])
        dev_batch = transform_with_support(reference, facts[dev, :86],
                                           split["supports"][dev])
        x_train, x_dev = named_matrix(train_batch), named_matrix(dev_batch)
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=labels[train], random_state=seed)
        signs = np.r_[directions(x_train[a, :172], labels[train][a]),
                      -np.ones(86, dtype=np.int8)]
        source_cols = [
            list(range(start, stop)) + list(range(86+start, 86+stop)) +
            list(range(172+start, 172+stop))
            for start, stop in ((0, 20), (20, 86))
        ]
        variants = []
        for ratio in (0.0, 0.05):
            fit_x, fit_y, weights = neutral_training_rows(
                x_train[b], labels[train][b], ratio)
            model = LGBMClassifier(
                objective="binary", n_estimators=400, num_leaves=31,
                min_child_samples=50, reg_lambda=10.0, learning_rate=0.05,
                n_jobs=4, random_state=0, deterministic=True,
                force_col_wise=True,
                monotone_constraints=signs.tolist(),
                monotone_constraints_method="advanced",
                interaction_constraints=source_cols, verbosity=-1)
            model.fit(fit_x, fit_y, sample_weight=weights)
            one_group, tree_counts = all_trees_one_group(model, source_cols)
            if not one_group:
                raise ValueError("source type groups crossed in a tree")
            p = model.predict_proba(x_dev)[:, 1]
            threshold = best_validation_threshold(p, labels[dev])
            zero = float(model.predict_proba(np.zeros((1,258)))[:,1][0])
            full_pred = p >= threshold
            margin = np.where(full_pred,p-threshold,threshold-p)
            zero_margin = np.where(full_pred,zero-threshold,threshold-zero)
            normality_removed = x_dev.copy()
            normality_removed[:,172:] = 0
            without_pred = model.predict_proba(normality_removed)[:,1] >= threshold
            variants.append({
                "anchor_ratio": ratio,
                "validation": evaluate(labels[dev], p, threshold),
                "threshold": float(threshold),
                "zero_evidence_probability": zero,
                "zero_evidence_90_percent_margin_accounts": int(np.count_nonzero(
                    zero_margin >= .9*margin-1e-12)),
                "normality_only_mask_class_changes": int(np.count_nonzero(
                    full_pred != without_pred)),
                "normality_only_mask_net_correct_lost": int(np.count_nonzero(
                    (full_pred==labels[dev])&(without_pred!=labels[dev]))-
                    np.count_nonzero((full_pred!=labels[dev])&
                                     (without_pred==labels[dev]))),
                "source_group_tree_counts": tree_counts,
            })
        runs.append({
            "seed": seed,
            "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "named_fact_count": 86,
            "native_relation_fact_count": 66,
            "published_content_projection_used": False,
            "variants": variants,
        })
        print(json.dumps({"seed":seed,"variants":[
            {"ratio":v["anchor_ratio"],"acc":v["validation"]["acc"],
             "f1":v["validation"]["macro_f1"],
             "zero_sufficient":v["zero_evidence_90_percent_margin_accounts"],
             "normality_changed":v["normality_only_mask_class_changes"]}
            for v in variants]}), flush=True)
    FOLDER.mkdir(parents=True, exist_ok=True)
    (FOLDER / "dev_only.json").write_text(json.dumps({
        "scope":"mgtab_named_published_profile_and_native_relation_positive_human_dev_only",
        "test_predictions_or_metrics_computed":False,
        "source_features_sha256":lock["source_features_sha256"],
        "base_named_lock_sha256":sha256_file(BASE_LOCK),
        "normality_rule":"source_present * exp(-max(named_fact_high_tail, named_fact_low_tail))",
        "runs":runs},indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
