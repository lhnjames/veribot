"""Dev-only capacity diagnostic for named source-backed Human evidence."""

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
from screen_mgtab_content_projection_evidence import transform_with_support
from screen_mgtab_named_positive_human_dev import named_matrix
from screen_mgtab_positive_human_evidence_dev import neutral_training_rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
BASE_LOCK = ROOT / "output/v3_mgtab_named_evidence_lightgbm/selection_lock.json"
OUT = ROOT / "output/v3_mgtab_named_positive_human/capacity_dev_only.json"


def main() -> None:
    lock = json.loads(BASE_LOCK.read_text(encoding="utf-8"))
    args = SimpleNamespace(
        dataset="mgtab",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB"),
        account_file=None, native_graph_facts=None,
        structural_facts=ROOT / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=ROOT / "output/v3_mgtab_ab_spline_content/selected")
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
            raise ValueError("train-only reference differs")
        x_train = named_matrix(transform_with_support(
            reference, facts[train, :86], split["supports"][train]))
        x_dev = named_matrix(transform_with_support(
            reference, facts[dev, :86], split["supports"][dev]))
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
        for mode in ("full_train_group_tree", "full_train_cross_group_tree"):
            fit_x, fit_y, weight = neutral_training_rows(
                x_train, labels[train], 0.05)
            params = dict(
                objective="binary", n_estimators=400, num_leaves=31,
                min_child_samples=50, reg_lambda=10.0, learning_rate=0.05,
                n_jobs=4, random_state=0, deterministic=True,
                force_col_wise=True,
                monotone_constraints=signs.tolist(),
                monotone_constraints_method="advanced", verbosity=-1)
            if mode == "full_train_group_tree":
                params["interaction_constraints"] = source_cols
            model = LGBMClassifier(**params)
            model.fit(fit_x, fit_y, sample_weight=weight)
            p = model.predict_proba(x_dev)[:, 1]
            threshold = best_validation_threshold(p, labels[dev])
            zero = float(model.predict_proba(np.zeros((1,258)))[:,1][0])
            full_pred = p >= threshold
            margin = np.where(full_pred,p-threshold,threshold-p)
            zero_margin = np.where(full_pred,zero-threshold,threshold-zero)
            without = x_dev.copy()
            without[:,172:] = 0
            without_pred = model.predict_proba(without)[:,1] >= threshold
            variants.append({
                "fit_mode":mode,
                "direction_labels_from_train_A":True,
                "tree_fit_uses_train_A_also":True,
                "anchor_ratio":0.05,
                "validation":evaluate(labels[dev],p,threshold),
                "threshold":float(threshold),
                "zero_evidence_probability":zero,
                "zero_evidence_90_percent_margin_accounts":int(np.count_nonzero(
                    zero_margin >= .9*margin-1e-12)),
                "normality_only_mask_class_changes":int(np.count_nonzero(
                    full_pred != without_pred)),
            })
        runs.append({"seed":seed,"split_sha256":old["manifest"]["split_sha256"],
                     "reference_version":reference.version,"variants":variants})
        print(json.dumps({"seed":seed,"variants":[
            {"mode":v["fit_mode"],"acc":v["validation"]["acc"],
             "macro_f1":v["validation"]["macro_f1"],
             "zero_sufficient":v["zero_evidence_90_percent_margin_accounts"]}
            for v in variants]}),flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({
        "scope":"mgtab_named_source_positive_human_capacity_diagnostic_official_dev_only",
        "test_predictions_or_metrics_computed":False,
        "published_content_projection_used":False,
        "base_named_lock_sha256":sha256_file(BASE_LOCK),
        "runs":runs},indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
