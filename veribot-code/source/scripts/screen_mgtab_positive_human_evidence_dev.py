"""Train/dev screen for source-present normality as affirmative Human evidence.

The fixed test split, saved test cards, and test metrics are never read here.
"""

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
from screen_mgtab_20_hybrid_concepts import columns
from screen_mgtab_20_train_selected_concepts import rank_content
from screen_mgtab_ab_monotone_evidence_tree_dev import directions, evidence_matrix
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_content_projection_evidence import transform_with_support
from screen_mgtab_targeted_20_concept_ab_group_tree_dev import grouped
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output/v3_mgtab_targeted_20_concept_ab_group_tree"
OUT = ROOT / "output/v3_mgtab_positive_human_evidence/dev_only.json"
RATIOS = (0.0, 0.05, 0.2)


def positive_human_matrix(batch) -> np.ndarray:
    """Each source group receives a source-present, small-deviation score."""
    base = evidence_matrix(batch)
    if base.shape[1] != 40 or len(batch.names) != 20:
        raise ValueError("twenty concepts and forty tail channels required")
    present = np.column_stack([
        np.any(batch.available[:, 2*k:2*k+2], axis=1)
        for k in range(10)
    ]).astype(float)
    deviation = np.column_stack([
        np.maximum(base[:, 2*k], base[:, 2*k+1])
        for k in range(10)
    ])
    normality = present * np.exp(-deviation)
    if np.any(normality < 0) or np.any(normality > 1):
        raise ValueError("normality score outside [0, 1]")
    return np.column_stack((base, normality))


def neutral_training_rows(x: np.ndarray, y: np.ndarray, ratio: float):
    if ratio == 0:
        return x, y, None
    anchor = np.zeros((100, x.shape[1]), dtype=float)
    anchor_labels = np.r_[np.zeros(50, dtype=y.dtype),
                          np.ones(50, dtype=y.dtype)]
    weight = np.r_[np.ones(len(x)),
                   np.full(100, ratio * len(x) / 100)]
    return np.vstack((x, anchor)), np.r_[y, anchor_labels], weight


def normality_tree_count(model: LGBMClassifier) -> int:
    def uses_normality(node: dict) -> bool:
        if "split_feature" not in node:
            return False
        return (int(node["split_feature"]) >= 40 or
                uses_normality(node["left_child"]) or
                uses_normality(node["right_child"]))
    return sum(uses_normality(tree["tree_structure"])
               for tree in model.booster_.dump_model()["tree_info"])


def main() -> None:
    lock = json.loads((BASE / "selection_lock.json").read_text(encoding="utf-8"))
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    if (lock["selected"]["scheme"] !=
            "four_profile_two_relation_four_selected_content" or
            lock["config"] != [31, 400, 50, 10.0] or
            lock["source_features_sha256"] != sha256_file(data_dir / "features.pt")):
        raise ValueError("selected source population or capacity differs")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
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
        train, validation = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(facts[train], names)
        if reference.version != lock["reference_version_by_seed"][str(seed)]:
            raise ValueError("train-only reference differs")
        atomic_train = transform_with_support(reference, facts[train],
                                              split["supports"][train])
        atomic_dev = transform_with_support(reference, facts[validation],
                                            split["supports"][validation])
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=labels[train], random_state=seed)
        content = rank_content(atomic_train, labels[train], a, count=4)
        if [names[j] for j in content] != lock["selected_content_by_seed"][str(seed)]:
            raise ValueError("training-A content selection differs")
        kept, groups = grouped(names, lock["selected"]["scheme"], content)
        train_concepts, _ = collapse_group_tails(columns(atomic_train, kept), groups)
        dev_concepts, _ = collapse_group_tails(columns(atomic_dev, kept), groups)
        x_train = positive_human_matrix(train_concepts)
        x_dev = positive_human_matrix(dev_concepts)
        signs = np.r_[directions(x_train[a, :40], labels[train][a]),
                      -np.ones(10, dtype=np.int8)]
        source_cols = [[2*k, 2*k+1, 20+2*k, 21+2*k, 40+k]
                       for k in range(10)]
        alternatives = []
        for ratio in RATIOS:
            fit_x, fit_y, weights = neutral_training_rows(
                x_train[b], labels[train][b], ratio)
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
            probability = model.predict_proba(x_dev)[:, 1]
            threshold = best_validation_threshold(probability, labels[validation])
            zero_probability = float(model.predict_proba(
                np.zeros((1, x_dev.shape[1])))[:, 1][0])
            full_prediction = probability >= threshold
            full_margin = np.where(full_prediction,
                                   probability-threshold, threshold-probability)
            zero_margin = np.where(full_prediction,
                                   zero_probability-threshold,
                                   threshold-zero_probability)
            zero_already_sufficient = int(np.count_nonzero(
                zero_margin >= 0.9*full_margin - 1e-12))
            alternatives.append({
                "neutral_anchor_weight_ratio": ratio,
                "validation": evaluate(labels[validation], probability, threshold),
                "threshold": float(threshold),
                "zero_evidence_probability": zero_probability,
                "zero_evidence_class": int(zero_probability >= threshold),
                "zero_evidence_90_percent_margin_accounts": zero_already_sufficient,
                "validation_accounts": len(validation),
                "human_normality_used_tree_count": normality_tree_count(model),
                "source_group_tree_counts": tree_counts,
            })
        runs.append({"seed": seed,
                     "split_sha256": old["manifest"]["split_sha256"],
                     "reference_version": reference.version,
                     "concept_version": train_concepts.reference_version,
                     "selected_group_names": list(groups),
                     "source_normality_definition": "source_present * exp(-max(group_high_tail_score, group_low_tail_score))",
                     "alternatives": alternatives})
        print(json.dumps({"seed": seed, "variants": [
            {"ratio": r["neutral_anchor_weight_ratio"],
             "dev_acc": r["validation"]["acc"],
             "dev_macro_f1": r["validation"]["macro_f1"],
             "zero_probability": r["zero_evidence_probability"],
             "zero_sufficient": r["zero_evidence_90_percent_margin_accounts"]}
            for r in alternatives]}), flush=True)
    result = {"scope": "mgtab_train_dev_only_positive_human_normality_evidence",
              "test_predictions_or_metrics_computed": False,
              "source_features_sha256": sha256_file(data_dir / "features.pt"),
              "selected_lock_sha256": sha256_file(BASE / "selection_lock.json"),
              "lightgbm_version": lightgbm.__version__,
              "anchor_ratios": RATIOS, "runs": runs}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
