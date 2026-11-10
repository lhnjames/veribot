"""Dev-only neutral-anchor screen for the frozen 150-fact A/B architecture.

Adds label-balanced zero-evidence pseudo rows only to B-half fitting. The
predeclared decision threshold is the model's own zero-evidence score, so
both classes require a signed movement from the neutral anchor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import lightgbm
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_ab_monotone_evidence_tree_dev import directions, evidence_matrix
from screen_mgtab_ab_group_additive_monotone_tree_dev import (
    all_trees_one_group, source_groups)
from screen_mgtab_content_projection_evidence import transform_with_support
from screen_mgtab_positive_human_evidence_dev import neutral_training_rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "output/v3_mgtab_ab_group_additive_tree"
SOURCE = ROOT / "output/v3_mgtab_ab_spline_content/selected"
OUT = ROOT / "output/v3_mgtab_150_neutral_anchor/dev_only.json"
RATIOS = (0.02, 0.05, 0.10)
MAX_MEAN_F1_LOSS = 0.005
MAX_SEED_ACC_LOSS = 0.01


def _direction_summary(p: np.ndarray, zero: float, threshold: float) -> dict:
    pred_bot = p >= threshold
    human = ~pred_bot
    return {
        "predicted_bot": int(pred_bot.sum()),
        "predicted_human": int(human.sum()),
        "bot_score_above_zero_anchor": int(np.count_nonzero(
            pred_bot & (p > zero + 1e-10))),
        "human_score_below_zero_anchor": int(np.count_nonzero(
            human & (p < zero - 1e-10))),
        "bot_score_not_above_zero_anchor": int(np.count_nonzero(
            pred_bot & ~(p > zero + 1e-10))),
        "human_score_not_below_zero_anchor": int(np.count_nonzero(
            human & ~(p < zero - 1e-10))),
    }


def main() -> None:
    lock = json.loads((FROZEN / "selection_lock.json").read_text(encoding="utf-8"))
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    if (lock["selected_config"] != [31, 400, 50, 10.0] or
            lock["source_features_sha256"] != sha256_file(data_dir / "features.pt")):
        raise ValueError("Frozen MGTAB model or published source differs")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=ROOT / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=SOURCE)
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((SOURCE / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        facts, labels, names, split = selected_facts(args, seed, old)
        if old["manifest"]["split_sha256"] != lock[
                "split_sha256_by_seed"][str(seed)]:
            raise ValueError("Official train/dev split differs")
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(facts[train], names)
        if reference.version != lock["reference_version_by_seed"][str(seed)]:
            raise ValueError("Train-only reference differs")
        design = evidence_matrix(transform_with_support(
            reference, facts, split["supports"]))
        a, b = train_test_split(train, test_size=0.5,
                                stratify=labels[train], random_state=seed)
        constraints = directions(design[a], labels[a])
        groups = source_groups(names, old["reasoner"]["groups"])
        frozen = joblib.load(FROZEN / f"seed{seed}/model.joblib")
        frozen_p = frozen.predict_proba(design[dev])[:, 1]
        frozen_threshold = float(lock["threshold_by_seed"][str(seed)])
        frozen_metrics = evaluate(labels[dev], frozen_p, frozen_threshold)
        frozen_zero = float(frozen.predict_proba(
            np.zeros((1, design.shape[1])))[:, 1][0])
        results = []
        for ratio in RATIOS:
            fit_x, fit_y, weight = neutral_training_rows(
                design[b], labels[b], ratio)
            model = LGBMClassifier(
                objective="binary", n_estimators=400, num_leaves=31,
                min_child_samples=50, reg_lambda=10.0, learning_rate=0.05,
                n_jobs=4, random_state=0, deterministic=True,
                force_col_wise=True,
                monotone_constraints=constraints.tolist(),
                monotone_constraints_method="advanced",
                interaction_constraints=groups, verbosity=-1)
            model.fit(fit_x, fit_y, sample_weight=weight)
            single_group, tree_counts = all_trees_one_group(model, groups)
            if not single_group:
                raise ValueError("Tree crossed frozen source-group boundary")
            p = model.predict_proba(design[dev])[:, 1]
            zero = float(model.predict_proba(
                np.zeros((1, design.shape[1])))[:, 1][0])
            threshold_opt = best_validation_threshold(p, labels[dev])
            results.append({
                "neutral_anchor_weight_fraction_of_B": ratio,
                "zero_evidence_probability": zero,
                "predeclared_neutral_threshold": zero,
                "neutral_threshold_validation": evaluate(labels[dev], p, zero),
                "neutral_threshold_score_direction":
                    _direction_summary(p, zero, zero),
                "dev_selected_threshold": float(threshold_opt),
                "dev_selected_validation": evaluate(
                    labels[dev], p, threshold_opt),
                "dev_selected_score_direction":
                    _direction_summary(p, zero, threshold_opt),
                "tree_count_by_group": tree_counts,
                "all_trees_one_source_group": single_group,
            })
        runs.append({
            "seed": seed,
            "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "frozen_model_sha256": sha256_file(
                FROZEN / f"seed{seed}/model.joblib"),
            "frozen_validation": frozen_metrics,
            "frozen_threshold": frozen_threshold,
            "frozen_zero_evidence_probability": frozen_zero,
            "candidates": results,
        })
        print({"seed": seed, "frozen_f1": frozen_metrics["macro_f1"],
               "anchored": [
                   (x["neutral_anchor_weight_fraction_of_B"],
                    round(x["neutral_threshold_validation"]["macro_f1"], 5),
                    round(x["zero_evidence_probability"], 5))
                   for x in results]}, flush=True)
    baseline_mean = float(np.mean([
        r["frozen_validation"]["macro_f1"] for r in runs]))
    gate = []
    for index, ratio in enumerate(RATIOS):
        mean_f1 = float(np.mean([r["candidates"][index][
            "neutral_threshold_validation"]["macro_f1"] for r in runs]))
        worst_acc_loss = float(max(
            r["frozen_validation"]["acc"] - r["candidates"][index][
                "neutral_threshold_validation"]["acc"] for r in runs))
        gate.append({
            "ratio": ratio, "mean_neutral_threshold_macro_f1": mean_f1,
            "mean_macro_f1_loss_vs_frozen": baseline_mean-mean_f1,
            "worst_seed_acc_loss_vs_frozen": worst_acc_loss,
            "passes_predeclared_dev_gate":
                baseline_mean-mean_f1 <= MAX_MEAN_F1_LOSS and
                worst_acc_loss <= MAX_SEED_ACC_LOSS})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "MGTAB_150_fact_ab_single_source_group_neutral_anchor_dev_only",
        "test_predictions_or_metrics_computed": False,
        "anchor_labels": "balanced_pseudo_labels_on_zero_evidence_only_B_fit",
        "direction_labels": "A_half_only",
        "reference_labels_used": False,
        "model_source_group_constraints_preserved": True,
        "ratios_predeclared": RATIOS,
        "dev_gate": {"mean_macro_f1_loss_at_most": MAX_MEAN_F1_LOSS,
                     "each_seed_acc_loss_at_most": MAX_SEED_ACC_LOSS},
        "frozen_selection_lock_sha256": sha256_file(
            FROZEN / "selection_lock.json"),
        "lightgbm_version": lightgbm.__version__,
        "runs": runs, "gate_results": gate}, indent=2) + "\n",
        encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
