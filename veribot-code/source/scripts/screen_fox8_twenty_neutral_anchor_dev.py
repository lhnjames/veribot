"""Dev-only neutral-anchor screen for the 20-concept Fox8 A/B model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_positive_human_evidence_dev import neutral_training_rows
from train_fox8_targeted_twenty_ab_group import OUT as BASE, source_and_concepts


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_fox8_twenty_neutral_anchor/dev_only.json"
RATIOS = (0.02, 0.05, 0.10)
MAX_MEAN_F1_LOSS = 0.005
MAX_SEED_ACC_LOSS = 0.01
GROUP_COLS = [[2*k, 2*k+1, 20+2*k, 21+2*k] for k in range(10)]


def main() -> None:
    runs = []
    for seed in (0, 1, 2):
        source = source_and_concepts(seed)
        base_report = json.loads((BASE / f"seed{seed}/dev_report.json")
                                 .read_text(encoding="utf-8"))
        if source["manifest"]["split_sha256"] != base_report[
                "split_sha256"]:
            raise ValueError("Fox8 official train/dev split differs")
        train, dev = source["split"]["train"], source["split"]["val"]
        x, labels = source["matrix"], source["labels"]
        baseline = joblib.load(BASE / f"seed{seed}/model.joblib")
        baseline_p = baseline.predict_proba(x[dev])[:, 1]
        base_validation = evaluate(
            labels[dev], baseline_p, float(base_report["threshold"]))
        if any(abs(base_validation[key] - base_report["validation"][key])
               > 1e-12 for key in ("acc", "macro_f1", "auroc")):
            raise ValueError("Frozen Fox8 validation does not replay")
        a, b = train_test_split(np.arange(len(train)), test_size=.5,
                                stratify=labels[train], random_state=seed)
        signs = directions(x[train][a], labels[train][a])
        if signs.tolist() != base_report["monotone_constraints"]:
            raise ValueError("Fox8 A-half direction contract differs")
        candidates = []
        for ratio in RATIOS:
            fit_x, fit_y, weight = neutral_training_rows(
                x[train][b], labels[train][b], ratio)
            model = LGBMClassifier(
                objective="binary", n_estimators=200, num_leaves=15,
                min_child_samples=30, reg_lambda=10.0,
                learning_rate=.05, n_jobs=4, random_state=0,
                deterministic=True, force_col_wise=True, verbosity=-1,
                monotone_constraints=signs.tolist(),
                monotone_constraints_method="advanced",
                interaction_constraints=GROUP_COLS)
            model.fit(fit_x, fit_y, sample_weight=weight)
            one_group, counts = all_trees_one_group(model, GROUP_COLS)
            if not one_group:
                raise ValueError("Fox8 concept tree crossed source groups")
            p = model.predict_proba(x[dev])[:, 1]
            neutral = float(model.predict_proba(
                np.zeros((1, x.shape[1])))[:, 1][0])
            tuned = best_validation_threshold(p, labels[dev])
            candidates.append({
                "ratio": ratio, "neutral_probability": neutral,
                "neutral_threshold_validation": evaluate(
                    labels[dev], p, neutral),
                "dev_tuned_validation": evaluate(labels[dev], p, tuned),
                "dev_tuned_threshold": float(tuned),
                "single_source_group_tree_count": counts})
        runs.append({
            "seed": seed,
            "split_sha256": base_report["split_sha256"],
            "reference_version": source["reference"].version,
            "base_model_sha256": sha256_file(
                BASE / f"seed{seed}/model.joblib"),
            "base_validation": base_validation,
            "candidates": candidates})
        print({"seed": seed, "base_acc": base_validation["acc"],
               "neutral_acc": [(x["ratio"],
                                x["neutral_threshold_validation"]["acc"],
                                x["neutral_probability"])
                               for x in candidates]}, flush=True)
    base_mean = float(np.mean([r["base_validation"]["macro_f1"]
                              for r in runs]))
    gates = []
    for index, ratio in enumerate(RATIOS):
        f1 = float(np.mean([r["candidates"][index][
            "neutral_threshold_validation"]["macro_f1"] for r in runs]))
        worst_acc_loss = float(max(
            r["base_validation"]["acc"] - r["candidates"][index][
                "neutral_threshold_validation"]["acc"] for r in runs))
        gates.append({
            "ratio": ratio, "mean_macro_f1": f1,
            "mean_macro_f1_loss": base_mean-f1,
            "worst_seed_acc_loss": worst_acc_loss,
            "passes_predeclared_gate":
                base_mean-f1 <= MAX_MEAN_F1_LOSS and
                worst_acc_loss <= MAX_SEED_ACC_LOSS})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "Fox8_20_concept_ab_single_source_group_neutral_anchor_dev_only",
        "test_predictions_or_metrics_computed": False,
        "anchor_labels": "balanced_zero_evidence_pseudo_labels_only_in_B_fit",
        "ratios_predeclared": RATIOS,
        "dev_gate": {"mean_macro_f1_loss_at_most": MAX_MEAN_F1_LOSS,
                     "each_seed_acc_loss_at_most": MAX_SEED_ACC_LOSS},
        "runs": runs, "gate_results": gates}, indent=2) + "\n",
        encoding="utf-8")
    print(gates, OUT)


if __name__ == "__main__":
    main()
