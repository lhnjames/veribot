"""Lock dev-selected Fox8 neutral models, then test the fixed population."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_positive_human_evidence_dev import neutral_training_rows
from train_fox8_targeted_twenty_ab_group import OUT as BASE, source_and_concepts
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from screen_fox8_twenty_neutral_anchor_dev import OUT as DEV, GROUP_COLS


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_fox8_twenty_neutral_anchor"
LOCK = FOLDER / "selection_lock.json"
BASE_CERT = ROOT / "research/certificates/three_dataset_observed_means_numeric_goal.json"


def lock() -> None:
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    if dev["test_predictions_or_metrics_computed"] is not False:
        raise ValueError("Fox8 candidate selection used test")
    eligible = [x for x in dev["gate_results"]
                if x["passes_predeclared_gate"]]
    if not eligible:
        raise ValueError("No Fox8 neutral anchor passed predeclared gate")
    chosen = max(eligible, key=lambda x: (x["mean_macro_f1"], -x["ratio"]))
    ratio = float(chosen["ratio"])
    index = list(dev["ratios_predeclared"]).index(ratio)
    runs = []
    for seed in (0, 1, 2):
        source = source_and_concepts(seed)
        x, labels = source["matrix"], source["labels"]
        train, validation = source["split"]["train"], source["split"]["val"]
        expected = dev["runs"][seed]["candidates"][index]
        a, b = train_test_split(np.arange(len(train)), test_size=.5,
                                stratify=labels[train], random_state=seed)
        signs = directions(x[train][a], labels[train][a])
        fit_x, fit_y, weight = neutral_training_rows(
            x[train][b], labels[train][b], ratio)
        model = LGBMClassifier(
            objective="binary", n_estimators=200, num_leaves=15,
            min_child_samples=30, reg_lambda=10.0, learning_rate=.05,
            n_jobs=4, random_state=0, deterministic=True,
            force_col_wise=True, verbosity=-1,
            monotone_constraints=signs.tolist(),
            monotone_constraints_method="advanced",
            interaction_constraints=GROUP_COLS)
        model.fit(fit_x, fit_y, sample_weight=weight)
        one_group, counts = all_trees_one_group(model, GROUP_COLS)
        if not one_group:
            raise ValueError("Fox8 neutral tree crossed source groups")
        neutral = float(model.predict_proba(
            np.zeros((1, x.shape[1])))[:, 1][0])
        p = model.predict_proba(x[validation])[:, 1]
        metrics = evaluate(labels[validation], p, neutral)
        if (abs(neutral-expected["neutral_probability"]) > 1e-12 or
                any(abs(metrics[k]-expected[
                    "neutral_threshold_validation"][k]) > 1e-12
                    for k in ("acc", "macro_f1", "auroc"))):
            raise ValueError("Fox8 neutral model failed exact dev replay")
        folder = FOLDER / f"seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "model.joblib"
        contract = folder / "direction_contract.npz"
        joblib.dump(model, path)
        np.savez_compressed(contract, monotone_constraints=signs,
                            group_names=np.asarray(list(source["groups"])))
        runs.append({
            "seed": seed,
            "split_sha256": dev["runs"][seed]["split_sha256"],
            "reference_version": source["reference"].version,
            "reference_path": str(BASE / f"seed{seed}/reference.json.gz"),
            "base_group_path": str(BASE / f"seed{seed}/groups.json"),
            "model_path": str(path), "model_sha256": sha256_file(path),
            "direction_contract_path": str(contract),
            "direction_contract_sha256": sha256_file(contract),
            "neutral_threshold": neutral,
            "validation": metrics,
            "source_group_tree_counts": counts})
        print({"seed": seed, "neutral": neutral,
               "dev_acc": metrics["acc"]}, flush=True)
    LOCK.write_text(json.dumps({
        "scope": "Fox8_20_concept_ab_neutral_anchor_selected_on_dev_before_test",
        "selection_rule": "highest_mean_dev_macro_f1_among_gate_passes_then_smallest_anchor_ratio",
        "chosen_ratio": ratio,
        "dev_screen_sha256": sha256_file(DEV),
        "test_predictions_or_metrics_used_for_selection": False,
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(LOCK)


def test() -> None:
    selection = json.loads(LOCK.read_text(encoding="utf-8"))
    if (selection["scope"] !=
            "Fox8_20_concept_ab_neutral_anchor_selected_on_dev_before_test" or
            selection["dev_screen_sha256"] != sha256_file(DEV) or
            selection["test_predictions_or_metrics_used_for_selection"] is not False):
        raise ValueError("Fox8 neutral selection lock differs")
    base = json.loads(BASE_CERT.read_text(encoding="utf-8"))
    hypbot = next(x for x in base["datasets"] if x["dataset"] == "fox8")
    runs = []
    for seed in (0, 1, 2):
        row = selection["runs"][seed]
        path = Path(row["model_path"])
        if sha256_file(path) != row["model_sha256"]:
            raise ValueError("Fox8 locked model changed")
        reference = LabelBlindReference.load(Path(row["reference_path"]))
        source = source_and_concepts(seed, reference)
        if (source["manifest"]["split_sha256"] != row["split_sha256"] or
                row["split_sha256"] != hypbot["runs"][seed]["split_sha256"]):
            raise ValueError("Fox8 fixed test population changed")
        model = joblib.load(path)
        test_ids = source["split"]["test"]
        x = source["matrix"][test_ids]
        neutral = float(model.predict_proba(
            np.zeros((1, x.shape[1])))[:, 1][0])
        if abs(neutral-row["neutral_threshold"]) > 1e-12:
            raise ValueError("Fox8 neutral boundary differs")
        p = model.predict_proba(x)[:, 1]
        if np.any(np.abs(p-neutral) <= 1e-10):
            raise ValueError("Fox8 account has no signed evidence movement")
        pred = p >= neutral
        metrics = evaluate(source["labels"][test_ids], p, neutral)
        correct = int(np.count_nonzero(
            pred == source["labels"][test_ids]))
        baseline_correct = int(hypbot["runs"][seed]["hypbot_correct"])
        matrix_path = FOLDER / f"seed{seed}/fixed_test_predictions.npz"
        np.savez_compressed(matrix_path,
                            account_ids=np.asarray(source["ids"])[test_ids],
                            probability=p, predicted_bot=pred,
                            threshold=np.asarray(neutral))
        runs.append({
            "seed": seed, "accounts": len(test_ids),
            "correct": correct, "metrics": metrics,
            "hypbot_correct_on_same_population": baseline_correct,
            "strictly_above_hypbot": correct > baseline_correct,
            "strictly_above_99_4_percent_acc": metrics["acc"] > .994,
            "neutral_threshold": neutral,
            "prediction_matrix_path": str(matrix_path),
            "prediction_matrix_sha256": sha256_file(matrix_path),
            "model_sha256": row["model_sha256"],
            "split_sha256": row["split_sha256"]})
        print({"seed": seed, "correct": correct,
               "hypbot": baseline_correct, "acc": metrics["acc"]}, flush=True)
    (FOLDER / "fixed_test_report.json").write_text(json.dumps({
        "scope": "dev_locked_Fox8_20_concept_neutral_anchor_fixed_test",
        "selection_lock_sha256": sha256_file(LOCK),
        "prior_numeric_certificate_sha256": sha256_file(BASE_CERT),
        "test_population_previously_explored_by_other_candidates": True,
        "all_seeds_above_hypbot": all(x["strictly_above_hypbot"] for x in runs),
        "all_seeds_above_99_4_percent": all(
            x["strictly_above_99_4_percent_acc"] for x in runs),
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(FOLDER / "fixed_test_report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("lock", "test"))
    stage = parser.parse_args().stage
    lock() if stage == "lock" else test()


if __name__ == "__main__":
    main()
