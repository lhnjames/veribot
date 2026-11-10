"""Freeze the best <=20-fact MGTAB A/B candidate selected on official dev."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from screen_mgtab_twenty_source_facts_dev import choose, rank_facts
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
SCREEN = ROOT / "output/v3_mgtab_twenty_source_facts/dev_only.json"
FOLDER = ROOT / "output/v3_mgtab_twenty_source_facts/selected"
OUT = FOLDER / "dev_lock.json"
CAP = 3.0


def main() -> None:
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if screen["test_predictions_or_metrics_accessed"] or len(screen["runs"]) != 3:
        raise ValueError("candidate screen is not dev-only")
    quota_scores = [np.mean([run["candidates"][j]["validation"]["macro_f1"]
                             for run in screen["runs"]])
                    for j in range(len(screen["quota_grid_profile_relation_content"]))]
    selected_index = int(np.argmax(quota_scores))
    quota = tuple(screen["quota_grid_profile_relation_content"][selected_index])
    if quota != (4, 8, 8):
        raise ValueError("pre-specified dev choice changed")
    FOLDER.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load("mgtab", runtime.data_dir, None, seed)
        if ids != runtime.ids or source["split_sha256"] != screen["runs"][seed]["split_sha256"]:
            raise ValueError("source or official split differs")
        with np.load(runtime.run_dir / "ab_direction_contract.npz",
                     allow_pickle=False) as contract:
            a, b = contract["a_indices"].astype(int), contract["b_indices"].astype(int)
        if len(set(a) & set(b)) or not np.array_equal(
                np.sort(np.r_[a, b]), np.sort(split["train"])):
            raise ValueError("A/B train split differs")
        dev = split["val"]
        x_a, x_b, x_dev = (evidence_design(runtime, rows)
                           for rows in (a, b, dev))
        selected = choose(runtime, rank_facts(x_a, labels[a]), *quota)
        prior = screen["runs"][seed]["candidates"][selected_index]
        if selected.tolist() != prior["selected_fact_indices"]:
            raise ValueError("A-only fact selection differs")
        d = len(selected)
        columns = np.r_[selected, selected + 150]
        signs = directions(x_a[:, columns], labels[a]).astype(int)
        groups = []
        group_names = []
        for name, members in runtime.group_fact_indices.items():
            local = [j for j, source_j in enumerate(selected) if source_j in members]
            if local:
                groups.append(local + [j + d for j in local])
                group_names.append(name)
        if signs.tolist() != prior["signs"] or group_names != prior["source_groups"]:
            raise ValueError("A-only directions or groups differ")
        model = LGBMClassifier(
            objective="binary", n_estimators=400, num_leaves=31,
            min_child_samples=50, reg_lambda=10.0, learning_rate=.05,
            n_jobs=4, random_state=0, deterministic=True,
            force_col_wise=True, verbosity=-1,
            monotone_constraints=signs.tolist(),
            monotone_constraints_method="advanced",
            interaction_constraints=groups)
        model.fit(x_b[:, columns], labels[b])
        single, counts = all_trees_one_group(model, groups)
        if not single or counts != {int(k): v for k, v in prior[
                "tree_count_by_group_index"].items()}:
            raise ValueError("model group restriction differs")
        dev_design = x_dev[:, columns]
        full_logit = np.asarray(model.booster_.predict(dev_design, raw_score=True))
        named = dev_design.copy()
        named[:, 12:20] = 0.0
        named[:, 32:40] = 0.0
        named_logit = np.asarray(model.booster_.predict(named, raw_score=True))
        probability = expit(named_logit + np.clip(full_logit - named_logit, -CAP, CAP))
        threshold = best_validation_threshold(probability, labels[dev])
        metrics = evaluate(labels[dev], probability, threshold)
        if (metrics != prior["validation"] or threshold != prior["threshold"]):
            raise ValueError("frozen candidate differs from dev screen")
        directory = FOLDER / f"seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "model.joblib"
        joblib.dump(model, model_path)
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "selected_fact_indices": selected.tolist(),
            "selected_fact_names": prior["selected_fact_names"],
            "encoded_source_groups": groups,
            "source_group_names": group_names,
            "monotone_constraints": signs.tolist(),
            "threshold": threshold, "validation": metrics,
            "model_sha256": sha256_file(model_path),
            "original_reference_sha256": sha256_file(
                runtime.run_dir / "label_blind_reference.json.gz"),
            "original_direction_contract_sha256": sha256_file(
                runtime.run_dir / "ab_direction_contract.npz"),
        })
        print({"seed": seed, "dev_acc": metrics["acc"]}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "mgtab_twenty_source_facts_A_selected_B_fit_dev_lock",
        "screen_sha256": sha256_file(SCREEN),
        "selected_quota": list(quota),
        "selection_rule": "highest_mean_official_dev_macro_f1_then_first_grid_order",
        "content_output_absolute_logit_cap": CAP,
        "test_predictions_or_metrics_accessed": False,
        "prior_test_population_exploration": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
