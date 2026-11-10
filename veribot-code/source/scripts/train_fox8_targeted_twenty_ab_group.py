"""Validation-lock a 20-concept A/B Fox8 classifier, then evaluate fixed test."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_20_hybrid_concepts import columns
from screen_fox8_neighbor_atomic_evidence_capacity_dev import (
    evidence_matrix, load_cache)
from screen_fox8_native_neighbor_profile_concepts_dev import weighted_atomic
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_fox8_targeted_twenty_ab_group"
SCREEN = ROOT / "output/v3_fox8_targeted_neighbor_20_concepts/dev_only.json"
CACHE = ROOT / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
GROUP_FILE = ROOT / "research/fox8_20_concept_groups.json"
NEIGHBOR = "native_interaction_neighbor_mean_statuses_count"
REMOVED = "presentation_flags"
CONFIG = (15, 200, 30, 10.0)
SOURCE_DATA = Path("D:/Code/SAHG/SAHG/code/data/fox8_23")


def source_and_concepts(seed: int, reference: LabelBlindReference | None = None):
    source_audit = json.loads((ROOT /
        "research/certificates/fox8_neighbor_atomic_evidence_full_replay.json"
                               ).read_text(encoding="utf-8"))
    if (source_audit["scope"] !=
            "fox8_all_account_raw_neighbor_facts_and_all_test_decisions_independent_replay"
            or not source_audit["all_neighbor_means_match_raw_source_at_float32_precision"]):
        raise ValueError("full native-neighbor source audit differs")
    sample_audit = json.loads((ROOT /
        "research/certificates/fox8_native_neighbor_profile_raw_sample_audit.json"
                               ).read_text(encoding="utf-8"))
    if sample_audit["fact_artifact_sha256"] != sha256_file(CACHE):
        raise ValueError("native-neighbor source cache differs")
    ids, neighbor_names, neighbor_values, supports = load_cache(CACHE)
    args = SimpleNamespace(
        dataset="fox8", data_dir=SOURCE_DATA,
        account_file=None, native_graph_facts=None,
        structural_facts=None, neighbor_facts=None,
        fox8_native_facts=ROOT / "data/processed/fox8_v3/native_interaction_facts.npz",
        selected_model_dir=ROOT / "output/v3_fox8_native_spline_l2/selected")
    old = json.loads((args.selected_model_dir / f"seed{seed}/report.json"
                      ).read_text(encoding="utf-8"))
    facts, labels, names, split = selected_facts(args, seed, old)
    if (len(names) != 43 or len(neighbor_names) != 31 or
            pd.read_csv(SOURCE_DATA / "features.csv")[
                "user_id"].astype(str).tolist() != ids or
            NEIGHBOR not in neighbor_names):
        raise ValueError("Fox8 source population differs")
    all_facts = np.column_stack((facts, neighbor_values))
    all_names = names + neighbor_names
    if reference is None:
        reference = LabelBlindReference(min_context_n=30).fit(
            all_facts[split["train"]], all_names)
    if reference.names != tuple(all_names):
        raise ValueError("frozen reference vocabulary differs")
    old_groups = json.loads(GROUP_FILE.read_text(encoding="utf-8"))
    removed_facts = set(old_groups[REMOVED])
    kept = [j for j, name in enumerate(all_names)
            if (j < 43 and name not in removed_facts) or name == NEIGHBOR]
    groups = {group: members for group, members in old_groups.items()
              if group != REMOVED}
    groups["native_neighbor_volume"] = [NEIGHBOR]
    atomic = weighted_atomic(reference, all_facts, supports)
    concept, winners = collapse_group_tails(columns(atomic, kept), groups)
    if len(concept.names) != 20 or len(kept) != 33:
        raise ValueError("20-concept source map differs")
    return {
        "ids": ids, "labels": labels, "split": split,
        "manifest": old["manifest"], "reference": reference,
        "concept": concept, "winners": winners,
        "groups": groups, "kept_fact_names": [all_names[j] for j in kept],
        "matrix": evidence_matrix(concept),
    }


def train_seed(seed: int, screen: dict) -> dict:
    source = source_and_concepts(seed)
    expected = screen["runs"][seed]
    if (source["manifest"]["split_sha256"] != expected["split_sha256"]
            or source["reference"].version != expected["reference_version"]):
        raise ValueError("training source differs from validation screen")
    chosen = next(row for row in expected["candidates"]
                  if row["removed_group"] == REMOVED)
    expected_model = next(row for row in chosen["ab_group_trees"]
                          if tuple(row["config"]) == CONFIG)
    train, val = source["split"]["train"], source["split"]["val"]
    a, b = train_test_split(np.arange(len(train)), test_size=.5,
                            stratify=source["labels"][train], random_state=seed)
    x = source["matrix"]
    signs = directions(x[train][a], source["labels"][train][a])
    group_indices = [[2*k, 2*k+1, 20+2*k, 21+2*k]
                     for k in range(10)]
    leaves, rounds, min_leaf, l2 = CONFIG
    model = LGBMClassifier(
        objective="binary", n_estimators=rounds, num_leaves=leaves,
        min_child_samples=min_leaf, reg_lambda=l2, learning_rate=.05,
        n_jobs=4, random_state=0, deterministic=True,
        force_col_wise=True, verbosity=-1,
        monotone_constraints=signs.tolist(),
        monotone_constraints_method="advanced",
        interaction_constraints=group_indices)
    model.fit(x[train][b], source["labels"][train][b])
    single_group, tree_counts = all_trees_one_group(model, group_indices)
    if (not single_group or
            {str(k): v for k, v in tree_counts.items()} !=
            expected_model["tree_count_by_group"]):
        raise ValueError("frozen trees cross concept groups")
    probability = model.predict_proba(x[val])[:, 1]
    threshold = best_validation_threshold(probability, source["labels"][val])
    validation = evaluate(source["labels"][val], probability, threshold)
    if (abs(threshold - expected_model["selected_threshold"]) > 1e-12
            or any(abs(validation[key] - expected_model["validation_selected"][key])
                   > 1e-12 for key in ("acc", "macro_f1", "auroc"))):
        raise ValueError("20-concept A/B model does not replay validation screen")
    out = OUT / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    source["reference"].save(out / "reference.json.gz")
    joblib.dump(model, out / "model.joblib")
    shutil.copyfile(
        ROOT / f"output/v3_current_compact_deployment/fox8_source_seed{seed}.json",
        out / "source_metadata.json")
    (out / "groups.json").write_text(json.dumps(source["groups"], indent=2)
                                      + "\n", encoding="utf-8")
    report = {
        "seed": seed, "split_sha256": expected["split_sha256"],
        "reference_version": source["reference"].version,
        "source_cache_sha256": sha256_file(CACHE),
        "removed_group": REMOVED, "added_fact": NEIGHBOR,
        "retained_fact_names": source["kept_fact_names"],
        "concept_names": source["concept"].names,
        "config": CONFIG,
        "direction_half_count": len(a), "fit_half_count": len(b),
        "monotone_constraints": signs.tolist(),
        "all_trees_one_source_group": single_group,
        "tree_count_by_group": tree_counts,
        "threshold": threshold, "validation": validation,
        "model_sha256": sha256_file(out / "model.joblib"),
        "reference_sha256": sha256_file(out / "reference.json.gz"),
        "groups_sha256": sha256_file(out / "groups.json"),
        "source_metadata_sha256": sha256_file(out / "source_metadata.json"),
    }
    (out / "dev_report.json").write_text(json.dumps(report, indent=2) + "\n",
                                         encoding="utf-8")
    return report


def test_seed(seed: int, lock: dict) -> dict:
    expected = lock["runs"][seed]
    out = OUT / f"seed{seed}"
    if any(sha256_file(out / filename) != expected[key]
           for filename, key in (("model.joblib", "model_sha256"),
                                 ("reference.json.gz", "reference_sha256"),
                                 ("groups.json", "groups_sha256"),
                                 ("source_metadata.json", "source_metadata_sha256"))):
        raise ValueError("20-concept frozen artifacts differ")
    source = source_and_concepts(
        seed, LabelBlindReference.load(out / "reference.json.gz"))
    if (source["manifest"]["split_sha256"] != expected["split_sha256"]
            or source["reference"].version != expected["reference_version"]
            or list(source["concept"].names) != expected["concept_names"]):
        raise ValueError("fixed-test source differs from validation lock")
    model = joblib.load(out / "model.joblib")
    if model.get_params()["monotone_constraints"] != expected[
            "monotone_constraints"]:
        raise ValueError("A/B monotone direction differs")
    test = source["split"]["test"]
    x = source["matrix"][test]
    probability = model.predict_proba(x)[:, 1]
    threshold = float(expected["threshold"])
    metrics = evaluate(source["labels"][test], probability, threshold)
    group_indices = [[2*k, 2*k+1, 20+2*k, 21+2*k]
                     for k in range(10)]
    single_group, counts = all_trees_one_group(model, group_indices)
    if (not single_group or
            {str(k): v for k, v in counts.items()} !=
            expected["tree_count_by_group"]):
        raise ValueError("frozen test trees cross source groups")
    neighbor_index = list(source["groups"]).index("native_neighbor_volume")
    masked = x.copy()
    masked[:, group_indices[neighbor_index]] = 0.0
    masked_probability = model.predict_proba(masked)[:, 1]
    changed = int(np.count_nonzero((probability >= threshold) !=
                                   (masked_probability >= threshold)))
    rows = [{"account_id": source["ids"][int(i)],
             "label": int(source["labels"][i]),
             "probability": float(p),
             "prediction": int(p >= threshold),
             "without_neighbor_probability": float(q)}
            for i, p, q in zip(test, probability, masked_probability, strict=True)]
    path = out / "test_predictions.json"
    path.write_text(json.dumps({"seed": seed, "threshold": threshold,
                                "predictions": rows}, indent=2) + "\n",
                    encoding="utf-8")
    return {
        "seed": seed, "split_sha256": expected["split_sha256"],
        "test": metrics,
        "neighbor_group_zero_changed_test_classes": changed,
        "test_predictions_sha256": sha256_file(path),
        "model_sha256": expected["model_sha256"],
        "reference_sha256": expected["reference_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dev", "test"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "dev":
        screen = json.loads(SCREEN.read_text(encoding="utf-8"))
        if screen["test_predictions_or_metrics_computed"]:
            raise ValueError("validation screen already accessed test")
        runs = [train_seed(seed, screen) for seed in (0, 1, 2)]
        lock = {
            "status": "selected_on_official_dev_before_targeted_twenty_test",
            "validation_screen_sha256": sha256_file(SCREEN),
            "selection": "remove presentation flags; add one native neighbor volume; 15 leaves 200 rounds",
            "runs": runs,
        }
        (OUT / "selection_lock.json").write_text(
            json.dumps(lock, indent=2) + "\n", encoding="utf-8")
        print({"stage": "dev", "acc": [run["validation"]["acc"]
                                       for run in runs]})
    else:
        lock_path = OUT / "selection_lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if (lock["status"] != "selected_on_official_dev_before_targeted_twenty_test"
                or lock["validation_screen_sha256"] != sha256_file(SCREEN)):
            raise ValueError("validation selection lock differs")
        runs = [test_seed(seed, lock) for seed in (0, 1, 2)]
        report = {
            "status": "validation_locked_twenty_concept_ab_group_tree_fixed_test",
            "selection_lock_sha256": sha256_file(lock_path),
            "prior_test_population_exploration": True,
            "runs": runs,
        }
        path = OUT / "fixed_test_report.json"
        path.write_text(json.dumps(report, indent=2) + "\n",
                        encoding="utf-8")
        print({"stage": "test", "acc": [run["test"]["acc"]
                                        for run in runs],
               "neighbor_changes": [run[
                   "neighbor_group_zero_changed_test_classes"]
                                    for run in runs]})
        print(path)


if __name__ == "__main__":
    main()
