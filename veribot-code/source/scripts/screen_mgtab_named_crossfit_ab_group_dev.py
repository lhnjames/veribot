"""Dev-only bidirectional A/B ensemble using named MGTAB evidence groups."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_ab_monotone_evidence_tree_dev import directions, evidence_matrix
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_content_projection_evidence import transform_with_support
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural
from verifiable_bot.mgtab_full_atomic_runtime import AB_RELATIONS
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "output/v3_mgtab_named_ab_group_tree/dev_only.json"
OUTPUT = ROOT / "output/v3_mgtab_named_crossfit_ab_group/dev_only.json"
CONFIG = (31, 400, 50, 10.0)


def groups_for_named(names: list[str]) -> list[list[int]]:
    groups = [list(range(20))]
    for relation in AB_RELATIONS:
        groups.append([j for j in range(20, 86)
                       if names[j].startswith(f"published_{relation}_")])
    if [len(group) for group in groups] != [20, 14, 14, 6, 14, 6, 6, 6]:
        raise ValueError("named source relation partition differs")
    output = [members + [j+86 for j in members] for members in groups]
    if sorted(j for group in output for j in group) != list(range(172)):
        raise ValueError("named evidence tails do not partition")
    return output


def fit_branch(design: np.ndarray, labels: np.ndarray,
               direction_rows: np.ndarray, fit_rows: np.ndarray,
               source_groups: list[list[int]]) -> tuple[LGBMClassifier, np.ndarray]:
    signs = directions(design[direction_rows], labels[direction_rows])
    leaves, rounds, min_leaf, l2 = CONFIG
    model = LGBMClassifier(
        objective="binary", n_estimators=rounds,
        num_leaves=leaves, min_child_samples=min_leaf,
        reg_lambda=l2, learning_rate=0.05, n_jobs=4,
        random_state=0, deterministic=True, force_col_wise=True,
        monotone_constraints=signs.tolist(),
        monotone_constraints_method="advanced",
        interaction_constraints=source_groups, verbosity=-1)
    model.fit(design[fit_rows], labels[fit_rows])
    single_group, _ = all_trees_one_group(model, source_groups)
    if not single_group:
        raise ValueError("tree crossed named source groups")
    return model, signs


def main() -> None:
    data = resolve_mgtab_source_dir(ROOT)
    structural = ROOT / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = ROOT / "data/processed/mgtab_v3/neighbor_facts.npz"
    prior = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    if (prior["source_features_sha256"] != sha256_file(data / "features.pt") or
            prior["configs"][1] != list(CONFIG)):
        raise ValueError("prior named A/B control changed")
    runs = []
    for seed in (0, 1, 2):
        facts, labels, ids, names, split, source = load(
            "mgtab", data, None, seed)
        facts, names, source = attach_mgtab_structural(
            facts, names, ids, source, data, structural)
        facts, names, supports, source = attach_mgtab_neighbor(
            facts, names, ids, source, data, neighbor)
        if source["split_sha256"] != prior["runs"][seed]["split_sha256"]:
            raise ValueError("official split differs")
        facts, names = facts[:, :86], names[:86]
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(
            facts[train], names)
        if reference.version != prior["runs"][seed]["reference_version"]:
            raise ValueError("named reference differs")
        train_design = evidence_matrix(transform_with_support(
            reference, facts[train], supports[train]))
        dev_design = evidence_matrix(transform_with_support(
            reference, facts[dev], supports[dev]))
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=labels[train], random_state=seed)
        named_groups = groups_for_named(names)
        forward, signs_a = fit_branch(
            train_design, labels[train], a, b, named_groups)
        reverse, signs_b = fit_branch(
            train_design, labels[train], b, a, named_groups)
        forward_raw = np.asarray(forward.booster_.predict(
            dev_design, raw_score=True))
        reverse_raw = np.asarray(reverse.booster_.predict(
            dev_design, raw_score=True))
        forward_p = expit(forward_raw)
        forward_threshold = best_validation_threshold(
            forward_p, labels[dev])
        forward_result = evaluate(labels[dev], forward_p,
                                  forward_threshold)
        earlier = prior["runs"][seed]["candidates"][1]
        if (abs(forward_threshold-earlier["selected_threshold"]) > 1e-9 or
                any(abs(forward_result[key]-earlier["validation"][key]) > 1e-9
                    for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("forward named A/B branch did not replay")
        reverse_p = expit(reverse_raw)
        reverse_threshold = best_validation_threshold(
            reverse_p, labels[dev])
        reverse_result = evaluate(labels[dev], reverse_p,
                                  reverse_threshold)
        ensemble_p = expit((forward_raw + reverse_raw) / 2)
        ensemble_threshold = best_validation_threshold(
            ensemble_p, labels[dev])
        ensemble_result = evaluate(labels[dev], ensemble_p,
                                   ensemble_threshold)
        disagreements = int(np.count_nonzero(signs_a != signs_b))
        runs.append({
            "seed": seed,
            "split_sha256": source["split_sha256"],
            "reference_version": reference.version,
            "direction_half_a": len(a), "fit_half_b": len(b),
            "direction_half_b": len(b), "fit_half_a": len(a),
            "named_fact_count": 86,
            "source_group_count": len(named_groups),
            "branch_tree_count": CONFIG[1],
            "all_branch_trees_single_source_group": True,
            "opposite_half_direction_disagreements": disagreements,
            "forward": {"validation": forward_result,
                        "threshold": forward_threshold},
            "reverse": {"validation": reverse_result,
                        "threshold": reverse_threshold},
            "mean_logit_ensemble": {"validation": ensemble_result,
                                    "threshold": ensemble_threshold},
        })
        print({"seed": seed,
               "forward_macro_f1": forward_result["macro_f1"],
               "reverse_macro_f1": reverse_result["macro_f1"],
               "ensemble_macro_f1": ensemble_result["macro_f1"],
               "direction_disagreements": disagreements}, flush=True)
    forward_mean = float(np.mean([r["forward"]["validation"]["macro_f1"]
                                  for r in runs]))
    ensemble_mean = float(np.mean([
        r["mean_logit_ensemble"]["validation"]["macro_f1"] for r in runs]))
    gate = (ensemble_mean >= forward_mean + 0.01 and all(
        r["mean_logit_ensemble"]["validation"]["macro_f1"] >=
        r["forward"]["validation"]["macro_f1"] for r in runs))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "scope": "mgtab_train_dev_only_named_evidence_crossfit_ab_group_ensemble",
        "test_predictions_or_metrics_accessed": False,
        "test_labels_loaded_by_full_dataset_adapter": True,
        "test_labels_used_for_fit_or_selection": False,
        "source_features_sha256": sha256_file(data / "features.pt"),
        "previous_dev_sha256": sha256_file(PREVIOUS),
        "config": CONFIG,
        "selection_gate": "ensemble mean dev Macro-F1 at least one point above original forward branch and every seed non-inferior",
        "forward_mean_dev_macro_f1": forward_mean,
        "ensemble_mean_dev_macro_f1": ensemble_mean,
        "gate_passed": gate,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
