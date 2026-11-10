"""Dev-only 25/75 direction/fit allocation for named MGTAB group trees."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_mgtab_ab_monotone_evidence_tree_dev import evidence_matrix, directions
from screen_mgtab_content_projection_evidence import transform_with_support
from screen_mgtab_named_crossfit_ab_group_dev import fit_branch, groups_for_named
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "output/v3_mgtab_named_ab_group_tree/dev_only.json"
OUTPUT = ROOT / "output/v3_mgtab_named_direction_quarter/dev_only.json"


def main() -> None:
    old = json.loads(PRIOR.read_text(encoding="utf-8"))
    data = resolve_mgtab_source_dir(ROOT)
    structural = ROOT / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = ROOT / "data/processed/mgtab_v3/neighbor_facts.npz"
    if old["source_features_sha256"] != sha256_file(data / "features.pt"):
        raise ValueError("published MGTAB features changed")
    runs = []
    for seed in (0, 1, 2):
        facts, labels, ids, names, split, source = load(
            "mgtab", data, None, seed)
        facts, names, source = attach_mgtab_structural(
            facts, names, ids, source, data, structural)
        facts, names, supports, source = attach_mgtab_neighbor(
            facts, names, ids, source, data, neighbor)
        if (len(names) != 86 or source["split_sha256"] !=
                old["runs"][seed]["split_sha256"]):
            raise ValueError("named source or official split differs")
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(
            facts[train], names)
        if reference.version != old["runs"][seed]["reference_version"]:
            raise ValueError("named reference differs")
        design_train = evidence_matrix(transform_with_support(
            reference, facts[train], supports[train]))
        design_dev = evidence_matrix(transform_with_support(
            reference, facts[dev], supports[dev]))
        groups = groups_for_named(names)
        half_a, half_b = train_test_split(
            np.arange(len(train)), test_size=0.5,
            stratify=labels[train], random_state=seed)
        quarter_a, rest_b = train_test_split(
            np.arange(len(train)), train_size=0.25,
            stratify=labels[train], random_state=seed)
        if (len(np.intersect1d(quarter_a, rest_b)) or
                len(quarter_a)+len(rest_b) != len(train)):
            raise ValueError("direction/fit data overlap")
        baseline, old_signs = fit_branch(
            design_train, labels[train], half_a, half_b, groups)
        candidate, new_signs = fit_branch(
            design_train, labels[train], quarter_a, rest_b, groups)
        def score(model):
            p = expit(np.asarray(model.booster_.predict(
                design_dev, raw_score=True)))
            threshold = best_validation_threshold(p, labels[dev])
            return evaluate(labels[dev], p, threshold), threshold
        old_eval, old_threshold = score(baseline)
        new_eval, new_threshold = score(candidate)
        prior_row = old["runs"][seed]["candidates"][1]
        if (abs(old_threshold-prior_row["selected_threshold"]) > 1e-9 or
                any(abs(old_eval[key]-prior_row["validation"][key]) > 1e-9
                    for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("half-split named group control did not replay")
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "reference_version": reference.version,
            "half_direction_count": len(half_a),
            "quarter_direction_count": len(quarter_a),
            "quarter_fit_count": len(rest_b),
            "direction_disagreements_vs_half": int(np.count_nonzero(
                old_signs != new_signs)),
            "baseline_half": {"threshold": old_threshold,
                              "validation": old_eval},
            "candidate_quarter": {"threshold": new_threshold,
                                  "validation": new_eval},
            "all_trees_single_named_source_group": True,
        })
        print({"seed": seed, "half_macro_f1": old_eval["macro_f1"],
               "quarter_macro_f1": new_eval["macro_f1"],
               "quarter_acc": new_eval["acc"]}, flush=True)
    half_mean = float(np.mean([r["baseline_half"]["validation"]["macro_f1"]
                               for r in runs]))
    quarter_mean = float(np.mean([
        r["candidate_quarter"]["validation"]["macro_f1"] for r in runs]))
    gate = (quarter_mean >= half_mean + 0.01 and all(
        r["candidate_quarter"]["validation"]["macro_f1"] >=
        r["baseline_half"]["validation"]["macro_f1"] - 0.002
        for r in runs))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "scope": "mgtab_train_dev_only_named_ab_group_direction_fraction_25_percent",
        "test_predictions_or_metrics_accessed": False,
        "test_labels_loaded_by_full_dataset_adapter": True,
        "test_labels_used_for_fit_or_selection": False,
        "source_features_sha256": sha256_file(data / "features.pt"),
        "previous_dev_sha256": sha256_file(PRIOR),
        "direction_fraction": 0.25,
        "selection_gate": "quarter-split mean dev Macro-F1 at least one point above half-split with every seed no more than 0.2 points lower",
        "half_mean_dev_macro_f1": half_mean,
        "quarter_mean_dev_macro_f1": quarter_mean,
        "gate_passed": gate,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
