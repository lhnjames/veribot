"""Lock the dev-selected neutral-anchor 150-fact MGTAB models before test."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_ab_monotone_evidence_tree_dev import directions, evidence_matrix
from screen_mgtab_ab_group_additive_monotone_tree_dev import (
    all_trees_one_group, source_groups)
from screen_mgtab_content_projection_evidence import transform_with_support
from screen_mgtab_positive_human_evidence_dev import neutral_training_rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference

from screen_mgtab_150_neutral_anchor_dev import ROOT, FROZEN, SOURCE, OUT


DEST = ROOT / "output/v3_mgtab_150_neutral_anchor"


def main() -> None:
    dev = json.loads(OUT.read_text(encoding="utf-8"))
    frozen_lock = json.loads((FROZEN / "selection_lock.json")
                             .read_text(encoding="utf-8"))
    if (dev["test_predictions_or_metrics_computed"] is not False or
            dev["frozen_selection_lock_sha256"] != sha256_file(
                FROZEN / "selection_lock.json")):
        raise ValueError("Dev result is not bound to frozen model")
    eligible = [x for x in dev["gate_results"]
                if x["passes_predeclared_dev_gate"]]
    if not eligible:
        raise ValueError("No candidate passed the predeclared dev gate")
    chosen = max(eligible, key=lambda x: (
        x["mean_neutral_threshold_macro_f1"], -x["ratio"]))
    ratio = float(chosen["ratio"])
    chosen_index = list(dev["ratios_predeclared"]).index(ratio)
    args = SimpleNamespace(
        dataset="mgtab",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB"),
        account_file=None, native_graph_facts=None,
        structural_facts=ROOT / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=SOURCE)
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((SOURCE / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        facts, labels, names, split = selected_facts(args, seed, old)
        train, validation = split["train"], split["val"]
        if (old["manifest"]["split_sha256"] !=
                dev["runs"][seed]["split_sha256"] or
                old["manifest"]["split_sha256"] !=
                frozen_lock["split_sha256_by_seed"][str(seed)]):
            raise ValueError("MGTAB official split changed")
        reference = LabelBlindReference(min_context_n=30).fit(
            facts[train], names)
        if reference.version != dev["runs"][seed]["reference_version"]:
            raise ValueError("Training reference changed")
        design = evidence_matrix(transform_with_support(
            reference, facts, split["supports"]))
        a, b = train_test_split(train, test_size=0.5,
                                stratify=labels[train], random_state=seed)
        signs = directions(design[a], labels[a])
        groups = source_groups(names, old["reasoner"]["groups"])
        fit_x, fit_y, weights = neutral_training_rows(
            design[b], labels[b], ratio)
        model = LGBMClassifier(
            objective="binary", n_estimators=400, num_leaves=31,
            min_child_samples=50, reg_lambda=10.0, learning_rate=0.05,
            n_jobs=4, random_state=0, deterministic=True,
            force_col_wise=True, monotone_constraints=signs.tolist(),
            monotone_constraints_method="advanced",
            interaction_constraints=groups, verbosity=-1)
        model.fit(fit_x, fit_y, sample_weight=weights)
        group_ok, group_counts = all_trees_one_group(model, groups)
        if not group_ok:
            raise ValueError("Fitted tree crossed source groups")
        zero = float(model.predict_proba(
            np.zeros((1, design.shape[1])))[:, 1][0])
        p = model.predict_proba(design[validation])[:, 1]
        metrics = evaluate(labels[validation], p, zero)
        expected = dev["runs"][seed]["candidates"][chosen_index]
        if (abs(zero - expected["zero_evidence_probability"]) > 1e-12 or
                any(abs(metrics[key] - expected[
                    "neutral_threshold_validation"][key]) > 1e-12
                    for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("Locked model fails exact dev replay")
        run_dir = DEST / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.joblib"
        contract_path = run_dir / "direction_contract.npz"
        joblib.dump(model, model_path)
        np.savez_compressed(contract_path,
                            monotone_constraints=signs,
                            group_names=np.asarray(list(old["reasoner"][
                                "groups"])) )
        runs.append({
            "seed": seed,
            "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "reference_path": str(FROZEN / f"seed{seed}/label_blind_reference.json.gz"),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "direction_contract_path": str(contract_path),
            "direction_contract_sha256": sha256_file(contract_path),
            "zero_evidence_threshold": zero,
            "validation": metrics,
            "single_source_group_tree_count": group_counts,
        })
        print({"seed": seed, "threshold": zero,
               "validation_acc": metrics["acc"]}, flush=True)
    (DEST / "selection_lock.json").write_text(json.dumps({
        "scope": "MGTAB_150_fact_neutral_anchor_selected_on_dev_before_test",
        "selection_rule": "highest_mean_neutral_threshold_dev_macro_f1_among_predeclared_gate_passes",
        "chosen_anchor_ratio": ratio,
        "chosen_dev_gate": chosen,
        "dev_result_sha256": sha256_file(OUT),
        "frozen_selection_lock_sha256": sha256_file(
            FROZEN / "selection_lock.json"),
        "test_predictions_or_metrics_used_for_selection": False,
        "published_source_features_sha256": frozen_lock[
            "source_features_sha256"],
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(DEST / "selection_lock.json")


if __name__ == "__main__":
    main()
