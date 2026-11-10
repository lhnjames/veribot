"""Dev-only retraining of the twelve-concept BotSim model with literal zeros."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_botsim_literal_upvote_zero_dev import (
    ACTIVITY, DATA, DEPLOYMENT, FULL, NATIVE, OUT as SOURCE_AUDIT,
    ROOT, SELECTED, USERS, literal_ratios)
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


OUT = ROOT / "output/v3_botsim_literal_upvote_zero_12/dev_only.json"
MODEL_ROOT = OUT.parent
L2 = .01


def main() -> None:
    source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if (source_audit["status"] !=
            "explicit_raw_upvote_zero_recode_and_frozen_dev_effect_audited"
            or source_audit["activity_sha256"] != sha256_file(ACTIVITY)):
        raise ValueError("literal zero source audit differs")
    runs = []
    locked_groups = None
    for seed in (0, 1, 2):
        x, labels, ids, names, split, source = load(
            "botsim", DATA, USERS, seed, NATIVE)
        literal, _ = literal_ratios(ids)
        corrected = x.copy()
        corrected[:, names.index("avg_upvote_ratio")] = literal
        if not np.isfinite(corrected).all():
            raise ValueError("corrected BotSim source facts are nonfinite")
        old = json.loads((SELECTED / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        full = json.loads((FULL / f"seed{seed}/report.json")
                          .read_text(encoding="utf-8"))
        if (source["split_sha256"] != old["manifest"]["split_sha256"]
                or old["validation"]["errors"] != 0):
            raise ValueError("old source or frozen dev baseline differs")
        groups = full["reasoner"]["groups"]
        if locked_groups is None:
            locked_groups = groups
        elif groups != locked_groups:
            raise ValueError("BotSim atomic concept groups differ across seeds")
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(
            corrected[train], names)
        concept_train, _ = collapse_group_tails(
            reference.transform(corrected[train]), groups)
        concept_dev, _ = collapse_group_tails(
            reference.transform(corrected[dev]), groups)
        concept_groups = {group: [f"{group}:high_tail_max",
                                  f"{group}:low_tail_max"]
                          for group in groups}
        a, b = train_test_split(np.arange(len(train)), test_size=.5,
                                stratify=labels[train], random_state=seed)
        reasoner = MonotoneSplineReasoner(concept_groups, l2=L2)
        reasoner.fit(rows(concept_train, a), labels[train][a],
                     rows(concept_train, b), labels[train][b])
        probability = reasoner.predict_proba(concept_dev)
        threshold = best_validation_threshold(probability, labels[dev])
        metrics = evaluate(labels[dev], probability, threshold)
        if (len(concept_dev.names) != 12 or
                concept_dev.reference_version != reasoner.reference_version):
            raise ValueError("corrected twelve-concept model contract differs")
        out = MODEL_ROOT / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        reference.save(out / "reference.json.gz")
        (out / "reasoner.json").write_text(json.dumps(
            reasoner.manifest(), indent=2) + "\n", encoding="utf-8")
        runs.append({
            "seed": seed,
            "split_sha256": source["split_sha256"],
            "users_sha256": sha256_file(USERS),
            "activity_sha256": sha256_file(ACTIVITY),
            "native_graph_facts_sha256": sha256_file(NATIVE),
            "literal_zero_source_audit_sha256": sha256_file(SOURCE_AUDIT),
            "reference_version": reference.version,
            "reference_artifact_sha256": sha256_file(
                out / "reference.json.gz"),
            "reasoner_artifact_sha256": sha256_file(
                out / "reasoner.json"),
            "concept_count": len(concept_dev.names),
            "l2": L2,
            "A_and_B_disjoint": bool(not len(np.intersect1d(a, b))),
            "threshold": threshold,
            "validation": metrics,
            "previous_validation": old["validation"],
            "previous_threshold": old["manifest"]["threshold"],
        })
        print({"seed": seed, "dev_errors": metrics["errors"],
               "dev_acc": metrics["acc"],
               "dev_macro_f1": metrics["macro_f1"]}, flush=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "botsim_literal_zero_12_concept_A_B_retrain_official_dev_only",
        "preprocessing": "upvote_ratio explicit zero preserved; .5 only for null/missing or no posts",
        "fixed_l2_from_prior_validation_lock": L2,
        "atomic_groups": locked_groups,
        "test_predictions_or_metrics_accessed": False,
        "prior_test_population_exploration": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
