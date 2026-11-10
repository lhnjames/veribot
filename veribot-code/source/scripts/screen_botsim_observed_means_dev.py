"""Train BotSim concepts using only observed, defined source averages."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_botsim_literal_upvote_zero_dev import (
    ACTIVITY, DATA, FULL, NATIVE, ROOT, USERS, literal_ratios)
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


FOLDER = ROOT / "output/v3_botsim_observed_means_12"
OUT = FOLDER / "dev_only.json"
OLD = ROOT / "output/v3_botsim_literal_upvote_zero_12/dev_only.json"
L2 = .01
POST_MEANS = ("avg_post_score", "avg_upvote_ratio", "avg_post_len")
COMMENT_MEANS = ("avg_comment_score", "avg_comment_len")
PREPROCESSING = (
    "literal zero preserved; post means missing without posts; "
    "comment means missing without first-party comments")


def corrected_facts(x: np.ndarray, names: list[str], ids: list[str],
                    activity: dict) -> tuple[np.ndarray, dict]:
    literal, stats = literal_ratios(ids)
    corrected = x.copy()
    corrected[:, names.index("avg_upvote_ratio")] = literal
    no_posts = np.asarray([not activity[uid].get("posts") for uid in ids])
    no_comments = np.asarray([not activity[uid].get("comment_1") for uid in ids])
    for name in POST_MEANS:
        corrected[no_posts, names.index(name)] = np.nan
    for name in COMMENT_MEANS:
        corrected[no_comments, names.index(name)] = np.nan
    if np.isinf(corrected).any():
        raise ValueError("source facts contain infinity")
    return corrected, {"no_posts": int(no_posts.sum()),
                       "no_first_party_comments": int(no_comments.sum()),
                       "missing_fact_cells": int(np.isnan(corrected).sum()),
                       "raw_post_statistics": stats}


def main() -> None:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    FOLDER.mkdir(parents=True, exist_ok=True)
    runs = []
    locked_groups = None
    for seed in (0, 1, 2):
        x, labels, ids, names, split, source = load(
            "botsim", DATA, USERS, seed, NATIVE)
        corrected, counts = corrected_facts(x, names, ids, activity)
        prior = old["runs"][seed]
        if (source["split_sha256"] != prior["split_sha256"] or
                sha256_file(ACTIVITY) != prior["activity_sha256"] or
                sha256_file(USERS) != prior["users_sha256"] or
                sha256_file(NATIVE) != prior["native_graph_facts_sha256"]):
            raise ValueError("source or official split differs from corrected predecessor")
        full = json.loads((FULL / f"seed{seed}/report.json")
                          .read_text(encoding="utf-8"))
        groups = full["reasoner"]["groups"]
        if locked_groups is None:
            locked_groups = groups
        elif groups != locked_groups:
            raise ValueError("atomic concept groups differ across seeds")
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
            raise ValueError("twelve-concept model contract differs")
        directory = FOLDER / f"seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        reference_path = directory / "reference.json.gz"
        reasoner_path = directory / "reasoner.json"
        reference.save(reference_path)
        reasoner_path.write_text(json.dumps(reasoner.manifest(), indent=2) + "\n",
                                 encoding="utf-8")
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "users_sha256": sha256_file(USERS),
            "activity_sha256": sha256_file(ACTIVITY),
            "native_graph_facts_sha256": sha256_file(NATIVE),
            "reference_version": reference.version,
            "reference_artifact_sha256": sha256_file(reference_path),
            "reasoner_artifact_sha256": sha256_file(reasoner_path),
            "concept_count": len(concept_dev.names), "l2": L2,
            "A_and_B_disjoint": bool(not len(np.intersect1d(a, b))),
            "threshold": threshold, "validation": metrics,
            "raw_source_counts": counts,
        })
        print({"seed": seed, "dev_errors": metrics["errors"],
               "dev_acc": metrics["acc"], "dev_macro_f1": metrics["macro_f1"]},
              flush=True)
    OUT.write_text(json.dumps({
        "scope": "botsim_twelve_concepts_defined_source_averages_dev_only",
        "preprocessing": PREPROCESSING,
        "atomic_groups": locked_groups,
        "fixed_l2_from_prior_validation_lock": L2,
        "test_predictions_or_metrics_accessed": False,
        "prior_test_population_exploration": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
