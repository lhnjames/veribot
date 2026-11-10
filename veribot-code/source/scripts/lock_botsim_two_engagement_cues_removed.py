"""Freeze the only two-cue BotSim removal passing all three dev splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_botsim_observed_means_dev import (
    ACTIVITY, DATA, FOLDER as CURRENT, NATIVE, ROOT, USERS,
    corrected_facts)
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


SCREEN = ROOT / "output/v3_botsim_engagement_shortcuts/dev_only.json"
OUT_DIR = ROOT / "output/v3_botsim_engagement_shortcuts/selected"
CONFIG = "drop_upvote_and_comment_score"
PREPROCESSING = (
    "literal zero preserved; post means missing without posts; "
    "comment means missing without first-party comments")


def main() -> None:
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    full = json.loads((CURRENT / "dev_only.json").read_text(encoding="utf-8"))
    excluded = tuple(screen["candidate_configs"][CONFIG])
    if (screen["fixed_test_predictions_or_metrics_accessed"] is not False or
            set(excluded) != {"avg_upvote_ratio", "avg_comment_score"} or
            any(row["configs"][CONFIG]["validation"]["errors"] > 2
                for row in screen["runs"])):
        raise ValueError("Two-cue candidate did not pass dev-only gate")
    groups = {name: [fact for fact in members if fact not in excluded]
              for name, members in full["atomic_groups"].items()}
    groups = {name: members for name, members in groups.items() if members}
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in (0, 1, 2):
        x, labels, ids, names, split, source = load(
            "botsim", DATA, USERS, seed, NATIVE)
        facts, _ = corrected_facts(x, names, ids, activity)
        keep = [j for j, name in enumerate(names) if name not in excluded]
        kept_names = [names[j] for j in keep]
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(
            facts[train][:, keep], kept_names)
        train_concepts, _ = collapse_group_tails(
            reference.transform(facts[train][:, keep]), groups)
        dev_concepts, _ = collapse_group_tails(
            reference.transform(facts[dev][:, keep]), groups)
        concept_groups = {name: [f"{name}:high_tail_max",
                                 f"{name}:low_tail_max"] for name in groups}
        a, b = train_test_split(np.arange(len(train)), test_size=.5,
                                stratify=labels[train], random_state=seed)
        reasoner = MonotoneSplineReasoner(concept_groups, l2=.01)
        reasoner.fit(rows(train_concepts, a), labels[train][a],
                     rows(train_concepts, b), labels[train][b])
        probability = reasoner.predict_proba(dev_concepts)
        threshold = best_validation_threshold(probability, labels[dev])
        metrics = evaluate(labels[dev], probability, threshold)
        prior = screen["runs"][seed]["configs"][CONFIG]
        if (source["split_sha256"] != screen["runs"][seed]["split_sha256"] or
                metrics != prior["validation"] or
                threshold != prior["threshold"] or
                reference.version != prior["reference_version"] or
                tuple(kept_names) != tuple(prior["retained_source_facts"])):
            raise ValueError("Frozen candidate differs from dev screen")
        folder = OUT_DIR / f"seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        ref_path = folder / "reference.json.gz"
        model_path = folder / "reasoner.json"
        reference.save(ref_path)
        model_path.write_text(json.dumps(reasoner.manifest(), indent=2) + "\n",
                              encoding="utf-8")
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "users_sha256": sha256_file(USERS),
            "activity_sha256": sha256_file(ACTIVITY),
            "native_graph_facts_sha256": sha256_file(NATIVE),
            "reference_version": reference.version,
            "reference_artifact_sha256": sha256_file(ref_path),
            "reasoner_artifact_sha256": sha256_file(model_path),
            "threshold": threshold, "validation": metrics,
            "A_and_B_disjoint": bool(not len(np.intersect1d(a, b)))})
        print({"seed": seed, "dev_errors": metrics["errors"]}, flush=True)
    lock = {
        "scope": "BotSim_two_engagement_cues_removed_dev_locked",
        "screen_sha256": sha256_file(SCREEN),
        "selected_configuration": CONFIG,
        "excluded_source_facts": list(excluded),
        "retained_source_facts": kept_names,
        "atomic_groups": groups,
        "preprocessing": PREPROCESSING,
        "test_predictions_or_metrics_accessed": False,
        "prior_test_population_exploration": True,
        "runs": runs}
    (OUT_DIR / "dev_only.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(OUT_DIR / "dev_only.json")


if __name__ == "__main__":
    main()
