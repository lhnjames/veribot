"""Dev-only ablation of BotSim publication-linked profile and score cues."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from screen_botsim_observed_means_dev import (
    ACTIVITY, DATA, FOLDER as CURRENT, NATIVE, ROOT, USERS,
    corrected_facts)
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


OUT = ROOT / "output/v3_botsim_simulator_shortcut_ablation/dev_only.json"
CONFIGS = {
    "locked_full_control": (),
    "drop_presentation": ("has_description", "description_len"),
    "drop_score_zero_cues": ("avg_post_score", "n_comments_received",
                             "avg_comment_score"),
    "drop_presentation_and_score_zero_cues": (
        "has_description", "description_len", "avg_post_score",
        "n_comments_received", "avg_comment_score"),
    "drop_presentation_and_post_engagement": (
        "has_description", "description_len", "avg_post_score",
        "avg_upvote_ratio", "avg_comment_score", "n_comments_received"),
}


def candidate(x: np.ndarray, labels: np.ndarray, names: list[str],
              train: np.ndarray, dev: np.ndarray, seed: int,
              groups: dict[str, list[str]], excluded: tuple[str, ...]) -> dict:
    kept = [j for j, name in enumerate(names) if name not in excluded]
    kept_names = [names[j] for j in kept]
    kept_groups = {group: [name for name in members if name not in excluded]
                   for group, members in groups.items()}
    kept_groups = {group: members for group, members in kept_groups.items()
                   if members}
    if set(kept_names) != {name for members in kept_groups.values()
                          for name in members}:
        raise ValueError("shortcut ablation does not partition source facts")
    reference = LabelBlindReference(min_context_n=30).fit(
        x[train][:, kept], kept_names)
    train_concepts, _ = collapse_group_tails(
        reference.transform(x[train][:, kept]), kept_groups)
    dev_concepts, _ = collapse_group_tails(
        reference.transform(x[dev][:, kept]), kept_groups)
    concept_groups = {group: [f"{group}:high_tail_max",
                              f"{group}:low_tail_max"]
                      for group in kept_groups}
    a, b = train_test_split(np.arange(len(train)), test_size=.5,
                            stratify=labels[train], random_state=seed)
    reasoner = MonotoneSplineReasoner(concept_groups, l2=.01)
    reasoner.fit(rows(train_concepts, a), labels[train][a],
                 rows(train_concepts, b), labels[train][b])
    probability = reasoner.predict_proba(dev_concepts)
    threshold = best_validation_threshold(probability, labels[dev])
    return {
        "excluded_source_facts": list(excluded),
        "retained_source_facts": kept_names,
        "retained_concept_count": len(dev_concepts.names),
        "reference_version": reference.version,
        "threshold": threshold,
        "validation": evaluate(labels[dev], probability, threshold),
        "A_and_B_disjoint": bool(not len(np.intersect1d(a, b))),
    }


def main() -> None:
    locked = json.loads((CURRENT / "dev_only.json").read_text(encoding="utf-8"))
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    all_runs = []
    for seed in (0, 1, 2):
        raw, labels, ids, names, split, source = load(
            "botsim", DATA, USERS, seed, NATIVE)
        facts, _ = corrected_facts(raw, names, ids, activity)
        plan = locked["runs"][seed]
        if (source["split_sha256"] != plan["split_sha256"] or
                sha256_file(ACTIVITY) != plan["activity_sha256"]):
            raise ValueError("official BotSim split or source differs")
        train, dev = split["train"], split["val"]
        rows_by_config = {}
        for name, excluded in CONFIGS.items():
            rows_by_config[name] = candidate(
                facts, labels, names, train, dev, seed,
                locked["atomic_groups"], excluded)
        control = rows_by_config["locked_full_control"]
        if (abs(control["threshold"] - plan["threshold"]) > 1e-12 or
                control["validation"] != plan["validation"] or
                control["reference_version"] != plan["reference_version"]):
            raise ValueError("current model development control did not replay")
        all_runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                         "configs": rows_by_config})
        print({"seed": seed, "errors": {key: value["validation"]["errors"]
                                       for key, value in rows_by_config.items()}},
              flush=True)
    # A candidate enters fixed test only if every official dev seed keeps at
    # most two errors (ACC strictly above 99.4%). Prefer the widest removal.
    eligible = [name for name in CONFIGS if name != "locked_full_control" and
                all(run["configs"][name]["validation"]["errors"] <= 2
                    for run in all_runs)]
    selected = max(eligible, key=lambda name: (len(CONFIGS[name]), name)) if eligible else None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "botsim_observed_means_simulator_shortcut_train_dev_screen",
        "test_predictions_or_metrics_accessed": False,
        "source_users_sha256": sha256_file(USERS),
        "source_activity_sha256": sha256_file(ACTIVITY),
        "configurations": {name: list(excluded) for name, excluded in CONFIGS.items()},
        "eligibility_rule": "each_seed_dev_errors_at_most_two_then_widest_removal",
        "eligible_configs": eligible,
        "selected_for_next_stage": selected,
        "runs": all_runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
