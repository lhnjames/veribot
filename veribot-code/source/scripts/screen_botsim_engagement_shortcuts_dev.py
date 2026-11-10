"""Second dev-only screen removing the identified BotSim engagement cues."""

from __future__ import annotations

import json

from gate0_shallow_tree import load, sha256_file
from screen_botsim_observed_means_dev import (
    ACTIVITY, DATA, FOLDER as CURRENT, NATIVE, ROOT, USERS,
    corrected_facts)
from screen_botsim_simulator_shortcuts_dev import candidate


OUT = ROOT / "output/v3_botsim_engagement_shortcuts/dev_only.json"
CONFIGS = {
    "drop_upvote_and_comment_score": (
        "avg_upvote_ratio", "avg_comment_score"),
    "drop_upvote_and_post_score": (
        "avg_upvote_ratio", "avg_post_score"),
    "drop_all_three_engagement_means": (
        "avg_upvote_ratio", "avg_comment_score", "avg_post_score"),
}


def main() -> None:
    full = json.loads((CURRENT / "dev_only.json").read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "research/certificates/botsim_remaining_single_fact_dev.json")
                       .read_text(encoding="utf-8"))
    assert prior["fixed_test_predictions_or_metrics_accessed"] is False
    assert all(row["single_fact_controls"][0]["fact"] == "avg_comment_score"
               for row in prior["runs"])
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    runs = []
    for seed in (0, 1, 2):
        raw, labels, ids, names, split, source = load(
            "botsim", DATA, USERS, seed, NATIVE)
        facts, _ = corrected_facts(raw, names, ids, activity)
        assert source["split_sha256"] == full["runs"][seed]["split_sha256"]
        results = {name: candidate(
            facts, labels, names, split["train"], split["val"],
            seed, full["atomic_groups"], excluded)
            for name, excluded in CONFIGS.items()}
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "configs": results})
        print({"seed": seed, "dev_errors": {
            name: row["validation"]["errors"]
            for name, row in results.items()}}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "BotSim_engagement_shortcut_removal_second_dev_screen",
        "development_exploration_round": 2,
        "fixed_test_predictions_or_metrics_accessed": False,
        "candidate_configs": {name: list(excluded)
                              for name, excluded in CONFIGS.items()},
        "source_users_sha256": sha256_file(USERS),
        "source_activity_sha256": sha256_file(ACTIVITY),
        "source_native_graph_sha256": sha256_file(NATIVE),
        "all_three_cues_removed_dev_above_99_4_percent": all(
            run["configs"]["drop_all_three_engagement_means"][
                "validation"]["acc"] > .994 for run in runs),
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
