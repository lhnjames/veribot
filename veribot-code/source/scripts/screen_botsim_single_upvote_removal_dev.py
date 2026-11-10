"""Prespecified dev-only test of removing BotSim's near-label upvote ratio."""

from __future__ import annotations

import json
from pathlib import Path

from gate0_shallow_tree import load, sha256_file
from screen_botsim_observed_means_dev import (
    ACTIVITY, DATA, FOLDER as CURRENT, NATIVE, ROOT, USERS,
    corrected_facts)
from screen_botsim_simulator_shortcuts_dev import candidate


OUT = ROOT / "output/v3_botsim_single_upvote_removed/dev_only.json"
EXCLUDED = ("avg_upvote_ratio",)


def main() -> None:
    full = json.loads((CURRENT / "dev_only.json").read_text(encoding="utf-8"))
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    runs = []
    for seed in (0, 1, 2):
        raw, labels, ids, names, split, source = load(
            "botsim", DATA, USERS, seed, NATIVE)
        facts, _ = corrected_facts(raw, names, ids, activity)
        prior = full["runs"][seed]
        if (source["split_sha256"] != prior["split_sha256"] or
                sha256_file(ACTIVITY) != prior["activity_sha256"]):
            raise ValueError("Current source or official split changed")
        result = candidate(
            facts, labels, names, split["train"], split["val"],
            seed, full["atomic_groups"], EXCLUDED)
        if EXCLUDED[0] in result["retained_source_facts"]:
            raise ValueError("Upvote ratio remains in candidate")
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     **result})
        print({"seed": seed, "dev_errors": result["validation"]["errors"],
               "dev_macro_f1": result["validation"]["macro_f1"]},
              flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "BotSim_single_upvote_ratio_removal_dev_only",
        "excluded_source_facts": list(EXCLUDED),
        "fixed_test_predictions_or_metrics_accessed": False,
        "source_users_sha256": sha256_file(USERS),
        "source_activity_sha256": sha256_file(ACTIVITY),
        "source_native_graph_sha256": sha256_file(NATIVE),
        "all_seeds_above_99_4_percent_on_dev": all(
            row["validation"]["acc"] > .994 for row in runs),
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
