"""Measure published BotSim cue prevalence by label and official split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_botsim_literal_zero_source_to_decision import independent_facts
from gate0_shallow_tree import load, sha256_file
from screen_botsim_observed_means_dev import DATA, NATIVE, ROOT, USERS
from verifiable_bot.botsim_literal_zero_runtime import BASE_NAMES, NATIVE_NAMES


OUT = ROOT / "research/certificates/botsim_simulator_cue_prevalence.json"
CUES = (
    "has_description", "description_len", "avg_post_score",
    "n_comments_received", "avg_comment_score", "avg_upvote_ratio",
)


def describe(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "accounts": len(values), "observed": len(finite),
        "missing": int(np.count_nonzero(~np.isfinite(values))),
        "zero": int(np.count_nonzero(finite == 0)),
        "positive": int(np.count_nonzero(finite > 0)),
        "median_observed": float(np.median(finite)) if len(finite) else None,
    }


def main() -> None:
    raw_ids, facts = independent_facts(observed_means_only=True)
    names = BASE_NAMES + NATIVE_NAMES
    runs = []
    for seed in (0, 1, 2):
        _, labels, ids, _, split, source = load("botsim", DATA, USERS, seed, NATIVE)
        if ids != raw_ids:
            raise ValueError("independent parser and official split account order differ")
        split_result = {}
        for part in ("train", "val", "test"):
            indices = split[part]
            split_result[part] = {
                label_name: {
                    name: describe(facts[indices[labels[indices] == label], names.index(name)])
                    for name in CUES
                }
                for label, label_name in ((0, "human"), (1, "bot"))
            }
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "by_split_and_label": split_result})
    OUT.write_text(json.dumps({
        "scope": "botsim_published_source_cue_prevalence_diagnostic",
        "raw_users_sha256": sha256_file(ROOT / "data/raw/botsim_24/Users.csv"),
        "raw_activity_sha256": sha256_file(ROOT / "data/raw/botsim_24/user_post_comment.json"),
        "independent_raw_parser": True,
        "labels_used_for_diagnostic_only": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    for run in runs:
        seed = run["seed"]
        for part in ("train", "val", "test"):
            bot = run["by_split_and_label"][part]["bot"]["avg_upvote_ratio"]
            human = run["by_split_and_label"][part]["human"]["avg_upvote_ratio"]
            print({"seed": seed, "split": part, "bot_upvote": bot,
                   "human_upvote": human}, flush=True)
    print(OUT)


if __name__ == "__main__":
    main()
