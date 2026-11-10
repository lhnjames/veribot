"""Train-only one-cue BotSim baseline to expose simulator-linked separation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_botsim_literal_zero_source_to_decision import independent_facts
from gate0_shallow_tree import evaluate, load, sha256_file
from screen_botsim_observed_means_dev import DATA, NATIVE, ROOT, USERS
from verifiable_bot.botsim_literal_zero_runtime import BASE_NAMES, NATIVE_NAMES


OUT = ROOT / "research/certificates/botsim_single_upvote_cue_baseline.json"
FACT = "avg_upvote_ratio"


def categories(values: np.ndarray) -> np.ndarray:
    return np.where(~np.isfinite(values), 2, np.where(values == 0, 0, 1))


def main() -> None:
    raw_ids, facts = independent_facts(observed_means_only=True)
    values = facts[:, (BASE_NAMES + NATIVE_NAMES).index(FACT)]
    states = categories(values)
    runs = []
    for seed in (0, 1, 2):
        _, labels, ids, _, split, source = load("botsim", DATA, USERS, seed, NATIVE)
        if ids != raw_ids:
            raise ValueError("independent raw parser differs from official split order")
        train = split["train"]
        learned = {}
        for state in (0, 1, 2):
            group = train[states[train] == state]
            counts = np.bincount(labels[group], minlength=2)
            learned[state] = int(np.argmax(counts))
        if learned != {0: 1, 1: 0, 2: 0}:
            raise ValueError("unexpected train-only category mapping")
        rows = {}
        for part in ("train", "val", "test"):
            indices = split[part]
            pred = np.asarray([learned[int(state)] for state in states[indices]])
            p = np.where(pred == 1, .999, .001)
            rows[part] = {
                **evaluate(labels[indices], p, .5),
                "correct": int(np.count_nonzero(pred == labels[indices])),
                "bot_missing": int(np.count_nonzero((labels[indices] == 1) & (states[indices] == 2))),
                "human_missing": int(np.count_nonzero((labels[indices] == 0) & (states[indices] == 2))),
            }
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "train_only_state_to_class": {str(key): value for key, value in learned.items()},
                     "state_meaning": {"0": "observed_zero", "1": "observed_positive", "2": "undefined_no_posts"},
                     "metrics": rows})
        print({"seed": seed, "dev_correct": rows["val"]["correct"],
               "test_correct": rows["test"]["correct"]}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "botsim_train_only_single_raw_fact_shortcut_diagnostic",
        "fact": FACT,
        "raw_users_sha256": sha256_file(ROOT / "data/raw/botsim_24/Users.csv"),
        "raw_activity_sha256": sha256_file(ROOT / "data/raw/botsim_24/user_post_comment.json"),
        "independent_raw_parser": True,
        "prior_test_population_exploration": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
