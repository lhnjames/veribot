"""Train-only single-fact controls after removing BotSim upvote ratio."""

from __future__ import annotations

import json

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

from gate0_shallow_tree import load, sha256_file
from screen_botsim_observed_means_dev import (
    ACTIVITY, DATA, NATIVE, ROOT, USERS, corrected_facts)


SCREEN = ROOT / "output/v3_botsim_single_upvote_removed/dev_only.json"
OUT = ROOT / "research/certificates/botsim_remaining_single_fact_dev.json"


def main() -> None:
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if (screen["fixed_test_predictions_or_metrics_accessed"] is not False or
            screen["excluded_source_facts"] != ["avg_upvote_ratio"]):
        raise ValueError("Single-fact control differs from dev-only candidate")
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    runs = []
    for seed in (0, 1, 2):
        raw, y, ids, names, split, source = load(
            "botsim", DATA, USERS, seed, NATIVE)
        facts, _ = corrected_facts(raw, names, ids, activity)
        assert source["split_sha256"] == screen["runs"][seed]["split_sha256"]
        train, dev = split["train"], split["val"]
        controls = []
        for j, name in enumerate(names):
            if name == "avg_upvote_ratio":
                continue
            x_train = facts[train, j:j + 1]
            x_dev = facts[dev, j:j + 1]
            imputer = SimpleImputer(strategy="median", add_indicator=True,
                                    keep_empty_features=True)
            fit = imputer.fit_transform(x_train)
            test = imputer.transform(x_dev)
            tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10,
                                          random_state=seed)
            tree.fit(fit, y[train])
            pred = tree.predict(test)
            controls.append({"fact": name, "dev_correct": int(np.count_nonzero(
                pred == y[dev])), "dev_accounts": len(dev),
                "dev_accuracy": float(np.mean(pred == y[dev])),
                "train_correct": int(np.count_nonzero(
                    tree.predict(fit) == y[train]))})
        controls.sort(key=lambda row: (-row["dev_correct"], row["fact"]))
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "single_fact_controls": controls})
        print({"seed": seed, "top_fact": controls[0]["fact"],
               "top_dev_correct": controls[0]["dev_correct"],
               "dev_accounts": len(dev)}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "BotSim_remaining_single_fact_depth3_train_dev_control",
        "fixed_test_predictions_or_metrics_accessed": False,
        "excluded_fact": "avg_upvote_ratio",
        "tree_capacity": {"max_depth": 3, "min_samples_leaf": 10},
        "source_users_sha256": sha256_file(USERS),
        "source_activity_sha256": sha256_file(ACTIVITY),
        "runs": runs}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
