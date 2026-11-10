"""Dev-only frozen-model check of withholding discordant BotSim summary facts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load, sha256_file
from screen_botsim_observed_means_dev import (
    ACTIVITY, DATA, FOLDER, NATIVE, ROOT, USERS, corrected_facts)
from verifiable_bot.botsim_literal_zero_runtime import BotSimObservedMeanRuntime
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


OUT = ROOT / "output/v3_botsim_summary_disagreement_gate/dev_only.json"
FIELDS = ("submission_num", "comment_num", "comment_num_1", "comment_num_2")


def main() -> None:
    lock = json.loads((FOLDER / "dev_only.json").read_text(encoding="utf-8"))
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    with USERS.open(encoding="utf-8-sig", newline="") as stream:
        users = list(csv.DictReader(stream))
    ids_from_users = [row["user_id"] for row in users]
    if len(users) != 2907 or set(ids_from_users) != set(activity):
        raise ValueError("BotSim published source population differs")
    disagrees = np.zeros((len(users), len(FIELDS)), dtype=bool)
    for i, row in enumerate(users):
        record = activity[row["user_id"]]
        counts = (len(record["posts"]),
                  len(record["comment_1"]) + len(record["comment_2"]),
                  len(record["comment_1"]), len(record["comment_2"]))
        disagrees[i] = [float(row[name]) != count
                        for name, count in zip(FIELDS, counts)]
    runs = []
    for seed in (0, 1, 2):
        runtime = BotSimObservedMeanRuntime(ROOT, seed)
        x, labels, ids, names, split, source = load(
            "botsim", DATA, USERS, seed, NATIVE)
        original, _ = corrected_facts(x, names, ids, activity)
        if (ids != runtime.ids or ids != ids_from_users or
                names != list(runtime.reference.names) or
                not np.array_equal(original, runtime.facts, equal_nan=True) or
                source["split_sha256"] != lock["runs"][seed]["split_sha256"]):
            raise ValueError("frozen runtime, source facts or official split differs")
        dev = split["val"]
        baseline, baseline_winners = collapse_group_tails(
            runtime.reference.transform(original[dev]), runtime.atomic_groups)
        baseline_probability = runtime.reasoner.predict_proba(baseline)
        baseline_metrics = evaluate(labels[dev], baseline_probability,
                                    runtime.threshold)
        locked_metrics = lock["runs"][seed]["validation"]
        if (baseline_metrics["errors"] != locked_metrics["errors"] or
                abs(baseline_metrics["macro_f1"] -
                    locked_metrics["macro_f1"]) > 1e-12):
            raise ValueError("dev baseline differs from frozen development lock")
        gated = original[dev].copy()
        masked_cells = 0
        for j, name in enumerate(FIELDS):
            mask = disagrees[dev, j]
            gated[mask, names.index(name)] = np.nan
            masked_cells += int(mask.sum())
        concepts, winners = collapse_group_tails(
            runtime.reference.transform(gated), runtime.atomic_groups)
        probability = runtime.reasoner.predict_proba(concepts)
        metrics = evaluate(labels[dev], probability, runtime.threshold)
        before = baseline_probability >= runtime.threshold
        after = probability >= runtime.threshold
        changed = before != after
        error_before = before != labels[dev]
        error_after = after != labels[dev]
        runs.append({
            "seed": seed,
            "official_dev_accounts": len(dev),
            "accounts_with_disagreed_summary": int(np.any(disagrees[dev], axis=1).sum()),
            "summary_fact_cells_masked": masked_cells,
            "changed_probability_accounts": int(np.count_nonzero(
                np.abs(probability - baseline_probability) > 1e-12)),
            "maximum_probability_change": float(np.max(np.abs(
                probability - baseline_probability))),
            "changed_concept_winner_slots": int(sum(np.count_nonzero(
                winners[name] != baseline_winners[name]) for name in winners)),
            "changed_class_accounts": int(changed.sum()),
            "correct_to_wrong": int(np.sum(~error_before & error_after)),
            "wrong_to_correct": int(np.sum(error_before & ~error_after)),
            "baseline_dev_metrics": baseline_metrics,
            "withheld_dev_metrics": metrics,
            "threshold_unchanged": runtime.threshold,
            "split_sha256": source["split_sha256"],
            "reference_artifact_sha256": lock["runs"][seed][
                "reference_artifact_sha256"],
            "reasoner_artifact_sha256": lock["runs"][seed][
                "reasoner_artifact_sha256"],
            "changed_account_ids": [ids[int(dev[i])] for i in np.where(changed)[0]],
        })
        print({"seed": seed, "masked_cells": masked_cells,
               "class_changes": int(changed.sum()),
               "baseline_errors": baseline_metrics["errors"],
               "withheld_errors": metrics["errors"]}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "botsim_frozen_summary_consistency_gate_official_dev_only",
        "test_predictions_or_metrics_accessed": False,
        "rule": "set a declared CSV count unavailable only if it differs from the corresponding number of published activity JSON objects; preserve the original frozen reference, reasoner and threshold",
        "source_sha256": {
            "Users.csv": sha256_file(USERS),
            "user_post_comment.json": sha256_file(ACTIVITY),
            "native_graph_facts.npz": sha256_file(NATIVE),
        },
        "runs": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
