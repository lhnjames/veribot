"""Independently replay all defined BotSim facts and frozen test decisions."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_botsim_literal_zero_source_to_decision import independent_facts
from gate0_shallow_tree import sha256_file
from verifiable_bot.botsim_literal_zero_runtime import BotSimObservedMeanRuntime


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_botsim_observed_means_12"
OUT = ROOT / "research/certificates/botsim_observed_means_source_to_decision.json"
POST_MEANS = {"avg_post_score", "avg_upvote_ratio", "avg_post_len"}
COMMENT_MEANS = {"avg_comment_score", "avg_comment_len"}


def main() -> None:
    ids, truth = independent_facts(observed_means_only=True)
    activity_path = ROOT / "data/raw/botsim_24/user_post_comment.json"
    activity = json.loads(activity_path.read_text(encoding="utf-8"))
    if int(np.isnan(truth).sum()) != 516:
        raise ValueError("empty-denominator fact count differs")
    runs = []
    for seed in (0, 1, 2):
        runtime = BotSimObservedMeanRuntime(ROOT, seed)
        if runtime.ids != ids or not np.array_equal(
                runtime.facts, truth, equal_nan=True):
            raise ValueError("production source facts differ from independent raw parser")
        saved_path = FOLDER / f"seed{seed}/test_predictions.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))["predictions"]
        decisions = runtime.predict_many([row["account_id"] for row in saved])
        max_probability = max_logit = max_fact = 0.0
        available = 0
        for recorded, decision in zip(saved, decisions, strict=True):
            uid = decision["account_id"]
            if (uid != recorded["account_id"] or
                    decision["prediction"] != recorded["prediction"] or
                    len(decision["terms"]) != 12):
                raise ValueError("fixed decision or concept vocabulary differs")
            max_probability = max(max_probability, abs(
                recorded["probability"] - decision["probability"]))
            max_logit = max(max_logit, abs(decision["logit"] - (
                decision["intercept"] + math.fsum(
                    term["logit_contribution"] for term in decision["terms"]))))
            for term in decision["terms"]:
                fact_name = term["source_fact_name"]
                if fact_name is None:
                    continue
                if ((fact_name in POST_MEANS and not activity[uid].get("posts")) or
                        (fact_name in COMMENT_MEANS and
                         not activity[uid].get("comment_1"))):
                    raise ValueError("undefined average won a source concept")
                if term["source_locator"] is None:
                    raise ValueError("winning fact lacks a source locator")
                expected = truth[runtime.index[uid],
                                 runtime.reference.names.index(fact_name)]
                if not np.isfinite(expected):
                    raise ValueError("winning fact lacks a defined observation")
                max_fact = max(max_fact, abs(
                    term["source_fact_value"] - float(expected)))
                available += 1
        if max_probability > 1e-12 or max_logit > 1e-9 or max_fact > 1e-6:
            raise ValueError("source, probability or additive decision differs")
        runs.append({
            "seed": seed, "population_accounts_checked": len(ids),
            "population_fact_cells_checked": int(truth.size),
            "undefined_average_fact_cells": int(np.isnan(truth).sum()),
            "fixed_test_accounts_checked": len(saved),
            "available_winning_facts_checked": available,
            "undefined_average_winners": 0,
            "max_probability_difference": max_probability,
            "max_logit_sum_difference": max_logit,
            "max_winning_source_fact_difference": max_fact,
            "saved_predictions_sha256": sha256_file(saved_path),
        })
        print(runs[-1], flush=True)
    OUT.write_text(json.dumps({
        "status": "botsim_observed_means_full_raw_source_and_decisions_verified",
        "raw_users_sha256": sha256_file(ROOT / "data/raw/botsim_24/Users.csv"),
        "raw_activity_sha256": sha256_file(activity_path),
        "native_graph_facts_sha256": sha256_file(
            ROOT / "data/processed/botsim_v3/native_graph_facts.npz"),
        "fixed_test_report_sha256": sha256_file(FOLDER / "fixed_test_report.json"),
        "independent_parser_uses_training_feature_function": False,
        "semantic_bot_behavior_validity_inferred": False,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
