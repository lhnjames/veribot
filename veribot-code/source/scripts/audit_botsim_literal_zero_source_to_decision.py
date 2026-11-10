"""Independent raw-record and frozen-decision audit of corrected BotSim model."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_botsim_twelve_independent_raw_truth import raw_account
from gate0_shallow_tree import sha256_file
from verifiable_bot.botsim_literal_zero_runtime import BotSimLiteralZeroRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/certificates/botsim_literal_zero_source_to_decision.json"
FOLDER = ROOT / "output/v3_botsim_literal_upvote_zero_12"


def independent_facts(observed_means_only: bool = False
                      ) -> tuple[list[str], np.ndarray]:
    with (ROOT / "data/raw/botsim_24/Users.csv").open(
            "r", encoding="utf-8-sig", newline="") as stream:
        users = list(csv.DictReader(stream))
    activity = json.loads((ROOT / "data/raw/botsim_24/user_post_comment.json")
                          .read_text(encoding="utf-8"))
    ids = [row["user_id"] for row in users]
    rows = []
    for row in users:
        uid = row["user_id"]
        record = activity[uid]
        values = raw_account(row, record)
        posts = record.get("posts", [])
        ratios = []
        for post in posts:
            raw = post.get("upvote_ratio")
            value = .5 if raw is None else float(raw)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid raw upvote ratio")
            ratios.append(value)
        values[11] = float(np.float32(sum(ratios) / len(ratios) if ratios else .5))
        if observed_means_only:
            if not posts:
                for index in (10, 11, 12):
                    values[index] = float("nan")
            if not record.get("comment_1"):
                for index in (13, 14):
                    values[index] = float("nan")
        rows.append(values)
    with np.load(ROOT / "data/processed/botsim_v3/native_graph_facts.npz",
                 allow_pickle=False) as archive:
        if archive["ids"].astype(str).tolist() != ids:
            raise ValueError("native graph account order differs")
        native = archive["values"].copy()
    return ids, np.column_stack((np.asarray(rows, dtype=np.float32), native))


def main() -> None:
    ids, truth = independent_facts()
    runs = []
    for seed in (0, 1, 2):
        runtime = BotSimLiteralZeroRuntime(ROOT, seed)
        if runtime.ids != ids or not np.array_equal(runtime.facts, truth):
            raise ValueError("production facts differ from independent raw parser")
        saved_path = FOLDER / f"seed{seed}/test_predictions.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))["predictions"]
        decisions = runtime.predict_many([row["account_id"] for row in saved])
        max_probability = max_logit = max_fact = 0.0
        available = 0
        for recorded, decision in zip(saved, decisions, strict=True):
            if (recorded["account_id"] != decision["account_id"] or
                    recorded["prediction"] != decision["prediction"] or
                    len(decision["terms"]) != 12):
                raise ValueError("fixed decision or evidence vocabulary differs")
            max_probability = max(max_probability, abs(
                recorded["probability"] - decision["probability"]))
            max_logit = max(max_logit, abs(decision["logit"] - (
                decision["intercept"] + math.fsum(
                    term["logit_contribution"] for term in decision["terms"]))))
            for term in decision["terms"]:
                fact_name = term["source_fact_name"]
                if fact_name is None:
                    continue
                available += 1
                if term["source_locator"] is None:
                    raise ValueError("selected concept lacks a source locator")
                expected = truth[runtime.index[decision["account_id"]],
                                 runtime.reference.names.index(fact_name)]
                max_fact = max(max_fact, abs(term["source_fact_value"] -
                                             float(expected)))
        if max_probability > 1e-12 or max_logit > 1e-9 or max_fact > 1e-6:
            raise ValueError("raw facts, model decision or evidence sum differ")
        runs.append({
            "seed": seed, "full_population_accounts": len(ids),
            "full_population_fact_cells": int(truth.size),
            "fixed_test_accounts": len(saved),
            "available_winning_facts": available,
            "max_probability_difference": max_probability,
            "max_logit_sum_difference": max_logit,
            "max_winning_source_fact_difference": max_fact,
            "saved_predictions_sha256": sha256_file(saved_path),
        })
        print(runs[-1], flush=True)
    OUT.write_text(json.dumps({
        "status": "literal_zero_raw_source_and_all_fixed_test_decisions_verified",
        "raw_users_sha256": sha256_file(ROOT / "data/raw/botsim_24/Users.csv"),
        "raw_activity_sha256": sha256_file(
            ROOT / "data/raw/botsim_24/user_post_comment.json"),
        "fixed_test_report_sha256": sha256_file(FOLDER / "fixed_test_report.json"),
        "independent_parser_uses_training_feature_function": False,
        "semantic_validity_inferred": False,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
