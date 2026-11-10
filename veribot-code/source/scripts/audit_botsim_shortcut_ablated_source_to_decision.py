"""Audit the shortcut-ablated BotSim model against an independent raw parser."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_botsim_literal_zero_source_to_decision import independent_facts
from gate0_shallow_tree import sha256_file
from verifiable_bot.botsim_literal_zero_runtime import (
    BASE_NAMES, NATIVE_NAMES, BotSimShortcutAblatedRuntime)


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_botsim_shortcut_ablated_10"
OUT = ROOT / "research/certificates/botsim_shortcut_ablated_source_to_decision.json"


def main() -> None:
    ids, full_truth = independent_facts(observed_means_only=True)
    lock = json.loads((FOLDER / "dev_only.json").read_text(encoding="utf-8"))
    retained = lock["retained_source_facts"]
    excluded = lock["excluded_source_facts"]
    if set(excluded) & set(retained) or len(retained) != 17:
        raise ValueError("excluded facts entered the retained model")
    truth = full_truth[:, [(BASE_NAMES + NATIVE_NAMES).index(name) for name in retained]]
    if any(name in retained for name in excluded):
        raise ValueError("simulator-linked cue retained")
    runs = []
    for seed in (0, 1, 2):
        runtime = BotSimShortcutAblatedRuntime(ROOT, seed)
        if (runtime.ids != ids or runtime.reference.names != tuple(retained) or
                not np.array_equal(runtime.facts, truth, equal_nan=True)):
            raise ValueError("deployed source facts differ from independent raw parser")
        saved_path = FOLDER / f"seed{seed}/test_predictions.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))["predictions"]
        decisions = runtime.predict_many([row["account_id"] for row in saved])
        max_probability = max_logit = max_fact = 0.0
        pivotal_by_group = {name: 0 for name in runtime.atomic_groups}
        available_winners = 0
        class_changes_without_evidence = 0
        for record, decision in zip(saved, decisions, strict=True):
            if (record["account_id"] != decision["account_id"] or
                    record["prediction"] != decision["prediction"] or
                    len(decision["terms"]) != 10):
                raise ValueError("saved decision differs from source runtime")
            max_probability = max(max_probability, abs(record["probability"] - decision["probability"]))
            effects = [term["logit_contribution"] for term in decision["terms"]]
            max_logit = max(max_logit, abs(decision["logit"] -
                                            (decision["intercept"] + math.fsum(effects))))
            threshold_logit = float(logit(decision["threshold"]))
            baseline = int(expit(decision["intercept"]) >= decision["threshold"])
            class_changes_without_evidence += int(baseline != decision["prediction"])
            for term in decision["terms"]:
                fact_name = term["source_fact_name"]
                if fact_name is None:
                    continue
                if fact_name not in retained or fact_name in excluded or term["source_locator"] is None:
                    raise ValueError("winner has an excluded or untraceable source")
                expected = truth[runtime.index[decision["account_id"]], retained.index(fact_name)]
                if not np.isfinite(expected):
                    raise ValueError("undefined source fact won a concept")
                max_fact = max(max_fact, abs(term["source_fact_value"] - float(expected)))
                available_winners += 1
                removed_class = int(decision["logit"] - term["logit_contribution"] >= threshold_logit)
                if removed_class != decision["prediction"]:
                    pivotal_by_group[term["name"].split(":", 1)[0]] += 1
        if max_probability > 1e-12 or max_logit > 1e-9 or max_fact > 1e-6:
            raise ValueError("source or additive decision did not replay")
        runs.append({
            "seed": seed, "population_accounts_checked": len(ids),
            "retained_fact_cells_checked": int(truth.size),
            "missing_retained_fact_cells": int(np.isnan(truth).sum()),
            "fixed_test_accounts_checked": len(saved),
            "available_winning_facts_checked": available_winners,
            "class_changes_without_all_evidence": class_changes_without_evidence,
            "single_concept_pivotal_by_group": pivotal_by_group,
            "max_probability_difference": max_probability,
            "max_logit_sum_difference": max_logit,
            "max_winning_source_fact_difference": max_fact,
            "saved_predictions_sha256": sha256_file(saved_path),
        })
        print({"seed": seed, "changed_without_evidence": class_changes_without_evidence,
               "pivotal_by_group": pivotal_by_group}, flush=True)
    OUT.write_text(json.dumps({
        "status": "botsim_shortcut_ablated_full_source_and_decision_audit_passed",
        "excluded_source_facts": excluded,
        "retained_source_facts": retained,
        "independent_parser_uses_training_feature_function": False,
        "semantic_bot_behavior_validity_inferred": False,
        "raw_users_sha256": sha256_file(ROOT / "data/raw/botsim_24/Users.csv"),
        "raw_activity_sha256": sha256_file(ROOT / "data/raw/botsim_24/user_post_comment.json"),
        "dev_lock_sha256": sha256_file(FOLDER / "dev_only.json"),
        "fixed_test_report_sha256": sha256_file(FOLDER / "fixed_test_report.json"),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
