"""Bind schema-v2 model inputs to independently recomputed raw facts.

This audits factual input values, not behavioral semantics or whether a
candidate model uses a fact to change its decision.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from gate1_fox8_botsim_schema_dev import _missing_means, _source_facts
from audit_fox8_twenty_independent_raw_truth import (
    EVENTS as FOX8_EVENTS,
    NAMES as FOX8_RAW_NAMES,
    RAW as FOX8_RAW,
    raw_truth as fox8_raw_truth,
)
from audit_botsim_twelve_independent_raw_truth import (
    ACTIVITY as BOTSIM_ACTIVITY,
    BASE_NAMES as BOTSIM_BASE_NAMES,
    NAMES as BOTSIM_RAW_NAMES,
    NATIVE_DIR as BOTSIM_NATIVE_DIR,
    NATIVE_NAMES as BOTSIM_NATIVE_NAMES,
    USERS as BOTSIM_USERS,
    raw_account as botsim_raw_account,
)
from verifiable_bot.botsim_source_audit import audit_all as botsim_native_audit


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v2.json"
LOCK = ROOT / "output/evidence_schema_v2_group_additive_candidates/selection_lock.json"
FOX8_PRIOR = ROOT / "research/certificates/fox8_twenty_independent_raw_truth.json"
BOTSIM_PRIOR = ROOT / "research/certificates/botsim_twelve_independent_raw_truth.json"
OUT = ROOT / "research/certificates/schema_v2_raw_fact_binding.json"


def independently_recomputed_botsim() -> tuple[list[str], np.ndarray, dict]:
    native_audit = botsim_native_audit(
        BOTSIM_USERS, BOTSIM_ACTIVITY, BOTSIM_NATIVE_DIR)
    if (native_audit["status"] != "verified_full_population_against_raw_comments"
            or native_audit["verified_fact_cells"] != 2907 * 5):
        raise ValueError("BotSim native graph audit differs")
    with BOTSIM_USERS.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["user_id"] for row in rows]
    activity = json.loads(BOTSIM_ACTIVITY.read_text(encoding="utf-8"))
    if len(ids) != 2907 or len(set(ids)) != len(ids) or set(activity) != set(ids):
        raise ValueError("BotSim raw account population differs")
    base = np.asarray(
        [botsim_raw_account(row, activity[row["user_id"]]) for row in rows],
        dtype=np.float32)
    with np.load(BOTSIM_NATIVE_DIR / "native_graph_facts.npz",
                 allow_pickle=False) as saved:
        if (saved["ids"].astype(str).tolist() != ids
                or saved["names"].astype(str).tolist() != list(BOTSIM_NATIVE_NAMES)):
            raise ValueError("BotSim native graph cache differs")
        native = saved["values"].astype(float)
    if base.shape != (len(ids), len(BOTSIM_BASE_NAMES)):
        raise ValueError("BotSim independent base-fact shape differs")
    return ids, np.column_stack((base, native)), native_audit


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    fox_prior = json.loads(FOX8_PRIOR.read_text(encoding="utf-8"))
    bot_prior = json.loads(BOTSIM_PRIOR.read_text(encoding="utf-8"))
    if lock["schema_sha256"] != sha256_file(SCHEMA):
        raise ValueError("Frozen candidate schema differs")
    if (fox_prior["raw_source_sha256"] != sha256_file(FOX8_RAW)
            or fox_prior["native_event_index_sha256"] != sha256_file(FOX8_EVENTS)
            or bot_prior["users_sha256"] != sha256_file(BOTSIM_USERS)
            or bot_prior["activity_sha256"] != sha256_file(BOTSIM_ACTIVITY)):
        raise ValueError("Raw sources differ from prior independent audits")

    fox_ids, fox_truth, fox_pairs = fox8_raw_truth()
    bot_ids, bot_truth, bot_native = independently_recomputed_botsim()
    populations = {
        "fox8": (fox_ids, fox_truth, FOX8_RAW_NAMES),
        "botsim": (bot_ids, bot_truth, BOTSIM_RAW_NAMES),
    }
    results = {}
    for dataset, (raw_ids, truth, raw_names) in populations.items():
        selected = [row["fact"] for row in schema["schema"][dataset]["concepts"]]
        if len(selected) != 16 or len(set(selected)) != 16:
            raise ValueError(f"{dataset} schema is not sixteen unique facts")
        if set(selected) - set(raw_names):
            raise ValueError(f"{dataset} schema has a fact outside raw audit")
        expected = truth[:, [raw_names.index(name) for name in selected]]
        runs = []
        for seed, frozen in enumerate(lock["runs"][dataset]):
            values, _, ids, names, _, manifest = _source_facts(dataset, seed)
            if (ids != raw_ids or frozen["selected_fact_names"] != selected
                    or manifest["files_sha256"] != frozen["source_files_sha256"]):
                raise ValueError(f"{dataset} seed {seed} account or lock differs")
            observed = values[:, [names.index(name) for name in selected]]
            if not np.array_equal(np.isfinite(observed), np.isfinite(expected)):
                raise ValueError(f"{dataset} seed {seed} availability differs")
            difference = float(np.nanmax(np.abs(observed - expected)))
            if difference > 1e-6:
                raise ValueError(f"{dataset} seed {seed} raw fact value differs")
            modeled = _missing_means(dataset, values, names, ids)
            modeled = modeled[:, [names.index(name) for name in selected]]
            intentional_missing = int(np.count_nonzero(
                np.isfinite(expected) & ~np.isfinite(modeled)))
            if np.any(~np.isfinite(expected) & np.isfinite(modeled)):
                raise ValueError(f"{dataset} raw missing value became observed")
            runs.append({
                "seed": seed,
                "accounts": len(ids),
                "selected_fact_cells": int(expected.size),
                "max_value_difference_before_missing_policy": difference,
                "intentionally_marked_missing_mean_cells": intentional_missing,
                "source_files_sha256": manifest["files_sha256"],
            })
        results[dataset] = {
            "selected_fact_names": selected,
            "source_population_accounts": len(raw_ids),
            "raw_nan_cells": int(np.count_nonzero(~np.isfinite(expected))),
            "runs": runs,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "schema_v2_factual_inputs_match_independent_raw_recomputation",
        "schema_sha256": sha256_file(SCHEMA),
        "lock_sha256": sha256_file(LOCK),
        "fox8_prior_audit_sha256": sha256_file(FOX8_PRIOR),
        "botsim_prior_audit_sha256": sha256_file(BOTSIM_PRIOR),
        "fox8_raw_sha256": sha256_file(FOX8_RAW),
        "fox8_native_event_pairs_recomputed": fox_pairs,
        "botsim_users_sha256": sha256_file(BOTSIM_USERS),
        "botsim_activity_sha256": sha256_file(BOTSIM_ACTIVITY),
        "botsim_native_fact_cells_recomputed": bot_native["verified_fact_cells"],
        "behavioral_semantic_truth_checked": False,
        "candidate_model_decision_effect_checked_here": False,
        "results": results,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(OUT)
    for dataset, row in results.items():
        print(dataset, len(row["selected_fact_names"]),
              row["source_population_accounts"],
              [run["max_value_difference_before_missing_policy"]
               for run in row["runs"]],
              [run["intentionally_marked_missing_mean_cells"]
               for run in row["runs"]])


if __name__ == "__main__":
    main()
