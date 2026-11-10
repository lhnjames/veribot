"""Independently check Fox8 20-concept winners against raw tweet/profile records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verify_fox8_full_evidence_cards import raw_matrix
from verifiable_bot.fox8_adapter import FEATURE_FIELDS
from verifiable_bot.fox8_targeted_twenty_runtime import Fox8TargetedTwentyRuntime


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
NATIVE = ROOT / "data/processed/fox8_v3/native_interaction_facts.npz"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
FOLDER = ROOT / "output/v3_fox8_targeted_twenty_ab_group"
OUT = ROOT / "research/certificates/fox8_twenty_raw_to_decision.json"


def source_population() -> tuple[list[str], np.ndarray, int]:
    first = Fox8TargetedTwentyRuntime(ROOT, 0)
    args = argparse.Namespace(data_dir=first.source.data_dir,
                              raw=RAW, native_facts=NATIVE)
    ids, raw43, event_pairs = raw_matrix(args)
    if ids != first.ids or raw43.shape != (2279, 43):
        raise ValueError("Raw Fox8 account population or fact count differs")
    index = {uid: i for i, uid in enumerate(ids)}
    adjacency = [set() for _ in ids]
    events = json.loads(EVENTS.read_text(encoding="utf-8"))["events"]
    for event in events:
        u = index[str(event["source_id"])]
        v = index[str(event["target_id"])]
        if u != v:
            adjacency[u].add(v)
            adjacency[v].add(u)
    raw_profile = raw43[:, :len(FEATURE_FIELDS)].astype(float)
    neighbor = np.full((len(ids), len(FEATURE_FIELDS)), np.nan, dtype=np.float32)
    for i, peers in enumerate(adjacency):
        if not peers:
            continue
        values = raw_profile[sorted(peers)]
        counts = np.isfinite(values).sum(axis=0)
        means = np.nansum(values, axis=0) / np.maximum(counts, 1)
        means[counts == 0] = np.nan
        neighbor[i] = means.astype(np.float32)
    return ids, np.column_stack((raw43, neighbor)), event_pairs


def main() -> None:
    ids, raw74, event_pairs = source_population()
    if raw74.shape != (2279, 74):
        raise ValueError("Raw Fox8 74-fact population differs")
    runs = []
    for seed in (0, 1, 2):
        runtime = Fox8TargetedTwentyRuntime(ROOT, seed)
        if runtime.ids != ids or runtime.source.facts.shape != raw74.shape:
            raise ValueError("Model source population differs from raw archive")
        candidate = runtime.source.facts
        if not np.array_equal(np.isfinite(candidate), np.isfinite(raw74)):
            raise ValueError("Raw Fox8 fact availability differs")
        difference = np.abs(candidate - raw74)
        max_fact_difference = float(np.nanmax(difference))
        max_kept_difference = float(np.nanmax(difference[:, runtime.kept]))
        unused_mismatches = [{
            "fact": runtime.reference.names[j],
            "max_difference": float(np.nanmax(difference[:, j])),
            "accounts_differing_over_1e_6": int(np.count_nonzero(
                difference[:, j] > 1e-6)),
        } for j in range(raw74.shape[1])
            if j not in set(runtime.kept)
            and np.isfinite(difference[:, j]).any()
            and float(np.nanmax(difference[:, j])) > 1e-6]
        if max_kept_difference > 1e-4:
            raise ValueError("Model-used fact differs from raw archive")
        saved_path = FOLDER / f"seed{seed}/test_predictions.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))["predictions"]
        account_ids = [row["account_id"] for row in saved]
        actual = runtime.predict_many(account_ids)
        winner_count = 0
        max_winner_difference = 0.0
        max_probability_difference = 0.0
        max_additivity_error = 0.0
        for card, row in zip(actual, saved, strict=True):
            if (card["account_id"] != row["account_id"] or
                    card["prediction"] != row["prediction"] or
                    len(card["concepts"]) != 20):
                raise ValueError("Raw-audited card differs from frozen decision")
            max_probability_difference = max(max_probability_difference,
                abs(card["probability"] - row["probability"]))
            max_additivity_error = max(max_additivity_error,
                                       card["group_additivity_error"])
            i = runtime.index[card["account_id"]]
            for concept in card["concepts"]:
                fact = concept["winning_source_fact"]
                if fact is None:
                    if concept["winning_source_value"] is not None:
                        raise ValueError("Unavailable concept reports a source value")
                    continue
                j = runtime.names.index(fact) if hasattr(runtime, "names") else \
                    runtime.reference.names.index(fact)
                expected = raw74[i, j]
                if not np.isfinite(expected):
                    raise ValueError("Concept winner absent from raw record")
                if concept["source_locator"] is None:
                    raise ValueError("Concept winner lacks source locator")
                delta = abs(float(concept["winning_source_value"]) - float(expected))
                max_winner_difference = max(max_winner_difference, delta)
                winner_count += 1
        if (max_winner_difference > 1e-4 or max_probability_difference > 1e-12
                or max_additivity_error > 1e-9):
            raise ValueError("Raw source-to-concept-to-decision chain differs")
        runs.append({
            "seed": seed,
            "raw_accounts_checked": len(ids),
            "raw_fact_cells_checked": raw74.size,
            "test_accounts_checked": len(actual),
            "test_concept_slots_checked": len(actual) * 20,
            "available_concept_winners_checked": winner_count,
            "max_all_fact_difference": max_fact_difference,
            "max_model_used_fact_difference": max_kept_difference,
            "unused_fact_differences_over_1e_6": unused_mismatches,
            "max_winner_raw_value_difference": max_winner_difference,
            "max_saved_probability_difference": max_probability_difference,
            "max_group_additivity_error": max_additivity_error,
            "model_sha256": runtime.record["model_sha256"],
            "test_predictions_sha256": sha256_file(saved_path),
        })
        print(runs[-1], flush=True)
    OUT.write_text(json.dumps({
        "status": "all_selected_fox8_twenty_concept_facts_recomputed_from_raw_records",
        "raw_source_sha256": sha256_file(RAW),
        "event_index_sha256": sha256_file(EVENTS),
        "native_fact_sha256": sha256_file(NATIVE),
        "typed_internal_event_pairs_checked": event_pairs,
        "source_depth": "raw sampled tweets, embedded profile snapshots, and native reply/retweet targets",
        "claim_type": "literal numeric facts and deterministic evidence concepts",
        "semantic_bot_behavior_validity_inferred": False,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
