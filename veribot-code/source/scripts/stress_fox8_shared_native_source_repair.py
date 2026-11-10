"""Test whether repairing one high-fanout raw profile can fix several decisions."""

from __future__ import annotations

import hashlib
import gzip
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from stress_fox8_targeted_twenty_source_repair import raw_statuses, score
from verifiable_bot.fox8_profile_integrity import (
    detect_statuses_discrepancies, read_first_raw_statuses)
from verifiable_bot.fox8_targeted_twenty_runtime import Fox8TargetedTwentyRuntime


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "research/certificates/fox8_twenty_independent_raw_truth.json"
OUT = ROOT / "output/v3_fox8_targeted_twenty_ab_group/shared_source_repair.json"
K = 20
BUDGETS = (0, 1, 2, 5, 10, 20)


def digest_key(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def ambiguous_accounts(path: Path, ids: list[str], first: np.ndarray) -> set[str]:
    index = {uid: i for i, uid in enumerate(ids)}
    ambiguous: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            uid = str(record["user_id"])
            if uid not in index:
                continue
            tweets = record.get("user_tweets") or []
            profile = (tweets[0].get("user") or {}) if tweets else {}
            if float(profile.get("statuses_count", 0) or 0) != first[index[uid]]:
                ambiguous.add(uid)
    return ambiguous


def donors(ids: list[str], eligible_ids: list[str], source_ids: list[str],
           status: np.ndarray, strategy: str) -> dict[str, str]:
    index = {uid: j for j, uid in enumerate(ids)}
    if strategy == "shared_hash_donor":
        ranked = sorted(eligible_ids, key=lambda uid: digest_key(
            f"Fox8-shared-source-random-donor:{uid}"))
        return {uid: next(candidate for candidate in ranked
                          if candidate != uid and status[index[candidate]] !=
                          status[index[uid]]) for uid in source_ids}
    if strategy == "per_source_hash_donor":
        return {uid: next(candidate for candidate in sorted(
                    eligible_ids, key=lambda candidate: digest_key(
                        f"Fox8-per-source-donor:{uid}:{candidate}"))
                    if candidate != uid and status[index[candidate]] !=
                    status[index[uid]]) for uid in source_ids}
    if strategy == "opposite_decile":
        ordered = sorted(eligible_ids, key=lambda uid: (status[index[uid]], uid))
        low = ordered[round(.10*(len(ordered)-1))]
        high = ordered[round(.90*(len(ordered)-1))]
        if status[index[low]] == status[index[high]]:
            raise ValueError("Profile count deciles are not distinct")
        result = {}
        for uid in source_ids:
            distances = [(abs(np.log1p(status[index[uid]]) -
                              np.log1p(status[index[d]])), d)
                         for d in (low, high) if d != uid]
            result[uid] = max(distances)[1]
        return result
    raise ValueError("unknown label-blind donor strategy")


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["status"] != "all_model_used_fox8_twenty_facts_independently_recomputed_from_raw":
        raise ValueError("Independent raw truth audit differs")
    runs = []
    for seed in (0, 1, 2):
        runtime = Fox8TargetedTwentyRuntime(ROOT, seed)
        if (sha256_file(runtime.source.raw_path) != audit["raw_source_sha256"]
                or runtime.record["model_sha256"] !=
                audit["runs"][seed]["model_sha256"]):
            raise ValueError("Raw source or frozen model differs")
        truth = raw_statuses(runtime.source.raw_path, runtime.ids)
        ambiguous = ambiguous_accounts(runtime.source.raw_path,
                                       runtime.ids, truth)
        snapshot = read_first_raw_statuses(runtime.source.raw_path,
                                           runtime.ids)
        if (not np.array_equal(snapshot.values, truth) or
                set(snapshot.ambiguous_account_ids) != ambiguous):
            raise ValueError("Raw profile verifier differs from source parser")
        eligible = [uid for uid in runtime.ids if uid not in ambiguous]
        status_col = runtime.reference.names.index("statuses_count")
        if not np.array_equal(truth.astype(np.float32),
                              runtime.source.facts[:, status_col].astype(np.float32)):
            raise ValueError("Raw profile truth differs from model source")
        saved_path = runtime.run_dir / "test_predictions.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))["predictions"]
        test_ids = [row["account_id"] for row in saved]
        labels = np.asarray([row["label"] for row in saved], dtype=int)
        test_rows = np.asarray([runtime.index[uid] for uid in test_ids], dtype=int)
        clean = score(runtime, truth, test_rows)
        if np.max(np.abs(clean - np.asarray([row["probability"]
                                            for row in saved]))) > 1e-12:
            raise ValueError("Clean scoring differs from selected model")
        ranked_sources = sorted(eligible,
                                key=lambda uid: (-len(runtime.source.neighbors[
                                    runtime.index[uid]]), uid))
        source_ids = ranked_sources[:K]
        if len(set(source_ids)) != K or not runtime.source.neighbors[
                runtime.index[source_ids[-1]]]:
            raise ValueError("Native source fanout selection differs")
        source_rows = [{"source_account_id": uid,
                        "native_neighbor_count": len(runtime.source.neighbors[
                            runtime.index[uid]]),
                        "test_accounts_receiving_source_or_self": sum(
                            uid == target or uid in runtime.source.neighbors[
                                runtime.index[target]] for target in test_ids),
                        "raw_statuses_count": float(truth[runtime.index[uid]])}
                       for uid in source_ids]
        scenarios = []
        test_index = {uid: j for j, uid in enumerate(test_ids)}
        clean_error = (clean >= runtime.threshold) != labels
        for strategy in ("shared_hash_donor", "per_source_hash_donor",
                         "opposite_decile"):
            donor_for = donors(runtime.ids, eligible, source_ids, truth, strategy)
            corrupted = truth.copy()
            for uid in source_ids:
                corrupted[runtime.index[uid]] = truth[
                    runtime.index[donor_for[uid]]]
            if np.count_nonzero(corrupted != truth) != K:
                raise ValueError("Known corruptions must alter twenty records")
            detected = detect_statuses_discrepancies(runtime.ids,
                                                     corrupted, snapshot)
            prioritized = sorted(detected, key=lambda uid: (
                -len(runtime.source.neighbors[runtime.index[uid]]), uid))
            if prioritized != source_ids:
                raise ValueError("Raw discrepancy priority differs from source order")
            bad = score(runtime, corrupted, test_rows)
            bad_error = (bad >= runtime.threshold) != labels
            by_uncertainty = sorted(test_ids, key=lambda uid: (
                abs(float(bad[test_index[uid]]) - runtime.threshold), uid))
            by_hash = sorted(test_ids, key=lambda uid: digest_key(
                f"Fox8-shared-source-direct-label:{uid}"))
            curves = []
            for budget in BUDGETS:
                repaired = corrupted.copy()
                for uid in prioritized[:budget]:
                    repaired[runtime.index[uid]] = snapshot.values[
                        runtime.index[uid]]
                repaired_probability = score(runtime, repaired, test_rows)
                repaired_errors = int(np.count_nonzero(
                    (repaired_probability >= runtime.threshold) != labels))
                direct = {}
                for name, ordered in (("model_uncertainty", by_uncertainty),
                                      ("hash_random", by_hash)):
                    remaining = bad_error.copy()
                    remaining[[test_index[uid] for uid in ordered[:budget]]] = False
                    direct[name] = int(remaining.sum())
                curves.append({"action_budget": budget,
                               "source_repair_remaining_errors": repaired_errors,
                               "direct_label_remaining_errors": direct,
                               "oracle_direct_label_remaining_errors": max(
                                   0, int(bad_error.sum()) - budget),
                               "source_repair_changed_classes": int(np.count_nonzero(
                                   (repaired_probability >= runtime.threshold) !=
                                   (bad >= runtime.threshold))),
                               "max_repair_probability_difference_from_clean":
                               float(np.max(np.abs(repaired_probability - clean)))})
            if (curves[-1]["source_repair_remaining_errors"] !=
                    int(clean_error.sum()) or
                    curves[-1]["max_repair_probability_difference_from_clean"] >
                    1e-12):
                raise ValueError("Source repair did not restore clean decision")
            row = {
                "wrong_join_strategy": strategy,
                "clean_errors": int(clean_error.sum()),
                "corrupted_errors": int(bad_error.sum()),
                "corruption_changed_classes": int(np.count_nonzero(
                    (bad >= runtime.threshold) != (clean >= runtime.threshold))),
                "corruption_new_errors": int(np.count_nonzero(
                    bad_error & ~clean_error)),
                "corruption_removed_errors": int(np.count_nonzero(
                    clean_error & ~bad_error)),
                "wrong_join_donors": donor_for,
                "raw_discrepancy_detected_sources_in_graph_priority_order":
                    prioritized,
                "budget_curves": curves,
            }
            scenarios.append(row)
            print({"seed": seed, "strategy": strategy,
                   "clean_errors": row["clean_errors"],
                   "corrupted_errors": row["corrupted_errors"],
                   "budget_1": curves[1], "budget_5": curves[3]}, flush=True)
        runs.append({"seed": seed, "test_accounts": len(test_ids),
                     "excluded_ambiguous_raw_profile_accounts": sorted(ambiguous),
                     "selected_sources": source_rows,
                     "test_predictions_sha256": sha256_file(saved_path),
                     "scenarios": scenarios})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "fox8_native_high_fanout_profile_join_error_and_shared_repair",
        "independent_raw_truth_sha256": sha256_file(AUDIT),
        "source_selection_uses_only_native_graph_degree_account_id_and_unambiguous_raw_profile": True,
        "source_and_donor_selection_use_bot_human_labels": False,
        "same_budget_unit": "one upstream source-record correction or one directly labeled test account",
        "natural_error_frequency_or_human_minutes_measured": False,
        "prior_test_population_exploration": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
