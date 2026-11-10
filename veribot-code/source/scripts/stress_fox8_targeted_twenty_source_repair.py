"""Known raw-profile join error and source repair for the selected Fox8 20-concept model."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.fox8_targeted_twenty_runtime import Fox8TargetedTwentyRuntime
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_fox8_targeted_twenty_ab_group/known_source_repair.json"
AUDIT = ROOT / "research/certificates/fox8_twenty_independent_raw_truth.json"


def order(ids: list[str], salt: str) -> list[str]:
    return sorted(ids, key=lambda uid: hashlib.sha256(
        f"{salt}:{uid}".encode()).digest())


def raw_statuses(path: Path, ids: list[str]) -> np.ndarray:
    index = {uid: i for i, uid in enumerate(ids)}
    values = np.full(len(ids), np.nan)
    seen = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid in index and uid not in seen:
                tweets = row.get("user_tweets") or []
                user = tweets[0].get("user") or {} if tweets else {}
                values[index[uid]] = float(user.get("statuses_count", 0) or 0)
                seen.add(uid)
    if seen != set(ids) or not np.isfinite(values).all():
        raise ValueError("Raw Fox8 statuses_count population incomplete")
    return values


def score(runtime: Fox8TargetedTwentyRuntime, status: np.ndarray,
          rows: np.ndarray) -> np.ndarray:
    facts = runtime.source.facts[rows].copy()
    status_col = runtime.reference.names.index("statuses_count")
    neighbor_col = runtime.reference.names.index(
        "native_interaction_neighbor_mean_statuses_count")
    facts[:, status_col] = status[rows]
    for j, row in enumerate(rows):
        peers = runtime.source.neighbors[int(row)]
        facts[j, neighbor_col] = (np.float32(np.mean(
            status[[runtime.index[uid] for uid in peers]])) if peers else np.nan)
    atomic = runtime.reference.transform(facts)
    reliability = atomic.reliability.copy()
    reliability[:, 43:74] *= runtime.source.supports[rows] / (
        runtime.source.supports[rows] + 5.0)
    selected = EvidenceBatch(
        atomic.upper[:, runtime.kept], atomic.lower[:, runtime.kept],
        reliability[:, runtime.kept], atomic.available[:, runtime.kept],
        atomic.reference_version,
        tuple(atomic.names[j] for j in runtime.kept))
    concepts, _ = collapse_group_tails(selected, runtime.groups)
    matrix = np.column_stack((np.log1p(concepts.upper),
                              np.log1p(concepts.lower)))
    return runtime.model.predict_proba(matrix)[:, 1]


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["status"] != "all_model_used_fox8_twenty_facts_independently_recomputed_from_raw":
        raise ValueError("Independent raw truth certificate missing")
    runs = []
    for seed in (0, 1, 2):
        runtime = Fox8TargetedTwentyRuntime(ROOT, seed)
        raw_path = runtime.source.raw_path
        if (sha256_file(raw_path) != audit["raw_source_sha256"]
                or audit["runs"][seed]["model_sha256"] !=
                runtime.record["model_sha256"]):
            raise ValueError("Raw truth or selected model differs")
        truth = raw_statuses(raw_path, runtime.ids)
        status_col = runtime.reference.names.index("statuses_count")
        if not np.array_equal(truth.astype(np.float32),
                              runtime.source.facts[:, status_col].astype(np.float32)):
            raise ValueError("Raw profile status count differs from model source")
        path = runtime.run_dir / "test_predictions.json"
        saved = json.loads(path.read_text(encoding="utf-8"))["predictions"]
        test_ids = [row["account_id"] for row in saved]
        labels = np.asarray([row["label"] for row in saved], dtype=int)
        test_rows = np.asarray([runtime.index[uid] for uid in test_ids], dtype=int)
        clean = score(runtime, truth, test_rows)
        saved_p = np.asarray([row["probability"] for row in saved])
        max_clean_difference = float(np.max(np.abs(clean - saved_p)))
        if max_clean_difference > 1e-12:
            raise ValueError("Clean raw-source scoring differs from selected model")
        candidates = [uid for uid in runtime.ids
                      if runtime.source.neighbors[runtime.index[uid]]]
        recipients = order(candidates, "Fox8-targeted-twenty-join-recipients-v1")
        selected = recipients[:round(.10 * len(recipients))]
        donors = order(runtime.ids, "Fox8-targeted-twenty-join-donors-v1")
        corrupted = truth.copy()
        rows = []
        for rank, uid in enumerate(selected):
            donor = next(item for item in donors[rank:] + donors[:rank]
                         if item != uid and truth[runtime.index[item]] !=
                         truth[runtime.index[uid]])
            corrupted[runtime.index[uid]] = truth[runtime.index[donor]]
            rows.append({"source_account_id": uid,
                         "wrong_join_donor_id": donor,
                         "source_value": float(truth[runtime.index[uid]]),
                         "wrong_join_value": float(truth[runtime.index[donor]]),
                         "test_targets_affected_by_self_or_native_neighbor": sum(
                             uid == target or uid in runtime.source.neighbors[
                                 runtime.index[target]] for target in test_ids)})
        if np.count_nonzero(corrupted != truth) != len(selected):
            raise ValueError("Known join errors are not distinct")
        bad = score(runtime, corrupted, test_rows)
        threshold = runtime.threshold
        clean_error = (clean >= threshold) != labels
        bad_error = (bad >= threshold) != labels
        direct_hash = order(test_ids, "Fox8-targeted-twenty-direct-label-v1")
        direct_uncertain = sorted(test_ids, key=lambda uid: (
            abs(float(bad[test_ids.index(uid)]) - threshold), uid))
        test_index = {uid: i for i, uid in enumerate(test_ids)}
        budgets = sorted(set((0, 10, 25, 50, 100, len(selected))))
        curves = {"hash_random": [], "model_uncertainty": []}
        for budget in budgets:
            repaired = corrupted.copy()
            for uid in selected[:budget]:
                repaired[runtime.index[uid]] = truth[runtime.index[uid]]
            repaired_p = score(runtime, repaired, test_rows)
            source_errors = int(np.count_nonzero((repaired_p >= threshold) != labels))
            for method, ordering in (("hash_random", direct_hash),
                                     ("model_uncertainty", direct_uncertain)):
                direct = bad_error.copy()
                direct[[test_index[uid] for uid in ordering[:budget]]] = False
                curves[method].append({
                    "account_actions": budget,
                    "source_corrected_errors": source_errors,
                    "direct_labeled_errors": int(direct.sum()),
                    "source_corrected_changed_classes": int(np.count_nonzero(
                        (repaired_p >= threshold) != (bad >= threshold))),
                })
        repaired = corrupted.copy()
        for uid in selected:
            repaired[runtime.index[uid]] = truth[runtime.index[uid]]
        max_repair_difference = float(np.max(np.abs(
            score(runtime, repaired, test_rows) - clean)))
        if (curves["hash_random"][-1]["source_corrected_errors"] !=
                int(clean_error.sum()) or max_repair_difference > 1e-12):
            raise ValueError("Full source repair failed to restore clean model")
        result = {
            "seed": seed,
            "test_accounts": len(test_ids),
            "wrong_source_records": len(selected),
            "clean_errors": int(clean_error.sum()),
            "corrupted_errors": int(bad_error.sum()),
            "fully_source_corrected_errors": int(clean_error.sum()),
            "corruption_changed_classes": int(np.count_nonzero(
                (bad >= threshold) != (clean >= threshold))),
            "corruption_new_errors": int(np.count_nonzero(
                bad_error & ~clean_error)),
            "corruption_removed_errors": int(np.count_nonzero(
                clean_error & ~bad_error)),
            "max_clean_saved_probability_difference": max_clean_difference,
            "max_full_repair_probability_difference": max_repair_difference,
            "budget_curves": curves,
            "source_rows": rows,
            "test_predictions_sha256": sha256_file(path),
        }
        runs.append(result)
        print({k: result[k] for k in (
            "seed", "wrong_source_records", "clean_errors",
            "corrupted_errors", "corruption_changed_classes")}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "selected_fox8_twenty_raw_profile_join_error_and_linked_repair",
        "independent_raw_truth_sha256": sha256_file(AUDIT),
        "source_selection_and_repair_use_bot_human_labels": False,
        "same_budget_unit": "one source-record correction or one directly labeled test account",
        "natural_error_frequency_or_human_minutes_measured": False,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
