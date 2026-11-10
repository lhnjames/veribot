"""Trace a linked native-neighbor source join error through the compact classifier."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.fox8_twenty_concept_runtime import Fox8TwentyConceptRuntime
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


def ordered(ids: list[str], salt: str) -> list[str]:
    return sorted(ids, key=lambda uid: hashlib.sha256(f"{salt}:{uid}".encode()).digest())


def raw_statuses(path: Path, ids: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    index = {uid: i for i, uid in enumerate(ids)}
    values = np.full(len(ids), np.nan)
    lines = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            account = json.loads(line)
            uid = str(account["user_id"])
            if uid in index and uid not in lines:
                values[index[uid]] = float(account["user_tweets"][0]["user"]
                                            .get("statuses_count", 0) or 0)
                lines[uid] = number
    if len(lines) != len(ids) or not np.isfinite(values).all():
        raise ValueError("Raw source profiles do not cover Fox8 accounts")
    return values, lines


def score(runtime: Fox8TwentyConceptRuntime, statuses: np.ndarray,
          test_indices: np.ndarray) -> np.ndarray:
    values = runtime.facts[test_indices].copy()
    status_col = runtime.reference.names.index("statuses_count")
    neighbor_col = runtime.reference.names.index(
        "native_interaction_neighbor_mean_statuses_count")
    values[:, status_col] = statuses[test_indices]
    means = np.full(len(test_indices), np.nan, dtype=np.float32)
    for j, original in enumerate(test_indices):
        adjacent = runtime.neighbors[int(original)]
        if adjacent:
            means[j] = np.float32(np.mean(statuses[
                [runtime.index[uid] for uid in adjacent]]))
    values[:, neighbor_col] = means
    batch = runtime.reference.transform(values)
    reliability = batch.reliability.copy()
    reliability[:, 43:] *= runtime.supports[test_indices] / (
        runtime.supports[test_indices] + 5.0)
    atomic = EvidenceBatch(batch.upper[:, runtime.kept],
                           batch.lower[:, runtime.kept],
                           reliability[:, runtime.kept],
                           batch.available[:, runtime.kept],
                           batch.reference_version,
                           tuple(batch.names[j] for j in runtime.kept))
    concepts, _ = collapse_group_tails(atomic, runtime.groups)
    x = np.column_stack((np.log1p(concepts.upper) * concepts.reliability,
                         np.log1p(concepts.lower) * concepts.reliability))
    return runtime.model.predict_proba(x)[:, 1]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_fox8_targeted_neighbor_20_concepts"
    source_audit_path = root / "research/certificates/fox8_neighbor_volume_full_raw_source.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if source_audit["status"] != "all_target_accounts_and_neighbor_raw_profiles_verified":
        raise ValueError("Full raw source truth is absent")
    runs = []
    for seed in (0, 1, 2):
        runtime = Fox8TwentyConceptRuntime(root, seed)
        truth, raw_lines = raw_statuses(runtime.raw_path, runtime.ids)
        if (source_audit["raw_source_sha256"] != sha256_file(runtime.raw_path)
                or not np.array_equal(truth.astype(np.float32),
                    runtime.facts[:, runtime.reference.names.index("statuses_count")]
                    .astype(np.float32))):
            raise ValueError("Raw profile truth differs from the frozen source")
        prediction_path = folder / f"seed{seed}/ab_group_tree_test_predictions.json"
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        test_ids = [row["account_id"] for row in predictions["predictions"]]
        test_indices = np.asarray([runtime.index[uid] for uid in test_ids], dtype=int)
        labels = np.asarray([row["label"] for row in predictions["predictions"]])
        clean_p = score(runtime, truth, test_indices)
        saved_p = np.asarray([row["probability"] for row in predictions["predictions"]])
        if np.max(np.abs(clean_p - saved_p)) > 1e-12:
            raise ValueError("Raw-source clean prediction differs from fixed test")
        candidates = [uid for uid in runtime.ids if runtime.neighbors[runtime.index[uid]]]
        recipients = ordered(candidates, "Fox8-neighbor-profile-source-join-recipient-v1")
        selected = recipients[:round(0.10 * len(recipients))]
        donors = ordered(runtime.ids, "Fox8-neighbor-profile-source-join-donor-v1")
        corrupted = truth.copy()
        source_rows = []
        for rank, uid in enumerate(selected):
            donor = next(item for item in donors[rank:] + donors[:rank]
                         if item != uid and truth[runtime.index[item]] !=
                         truth[runtime.index[uid]])
            corrupted[runtime.index[uid]] = truth[runtime.index[donor]]
            source_rows.append({
                "source_account_id": uid,
                "raw_gzip_line": raw_lines[uid],
                "source_statuses_count": float(truth[runtime.index[uid]]),
                "wrong_join_donor_id": donor,
                "wrong_join_statuses_count": float(truth[runtime.index[donor]]),
                "affected_test_targets_by_graph_or_self": sum(
                    uid == target or uid in runtime.neighbors[runtime.index[target]]
                    for target in test_ids)})
        if np.count_nonzero(corrupted != truth) != len(selected):
            raise ValueError("Known source join errors are not distinct")
        bad_p = score(runtime, corrupted, test_indices)
        threshold = float(predictions["manifest"]["threshold"])
        clean_error = (clean_p >= threshold) != labels
        bad_error = (bad_p >= threshold) != labels
        account_order = ordered(test_ids, "Fox8-direct-account-label-order-v1")
        test_index = {uid: j for j, uid in enumerate(test_ids)}
        budgets = sorted(set([0, 10, 25, 50, 100, len(selected)]))
        points = []
        for budget in budgets:
            repaired = corrupted.copy()
            for uid in selected[:budget]:
                repaired[runtime.index[uid]] = truth[runtime.index[uid]]
            repaired_p = score(runtime, repaired, test_indices)
            source_errors = (repaired_p >= threshold) != labels
            directly_labeled = np.asarray(
                [test_index[uid] for uid in account_order[:budget]], dtype=int)
            direct_errors = bad_error.copy()
            direct_errors[directly_labeled] = False
            points.append({
                "source_record_actions": budget,
                "direct_account_label_actions": budget,
                "source_corrected_errors": int(source_errors.sum()),
                "direct_labeled_errors": int(direct_errors.sum()),
                "source_corrected_changed_classes": int(np.count_nonzero(
                    (repaired_p >= threshold) != (bad_p >= threshold))),
            })
        if points[-1]["source_corrected_errors"] != int(clean_error.sum()):
            raise ValueError("Full source repair does not restore clean decisions")
        fully_repaired = corrupted.copy()
        for uid in selected:
            fully_repaired[runtime.index[uid]] = truth[runtime.index[uid]]
        if not np.array_equal(fully_repaired, truth) or np.max(np.abs(
                score(runtime, fully_repaired, test_indices) - clean_p)) > 1e-12:
            raise ValueError("Full source correction does not restore probabilities")
        row = {
            "seed": seed,
            "test_accounts": len(test_ids),
            "wrong_source_record_count": len(selected),
            "clean_errors": int(clean_error.sum()),
            "corrupted_errors": int(bad_error.sum()),
            "fully_source_corrected_errors": points[-1]["source_corrected_errors"],
            "corruption_changed_classes": int(np.count_nonzero(
                (bad_p >= threshold) != (clean_p >= threshold))),
            "corruption_new_errors": int(np.count_nonzero(bad_error & ~clean_error)),
            "corruption_removed_errors": int(np.count_nonzero(clean_error & ~bad_error)),
            "max_repair_probability_difference": 0.0,
            "budget_curve": points,
            "source_rows": source_rows,
            "fixed_test_prediction_sha256": sha256_file(prediction_path),
        }
        runs.append(row)
        print(json.dumps({key: row[key] for key in (
            "seed", "wrong_source_record_count", "clean_errors",
            "corrupted_errors", "fully_source_corrected_errors",
            "corruption_changed_classes", "corruption_new_errors",
            "corruption_removed_errors")}), flush=True)
    output = {
        "scope": "Fox8_compact_20_linked_neighbor_source_join_injection_and_repair",
        "error_selection_and_correction_use_account_labels": False,
        "raw_source_truth_sha256": source_audit["raw_source_sha256"],
        "source_audit_sha256": sha256_file(source_audit_path),
        "source_change_propagates_to_own_statuses_and_all_native_neighbors": True,
        "direct_label_comparison_unit": "one_source_record_or_one_account_action_not_measured_human_minutes",
        "natural_error_frequency_measured": False,
        "runs": runs,
    }
    path = root / "output/v3_fox8_targeted_neighbor_20_concepts/linked_source_correction.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
