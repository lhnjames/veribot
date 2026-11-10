"""Independently replay clean, corrupted, and repaired Fox8 shared-source scores."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.fox8_targeted_twenty_runtime import Fox8TargetedTwentyRuntime
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "output/v3_fox8_targeted_twenty_ab_group/shared_source_repair.json"
OUT = ROOT / "research/certificates/fox8_shared_source_repair_audit.json"


def raw_counts(path: Path, ids: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    index = {uid: i for i, uid in enumerate(ids)}
    values = np.full(len(ids), np.nan)
    later_different: dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            uid = str(record["user_id"])
            if uid not in index:
                continue
            tweets = record.get("user_tweets") or []
            profile = (tweets[0].get("user") or {}) if tweets else {}
            value = float(profile.get("statuses_count", 0) or 0)
            position = index[uid]
            if np.isfinite(values[position]):
                if values[position] != value:
                    later_different[uid] = later_different.get(uid, 0) + 1
                continue
            values[position] = value
    if not np.isfinite(values).all():
        raise ValueError("Raw Fox8 profile population incomplete")
    return values, later_different


def independent_score(runtime: Fox8TargetedTwentyRuntime,
                      counts: np.ndarray, rows: np.ndarray) -> np.ndarray:
    raw = runtime.source.facts[rows].copy()
    own_col = runtime.reference.names.index("statuses_count")
    neighbor_col = runtime.reference.names.index(
        "native_interaction_neighbor_mean_statuses_count")
    raw[:, own_col] = counts[rows]
    for j, i in enumerate(rows):
        neighbors = runtime.source.neighbors[int(i)]
        if neighbors:
            positions = [runtime.index[uid] for uid in neighbors]
            raw[j, neighbor_col] = np.float32(np.mean(counts[positions]))
        else:
            raw[j, neighbor_col] = np.nan
    atomic = runtime.reference.transform(raw)
    reliability = atomic.reliability.copy()
    reliability[:, 43:74] = reliability[:, 43:74] * (
        runtime.source.supports[rows] / (runtime.source.supports[rows] + 5.0))
    kept = runtime.kept
    reduced = EvidenceBatch(atomic.upper[:, kept], atomic.lower[:, kept],
                            reliability[:, kept], atomic.available[:, kept],
                            atomic.reference_version,
                            tuple(atomic.names[j] for j in kept))
    concepts, _ = collapse_group_tails(reduced, runtime.groups)
    matrix = np.concatenate((np.log1p(concepts.upper),
                             np.log1p(concepts.lower)), axis=1)
    return expit(np.asarray(runtime.model.booster_.predict(
        matrix, raw_score=True), dtype=float))


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if (result["scope"] !=
            "fox8_native_high_fanout_profile_join_error_and_shared_repair"
            or len(result["runs"]) != 3):
        raise ValueError("Shared-source result differs")
    audits = []
    for saved_run in result["runs"]:
        seed = saved_run["seed"]
        runtime = Fox8TargetedTwentyRuntime(ROOT, seed)
        if sha256_file(runtime.run_dir / "test_predictions.json") != (
                saved_run["test_predictions_sha256"]):
            raise ValueError("Frozen test predictions differ")
        truth, later_different = raw_counts(runtime.source.raw_path, runtime.ids)
        if not np.array_equal(truth.astype(np.float32),
                              runtime.source.facts[:, runtime.reference.names.index(
                                  "statuses_count")].astype(np.float32)):
            raise ValueError("Raw count differs from model source")
        saved_predictions = json.loads((runtime.run_dir / "test_predictions.json")
                                       .read_text(encoding="utf-8"))["predictions"]
        test_rows = np.asarray([runtime.index[r["account_id"]]
                                for r in saved_predictions], dtype=int)
        labels = np.asarray([r["label"] for r in saved_predictions], dtype=int)
        clean = independent_score(runtime, truth, test_rows)
        expected = np.asarray([r["probability"] for r in saved_predictions])
        max_clean_diff = float(np.max(np.abs(clean - expected)))
        if max_clean_diff > 1e-12:
            raise ValueError("Independent clean probabilities differ")
        selected = [r["source_account_id"] for r in saved_run["selected_sources"]]
        ranked = sorted([uid for uid in runtime.ids
                         if uid not in later_different], key=lambda uid: (
            -len(runtime.source.neighbors[runtime.index[uid]]), uid))[:20]
        if selected != ranked:
            raise ValueError("Source selection is not label-blind native degree")
        if (saved_run["excluded_ambiguous_raw_profile_accounts"] !=
                sorted(later_different)):
            raise ValueError("Ambiguous source exclusion differs")
        scenarios = []
        for scenario in saved_run["scenarios"]:
            corrupted = truth.copy()
            for uid, donor in scenario["wrong_join_donors"].items():
                corrupted[runtime.index[uid]] = truth[runtime.index[donor]]
            if set(scenario["wrong_join_donors"]) != set(selected):
                raise ValueError("Wrong join source list differs")
            if any(uid in later_different for uid in
                   scenario["wrong_join_donors"].values()):
                raise ValueError("Ambiguous raw profile served as wrong-join donor")
            checked = []
            for budget in (0, 1, 20):
                repaired = corrupted.copy()
                for uid in selected[:budget]:
                    repaired[runtime.index[uid]] = truth[runtime.index[uid]]
                probability = independent_score(runtime, repaired, test_rows)
                errors = int(np.count_nonzero(
                    (probability >= runtime.threshold) != labels))
                recorded = next(row for row in scenario["budget_curves"]
                                if row["action_budget"] == budget)
                max_diff = float(np.max(np.abs(probability - clean)))
                if (errors != recorded["source_repair_remaining_errors"]
                        or abs(max_diff - recorded[
                            "max_repair_probability_difference_from_clean"]) > 1e-12
                        or recorded["oracle_direct_label_remaining_errors"] !=
                        max(0, scenario["corrupted_errors"] - budget)):
                    raise ValueError("Independent repair or oracle budget differs")
                checked.append({"budget": budget, "errors": errors,
                                "max_probability_diff_from_clean": max_diff})
            scenarios.append({"strategy": scenario["wrong_join_strategy"],
                              "verified_budgets": checked,
                              "source_repair_budget_one_beats_oracle_direct_label":
                              checked[1]["errors"] < scenario[
                                  "budget_curves"][1][
                                  "oracle_direct_label_remaining_errors"]})
        audits.append({"seed": seed, "max_clean_saved_probability_difference":
                       max_clean_diff, "selected_source_count": len(selected),
                       "raw_accounts_with_later_different_statuses_count":
                       len(later_different),
                       "selected_sources_with_later_different_statuses_count":
                       [uid for uid in selected if uid in later_different],
                       "top_source_native_degree": len(runtime.source.neighbors[
                           runtime.index[selected[0]]]),
                       "scenarios": scenarios})
        print({"seed": seed, "scenarios": scenarios}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "independent_raw_to_model_shared_source_repair_replay_verified",
        "result_sha256": sha256_file(RESULT),
        "all_budget_one_repair_beats_oracle_direct_label": all(
            s["source_repair_budget_one_beats_oracle_direct_label"]
            for r in audits for s in r["scenarios"]),
        "budget_one_beats_oracle_by_strategy": {
            strategy: all(s["source_repair_budget_one_beats_oracle_direct_label"]
                          for r in audits for s in r["scenarios"]
                          if s["strategy"] == strategy)
            for strategy in ("shared_hash_donor", "per_source_hash_donor",
                             "opposite_decile")},
        "natural_error_frequency_or_human_cost_measured": False,
        "runs": audits,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
