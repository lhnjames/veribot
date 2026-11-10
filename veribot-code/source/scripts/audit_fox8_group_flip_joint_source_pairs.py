"""Replay joint redirection of every raw relation pointer incident to each target."""

from __future__ import annotations

from collections import Counter
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_fox8_group_flip_source_pairs import encoded
from audit_fox8_interaction_all_tweet_deletions import pointers
from audit_fox8_interaction_raw_tweet_deletions import selected_raw_rows, sha256
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
PAIR = ROOT / "research/certificates/fox8_group_flip_source_pairs.json"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
OUTPUT = ROOT / "research/certificates/fox8_group_flip_joint_source_pairs.json"


def run() -> dict:
    source = json.loads(PAIR.read_text(encoding="utf-8"))
    if (source["fixed_test_labels_read"] is not False or
            source["event_index_sha256"] != sha256(EVENTS)):
        raise ValueError("pair source certificate or event index changed")
    event_index = json.loads(EVENTS.read_text(encoding="utf-8"))["events"]
    all_targets = {t["target_account_id"] for r in source["runs"]
                   for t in r["target_counts"]}
    raw_rows = selected_raw_rows(all_targets)
    runs = []
    for prior in source["runs"]:
        seed = prior["seed"]
        runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
        if prior["model_sha256"] != sha256(runtime.run_dir / "model.joblib"):
            raise ValueError("model differs from source pair certificate")
        by_target = {}
        for item in prior["pair_operations"]:
            by_target.setdefault(item["target_account_id"], []).append(item)
        target_ids = [item["target_account_id"] for item in prior["target_counts"]]
        alt_facts = []
        before_facts = []
        before_supports = []
        after_supports = []
        rows = []
        known = set(runtime.ids)
        for target in target_ids:
            idx = runtime.index[target]
            neighbors = set(runtime.neighbors[idx])
            pairs = by_target[target]
            if {x["neighbor_account_id"] for x in pairs} != neighbors:
                raise ValueError("pair operations do not exhaust native neighbors")
            expected = [e for e in event_index
                        if target in (str(e["source_id"]), str(e["target_id"]))
                        and str(e["source_id"]) != str(e["target_id"])]
            source_locator_set = {
                (loc["source_account_id"], loc["old_target_account_id"],
                 loc["relation"], loc["source_line"], loc["tweet_index"])
                for pair in pairs for loc in pair["raw_relation_pointers"]}
            event_locator_set = {
                (str(e["source_id"]), str(e["target_id"]), str(e["relation"]),
                 int(loc["line"]), int(loc["tweet_index"]))
                for e in expected for loc in e["locators"]}
            if source_locator_set != event_locator_set:
                raise ValueError("joint raw pointer set differs from full event index")
            line_number, _, record = raw_rows[target]
            denominators = Counter(
                kind for tweet in record["user_tweets"]
                for kind, _, _ in pointers(tweet, target, known))
            revised = runtime.facts[idx].copy()
            revised[31:41] = 0.0
            revised[41] = 0.0 if denominators["reply"] else np.nan
            revised[42] = 0.0 if denominators["retweet"] else np.nan
            revised[43:74] = np.nan
            before_facts.append(runtime.facts[idx])
            before_supports.append(runtime.supports[idx])
            alt_facts.append(revised)
            after_supports.append(np.zeros(31, dtype=np.int32))
            rows.append({
                "target_account_id": target,
                "target_raw_line": line_number,
                "native_neighbor_count": len(neighbors),
                "raw_relation_pointer_count": len(source_locator_set),
                "single_pair_operations": len(pairs),
                "single_pair_class_change_count": sum(
                    p["target_class_changed"] for p in pairs),
            })
        before_matrix = encoded(runtime, np.asarray(before_facts),
                                np.asarray(before_supports))
        after_matrix = encoded(runtime, np.asarray(alt_facts),
                               np.asarray(after_supports))
        full = expit(np.asarray(runtime.model.booster_.predict(
            before_matrix, raw_score=True), dtype=float))
        after = expit(np.asarray(runtime.model.booster_.predict(
            after_matrix, raw_score=True), dtype=float))
        group_mask = before_matrix.copy()
        group_mask[:, 43:74] = 0.0
        group_mask[:, 117:148] = 0.0
        group_p = expit(np.asarray(runtime.model.booster_.predict(
            group_mask, raw_score=True), dtype=float))
        baseline_max = max(abs(float(full[i])-
                               by_target[target][0]["target_probability_before"])
                           for i, target in enumerate(target_ids))
        if baseline_max > 1e-12:
            raise ValueError("full source model does not reproduce pair baseline")
        for i, row in enumerate(rows):
            row["probability_before"] = float(full[i])
            row["probability_after_joint_source"] = float(after[i])
            row["probability_after_encoded_neighbor_mask"] = float(group_p[i])
            row["joint_source_class_change"] = bool(
                (full[i] >= runtime.threshold) !=
                (after[i] >= runtime.threshold))
            row["encoded_neighbor_mask_class_change"] = bool(
                (full[i] >= runtime.threshold) !=
                (group_p[i] >= runtime.threshold))
        if not all(r["encoded_neighbor_mask_class_change"] for r in rows):
            raise ValueError("source target no longer has neighbor group class flip")
        runs.append({
            "seed": seed,
            "targets": len(rows),
            "joint_source_class_changes": sum(
                r["joint_source_class_change"] for r in rows),
            "joint_source_score_changes": int(np.count_nonzero(
                np.abs(after-full) > 1e-10)),
            "encoded_neighbor_mask_class_changes": len(rows),
            "joint_source_and_encoded_mask_class_disagreements": sum(
                r["joint_source_class_change"] !=
                r["encoded_neighbor_mask_class_change"] for r in rows),
            "maximum_joint_source_vs_encoded_mask_probability_difference":
                float(np.max(np.abs(after-group_p))),
            "maximum_original_replay_error": baseline_max,
            "target_operations": rows,
        })
        print(json.dumps({k: runs[-1][k] for k in (
            "seed", "targets", "joint_source_class_changes",
            "joint_source_and_encoded_mask_class_disagreements")}), flush=True)
    result = {
        "status": "all_group_pivotal_fox8_targets_joint_incident_source_pointer_response",
        "operation": "redirect every incident raw reply/retweet target pointer to an external account simultaneously; preserve tweet/profile content and reply/retweet status; rebuild native facts and empty neighbor means/support; frozen reference/model",
        "scope_limit": "joint redirection is a hypothetical source intervention; it neither certifies a real error nor establishes unique minimal source sets",
        "fixed_test_labels_read": False,
        "pair_certificate_sha256": sha256(PAIR),
        "event_index_sha256": sha256(EVENTS),
        "runs": runs,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
