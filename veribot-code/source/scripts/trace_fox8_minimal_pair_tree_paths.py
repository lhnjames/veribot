"""Trace one exact two-source Fox8 class flip through frozen tree leaves."""

from __future__ import annotations

from collections import Counter
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_fox8_group_flip_source_pairs import encoded
from audit_fox8_interaction_all_tweet_deletions import native_values, pointers
from audit_fox8_interaction_raw_tweet_deletions import selected_raw_rows, sha256
from verifiable_bot.fox8_adapter import FEATURE_FIELDS
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
MINIMAL = ROOT / "research/certificates/fox8_minimal_source_pair_coalitions.json"
PAIR = ROOT / "research/certificates/fox8_group_flip_source_pairs.json"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
OUTPUT = ROOT / "research/certificates/fox8_minimal_pair_tree_paths.json"


def path_for(tree: dict, row: np.ndarray, names: list[str]) -> tuple[int, float, list[dict]]:
    node = tree["tree_structure"]
    path = []
    while "split_feature" in node:
        feature = int(node["split_feature"])
        value = float(row[feature])
        threshold = float(node["threshold"])
        if node["decision_type"] != "<=":
            raise ValueError("unhandled frozen tree decision type")
        left = bool(node["default_left"]) if np.isnan(value) else value <= threshold
        path.append({
            "encoded_column": feature,
            "fact": names[feature % 74],
            "tail": "upper" if feature < 74 else "lower",
            "encoded_value": value,
            "threshold": threshold,
            "branch": "left" if left else "right",
        })
        node = node["left_child"] if left else node["right_child"]
    return int(node["leaf_index"]), float(node["leaf_value"]), path


def run() -> dict:
    minimal = json.loads(MINIMAL.read_text(encoding="utf-8"))
    pair = json.loads(PAIR.read_text(encoding="utf-8"))
    selected = sorted((r["seed"], x) for r in minimal["runs"]
                      for x in r["targets"]
                      if x["minimum_source_pairs_changing_class"] == 2)
    if len(selected) != 1:
        raise ValueError("unique two-pair diagnostic example changed")
    seed, case = selected[0]
    target = case["target_account_id"]
    chosen = case["lowest_bitmask_minimal_neighbor_ids"]
    runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
    raw = selected_raw_rows({target})[target][2]
    denominator = Counter(kind for tweet in raw["user_tweets"]
                          for kind, _, _ in pointers(tweet, target,
                                                     set(runtime.ids)))
    events = json.loads(EVENTS.read_text(encoding="utf-8"))["events"]
    counts = Counter({
        (str(e["source_id"]), str(e["target_id"]), str(e["relation"])):
            int(e["count"]) for e in events})
    frame = pd.read_csv(runtime.features_path)
    if frame["user_id"].astype(str).tolist() != runtime.ids:
        raise ValueError("published profile population differs")
    published = frame[list(FEATURE_FIELDS)].to_numpy(dtype=float)
    idx = runtime.index[target]
    facts = []
    supports = []
    for mask in range(4):
        removed = {chosen[j] for j in range(2) if mask & (1 << j)}
        after_counts = counts.copy()
        for (source, destination, kind) in counts:
            if frozenset((source, destination)) in {
                    frozenset((target, peer)) for peer in removed}:
                after_counts[(source, destination, kind)] = 0
        fractions = tuple(
            (sum(n for (source, _, relation), n in after_counts.items()
                 if source == target and relation == kind) / denominator[kind])
            if denominator[kind] else np.nan
            for kind in ("reply", "retweet"))
        revised = runtime.facts[idx].copy()
        revised[31:43] = native_values(target, after_counts, fractions)
        remaining = [runtime.index[uid] for uid in runtime.neighbors[idx]
                     if uid not in removed]
        if remaining:
            vals = published[remaining]
            support = np.isfinite(vals).sum(axis=0).astype(np.int32)
            means = np.nansum(vals, axis=0) / np.maximum(support, 1)
            means[support == 0] = np.nan
            revised[43:74] = means.astype(np.float32)
        else:
            support = np.zeros(31, dtype=np.int32)
            revised[43:74] = np.nan
        facts.append(revised)
        supports.append(support)
    facts = np.asarray(facts)
    supports = np.asarray(supports)
    matrix = encoded(runtime, facts, supports)
    booster = runtime.model.booster_
    logits = np.asarray(booster.predict(matrix, raw_score=True), dtype=float)
    probabilities = expit(logits)
    prior_run = next(r for r in pair["runs"] if r["seed"] == seed)
    previous_pairs = {r["neighbor_account_id"]: r
                      for r in prior_run["pair_operations"]
                      if r["target_account_id"] == target}
    expected = [case["probability_before"],
                previous_pairs[chosen[0]]["target_probability_after"],
                previous_pairs[chosen[1]]["target_probability_after"],
                case["probability_after_first_minimal"]]
    if np.max(np.abs(probabilities-np.asarray(expected))) > 1e-12:
        raise ValueError("four-state frozen response differs from source certificates")
    names = list(runtime.reference.names)
    trees = booster.dump_model()["tree_info"]
    leaves = np.asarray(booster.predict(matrix, pred_leaf=True), dtype=int)
    if leaves.shape != (4, len(trees)):
        raise ValueError("frozen leaf matrix differs")
    per_tree = []
    sums = np.zeros(4, dtype=float)
    for tree_index, tree in enumerate(trees):
        paths = [path_for(tree, matrix[i], names) for i in range(4)]
        for i, (leaf_id, leaf_value, _) in enumerate(paths):
            if leaf_id != leaves[i, tree_index]:
                raise ValueError("tree path does not reproduce native leaf index")
            sums[i] += leaf_value
        values = [p[1] for p in paths]
        interaction = values[3] - values[1] - values[2] + values[0]
        if any(paths[i][0] != paths[0][0] for i in (1, 2, 3)) or abs(interaction) > 1e-15:
            branches = [p[2] for p in paths]
            first_divergence = None
            for old, new in zip(branches[0], branches[3]):
                if (old["encoded_column"] != new["encoded_column"] or
                        old["branch"] != new["branch"]):
                    first_divergence = {"before": old, "after": new}
                    break
            per_tree.append({
                "tree_index": tree_index,
                "leaf_indices": [p[0] for p in paths],
                "leaf_values": values,
                "joint_minus_original_leaf_logit": values[3]-values[0],
                "two_source_interaction_leaf_logit": interaction,
                "first_joint_path_divergence": first_divergence,
                "paths": branches,
            })
    if np.max(np.abs(sums-logits)) > 1e-10:
        raise ValueError("sum of native leaf values does not reproduce raw logits")
    interaction_total = float(logits[3]-logits[1]-logits[2]+logits[0])
    if abs(sum(r["two_source_interaction_leaf_logit"] for r in per_tree)
           - interaction_total) > 1e-10:
        raise ValueError("tree interactions do not reconstruct total interaction")
    changed = [
        {"fact": names[j],
         "before": float(facts[0, j]) if np.isfinite(facts[0, j]) else None,
         "after_joint": float(facts[3, j]) if np.isfinite(facts[3, j]) else None,
         "encoded_upper_before": float(matrix[0, j]),
         "encoded_upper_joint": float(matrix[3, j]),
         "encoded_lower_before": float(matrix[0, j+74]),
         "encoded_lower_joint": float(matrix[3, j+74])}
        for j in range(74) if not np.array_equal(facts[0, j:j+1],
                                                  facts[3, j:j+1], equal_nan=True)]
    first_features = Counter(
        x["first_joint_path_divergence"]["before"]["fact"]
        for x in per_tree if x["first_joint_path_divergence"] is not None)
    result = {
        "status": "two_raw_source_pair_class_flip_reconstructed_through_frozen_tree_leaves",
        "selection": "the unique size-two minimal source-pair coalition in the earlier exact at-most-12-neighbor sweep",
        "seed": seed,
        "target_account_id": target,
        "removed_neighbor_account_ids": chosen,
        "raw_source_pointers": case["first_minimal_raw_pointers"],
        "model_sha256": sha256(runtime.run_dir / "model.joblib"),
        "minimal_coalition_certificate_sha256": sha256(MINIMAL),
        "raw_source_sha256": sha256(runtime.raw_path),
        "fixed_test_labels_read": False,
        "threshold": runtime.threshold,
        "state_order": ["original", "remove_first_pair", "remove_second_pair",
                        "remove_both_pairs"],
        "probabilities": probabilities.tolist(),
        "raw_logits": logits.tolist(),
        "classes": (probabilities >= runtime.threshold).astype(int).tolist(),
        "changed_fact_count": len(changed),
        "changed_facts_original_to_joint": changed,
        "changed_tree_count": len(per_tree),
        "tree_leaf_sum_max_error": float(np.max(np.abs(sums-logits))),
        "joint_minus_original_logit": float(logits[3]-logits[0]),
        "two_source_interaction_logit": interaction_total,
        "first_joint_path_divergence_fact_counts": dict(first_features),
        "changed_tree_paths": per_tree,
        "scope_limit": "exact frozen function trace for one preselected case under pointer redirection; no general semantic or natural-error conclusion",
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n",
                      encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "seed", "target_account_id", "probabilities", "changed_fact_count",
        "changed_tree_count", "joint_minus_original_logit",
        "two_source_interaction_logit", "tree_leaf_sum_max_error")}))
    return result


if __name__ == "__main__":
    run()
