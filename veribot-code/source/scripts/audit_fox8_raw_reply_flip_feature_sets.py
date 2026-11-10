"""Exhaustively isolate encoded fact changes in five raw-reply class flips.

The five edits are selected by the existing raw-source-deletion certificate.
Hybrid feature vectors are mathematical diagnostics; only mask 255 is the
actual source-edited row. No claim is made that other masks are valid tweets.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CERT = ROOT / "research/certificates/fox8_interaction_all_tweet_deletions.json"
OUTPUT = ROOT / "research/certificates/fox8_raw_reply_flip_feature_sets.json"
EXPECTED = ["n_tweets", "retweet_ratio", "reply_ratio", "avg_tweet_len",
            "avg_favorites", "avg_retweets", "avg_score", "unique_lang_ratio"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_path(node: dict, values: np.ndarray) -> tuple[float, list[tuple[int, float, str]]]:
    steps = []
    while "split_feature" in node:
        feature = int(node["split_feature"])
        threshold = float(node["threshold"])
        if node["decision_type"] != "<=" or not np.isfinite(values[feature]):
            raise ValueError("unhandled LightGBM tree path")
        direction = "left" if values[feature] <= threshold else "right"
        steps.append((feature, threshold, direction))
        node = node[f"{direction}_child"]
    return float(node["leaf_value"]), steps


def run() -> dict:
    source = json.loads(SOURCE_CERT.read_text(encoding="utf-8"))
    case = next(row for row in source["cases"] if row["seed"] == 0)
    changed = [row for row in case["rows"] if row["target_class_changes"]]
    if case["account_id"] != "1642398165679714305" or len(changed) != 5:
        raise ValueError("preselected raw-reply flip cohort changed")
    runtime = Fox8NeighborAtomicRuntime(ROOT, 0)
    idx = runtime.index[case["account_id"]]
    if list(runtime.reference.names[23:31]) != EXPECTED:
        raise ValueError("sampled activity fact vocabulary changed")
    base = runtime.facts[idx].copy()
    threshold = runtime.threshold
    rows = []
    reply_tree_path = None
    for edit in changed:
        altered = {item["fact"]: item for item in edit["changed_target_model_facts"]}
        if list(altered) != EXPECTED or any(
                abs(float(altered[name]["before"])-base[23+j]) > 1e-12
                for j, name in enumerate(EXPECTED)):
            raise ValueError("raw edit changes facts outside the eight tweet aggregates")
        edited_values = np.asarray([float(altered[name]["after"])
                                    for name in EXPECTED])
        masks = np.arange(256, dtype=np.int32)
        hybrids = np.tile(base, (256, 1))
        for j in range(8):
            hybrids[(masks & (1 << j)) != 0, 23+j] = edited_values[j]
        evidence = runtime.reference.transform(hybrids)
        reliability = evidence.reliability.copy()
        reliability[:, 43:74] *= runtime.supports[idx] / (
            runtime.supports[idx] + 5.0)
        matrix = np.column_stack((np.log1p(evidence.upper)*reliability,
                                  np.log1p(evidence.lower)*reliability))
        probability = expit(np.asarray(runtime.model.booster_.predict(
            matrix, raw_score=True), dtype=float))
        if reply_tree_path is None:
            model_dump = runtime.model.booster_.dump_model()
            changed_trees = []
            for tree_index, tree in enumerate(model_dump["tree_info"]):
                first_value, first_path = tree_path(tree["tree_structure"], matrix[0])
                reply_value, reply_path = tree_path(
                    tree["tree_structure"], matrix[1 << 2])
                if first_value == reply_value:
                    continue
                divergence = next((a for a, b in zip(first_path, reply_path)
                                   if a != b), None)
                if divergence is None or divergence[0] not in (25, 99):
                    raise ValueError("reply-only tree diverged outside reply input")
                changed_trees.append({
                    "tree_index": tree_index,
                    "first_divergence_encoded_column": divergence[0],
                    "first_divergence_threshold": divergence[1],
                    "leaf_logit_change": reply_value-first_value,
                })
            full_logits = np.asarray(runtime.model.booster_.predict(
                matrix[[0, 1 << 2]], raw_score=True), dtype=float)
            leaf_sum = sum(row["leaf_logit_change"] for row in changed_trees)
            if abs(leaf_sum - float(full_logits[1]-full_logits[0])) > 1e-10:
                raise ValueError("LightGBM leaf changes do not reconstruct reply response")
            reply_tree_path = {
                "reply_upper_encoded_before_after": [float(matrix[0, 25]),
                                                       float(matrix[1 << 2, 25])],
                "reply_lower_encoded_before_after": [float(matrix[0, 99]),
                                                       float(matrix[1 << 2, 99])],
                "changed_tree_count": len(changed_trees),
                "sum_changed_leaf_logits": leaf_sum,
                "direct_reply_only_logit_change": float(full_logits[1]-full_logits[0]),
                "changed_tree_paths": changed_trees,
            }
        if (abs(float(probability[0])-edit["target_probability_before"]) > 1e-12 or
                abs(float(probability[255])-edit["target_probability_after"]) > 1e-12):
            raise ValueError("hybrid endpoints differ from raw-source response")
        if (probability[0] >= threshold) == (probability[255] >= threshold):
            raise ValueError("source edit no longer flips frozen class")
        flip = (probability >= threshold) != (probability[0] >= threshold)
        if not np.array_equal(flip, (masks & (1 << 2)) != 0):
            raise ValueError("five source flips no longer isolate reply_ratio")
        minimum = min(mask.bit_count() for mask in masks[flip])
        minimal = [int(mask) for mask in masks[flip]
                   if mask.bit_count() == minimum]
        necessary = [EXPECTED[j] for j in range(8)
                     if all(mask & (1 << j) for mask in minimal)]
        rows.append({
            "removed_tweet_position": edit["removed_tweet_position"],
            "raw_tweet_locator": edit["raw_tweet_locator"],
            "tweet_id": edit["tweet_id"],
            "original_probability": float(probability[0]),
            "raw_edited_probability": float(probability[255]),
            "threshold": threshold,
            "fact_names": EXPECTED,
            "single_field_probability": {
                name: float(probability[1 << j]) for j, name in enumerate(EXPECTED)},
            "single_field_changes_class": {
                name: bool(flip[1 << j]) for j, name in enumerate(EXPECTED)},
            "field_removed_from_full_probability": {
                name: float(probability[255 ^ (1 << j)])
                for j, name in enumerate(EXPECTED)},
            "minimum_changed_field_count_to_flip": minimum,
            "all_minimum_flip_masks": minimal,
            "fields_in_every_minimum_flip_mask": necessary,
            "hybrid_masks_that_flip_class": int(np.count_nonzero(flip)),
            "all_256_hybrid_probabilities": probability.tolist(),
        })
    result = {
        "status": "five_raw_reply_flips_have_exhaustive_eight_fact_hybrid_diagnostics",
        "model": "fox8_raw_neighbor_74_seed0",
        "actual_source_edit_mask": 255,
        "hybrid_scope_limit": "Masks 1..254 mix old and new aggregate values and need not correspond to any feasible raw tweet list; their effects do not establish single-record causality.",
        "fixed_test_labels_read": False,
        "source_certificate_sha256": sha256(SOURCE_CERT),
        "model_sha256": sha256(runtime.run_dir / "model.joblib"),
        "account_id": case["account_id"],
        "reply_only_frozen_tree_path": reply_tree_path,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n",
                      encoding="utf-8")
    for row in rows:
        print(row["removed_tweet_position"],
              row["minimum_changed_field_count_to_flip"],
              row["fields_in_every_minimum_flip_mask"],
              [name for name, value in row["single_field_changes_class"].items()
               if value])
    return result


if __name__ == "__main__":
    run()
