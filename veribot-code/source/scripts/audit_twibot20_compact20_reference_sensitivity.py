"""Audit reference dependence of the frozen TwiBot-20 20-fact decision cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


GROUPS = {
    "profile_counts": ("followers_count", "friends_count", "statuses_count",
                       "listed_count", "favourites_count"),
    "profile_presentation": ("protected", "default_profile", "default_profile_image",
                             "description_length"),
    "sampled_tweet_stats": ("retweet_fraction", "url_fraction", "hashtag_fraction",
                            "mention_fraction", "mean_tweet_length", "unique_tweet_fraction"),
    "native_follow_graph": ("sampled_following_count", "sampled_follower_count",
                            "graph_out_degree", "graph_in_degree", "graph_reciprocal_degree"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    for key, path in (("base_facts_sha256", args.facts),
                      ("reference_artifact_sha256", args.reference)):
        if sha256(path) != compact[key] or cards[key] != compact[key]:
            raise ValueError(f"Frozen input differs: {key}")
    if cards["compact_certificate_sha256"] != sha256(args.compact):
        raise ValueError("Decision cards do not bind to this compact model")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        splits = saved["split"].astype(str)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        positions = saved["positions"].astype(int)
        names = saved["names"].astype(str).tolist()
    train = np.flatnonzero(splits == "train")
    test = np.flatnonzero(splits == "test")
    if len(test) != cards["cards"] or [card["account_id"] for card in cards[
            "decision_cards"]] != ids[test].tolist():
        raise ValueError("Official test population and decision-card order differ")
    reference = LabelBlindReference.load(args.reference)
    if reference.names != tuple(names):
        raise ValueError("Fact vocabulary differs from the label-blind reference")
    evidence = reference.transform(values, contexts=contexts)
    all_encoded = np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                                   np.log1p(evidence.lower) * evidence.reliability))
    retained = compact["retained_facts"]
    if len(retained) != 20 or set(retained) != set().union(*map(set, GROUPS.values())):
        raise ValueError("The expected 20 facts are not present")
    selected = [names.index(name) for name in retained]
    selected += [j + len(names) for j in selected]
    encoded = all_encoded[:, selected]
    train_encoded = encoded[train]
    center = np.median(train_encoded, axis=0)
    distance = np.sum((train_encoded - center) ** 2, axis=1)
    # The first official train occurrence resolves an exact distance tie.
    anchor_train_position = int(np.argmin(distance))
    anchor_index = int(train[anchor_train_position])
    coefficients = np.asarray(
        [compact["coefficient_by_fact"][name]["upper"] for name in retained]
        + [compact["coefficient_by_fact"][name]["lower"] for name in retained],
        dtype=float)
    test_terms = encoded[test] * coefficients
    anchor_terms = encoded[anchor_index] * coefficients
    term_difference = test_terms - anchor_terms
    original_logit = float(compact["intercept"]) + np.sum(test_terms, axis=1)
    anchor_logit = float(compact["intercept"]) + float(np.sum(anchor_terms))
    threshold = float(compact["threshold"])
    threshold_logit = float(logit(threshold))
    anchor_margin = anchor_logit - threshold_logit
    replay_error = max(abs(float(original_logit[k]) - card["full_logit"])
                       for k, card in enumerate(cards["decision_cards"]))
    if replay_error > 1e-10:
        raise ValueError("Frozen logit cannot be reconstructed")
    group_result = {}
    rows = []
    group_indexes = {group: [retained.index(name) for name in members]
                     for group, members in GROUPS.items()}
    for group, indexes in group_indexes.items():
        cols = indexes + [j + len(retained) for j in indexes]
        test_effect = np.sum(test_terms[:, cols], axis=1)
        anchor_effect = float(np.sum(anchor_terms[cols]))
        observed_difference = test_effect - anchor_effect
        saved_effect = np.asarray([card["group_logit_effects"][group]
                                   for card in cards["decision_cards"]])
        if np.max(np.abs(test_effect - saved_effect)) > 1e-10:
            raise ValueError(f"Frozen group ledger differs: {group}")
        original_class = expit(original_logit) >= threshold
        zero_logit = original_logit - test_effect
        observed_logit = original_logit - test_effect + anchor_effect
        nonzero = (np.abs(test_effect) > 1e-12) & (np.abs(observed_difference) > 1e-12)
        reversed_sign = nonzero & (np.sign(test_effect) != np.sign(observed_difference))
        group_result[group] = {
            "anchor_logit_effect": anchor_effect,
            "zero_reference_sign_positive": int(np.count_nonzero(test_effect > 1e-12)),
            "observed_reference_sign_positive": int(np.count_nonzero(
                observed_difference > 1e-12)),
            "direction_reversals_nonzero_both": int(np.count_nonzero(reversed_sign)),
            "zero_encoded_group_substitution_class_changes": int(np.count_nonzero(
                (expit(zero_logit) >= threshold) != original_class)),
            "observed_anchor_group_substitution_class_changes": int(np.count_nonzero(
                (expit(observed_logit) >= threshold) != original_class)),
        }
    for k, card in enumerate(cards["decision_cards"]):
        differences = {group: float(np.sum(term_difference[k, indexes
                             + [j + len(retained) for j in indexes]]))
                       for group, indexes in group_indexes.items()}
        if abs(anchor_margin + sum(differences.values()) - card["full_margin"]) > 1e-10:
            raise ValueError("Observed-anchor group effects do not sum to test margin")
        rows.append({
            "account_id": card["account_id"],
            "full_margin": card["full_margin"],
            "prediction": card["prediction"],
            "prediction_correct": card["prediction_correct"],
            "zero_reference_group_effects": card["group_logit_effects"],
            "observed_reference_group_differences": differences,
        })
    output = {
        "scope": "Frozen compact 20-fact TwiBot-20 classifier, official test population",
        "operation": "Two exact logit decompositions of the same frozen predictions, plus encoded-group substitutions; no source edits or corrections",
        "facts_sha256": sha256(args.facts),
        "reference_sha256": sha256(args.reference),
        "compact_sha256": sha256(args.compact),
        "decision_cards_sha256": sha256(args.cards),
        "anchor_selection": "Official train account closest to the coordinate-wise median in 40 retained encoded dimensions, label-blind Euclidean distance; ties by original train order",
        "anchor_account_id": str(ids[anchor_index]),
        "anchor_official_train_position": int(positions[anchor_index]),
        "anchor_source_facts": {name: (float(values[anchor_index, names.index(name)])
                                        if np.isfinite(values[anchor_index, names.index(name)])
                                        else None) for name in retained},
        "zero_reference_margin": float(compact["all_zero_evidence_margin"]),
        "observed_reference_margin": anchor_margin,
        "observed_reference_prediction": int(expit(anchor_logit) >= threshold),
        "frozen_replay_max_logit_error": replay_error,
        "test_accounts": len(test),
        "group_results": group_result,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
