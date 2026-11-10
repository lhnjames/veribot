"""Audit tweet sample support and screen one fixed support-aware dev-only variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import metric, sha256, threshold_on_dev
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


TWEET_FACTS = ("retweet_fraction", "url_fraction", "hashtag_fraction",
               "mention_fraction", "mean_tweet_length", "unique_tweet_fraction")
SUPPORT_PRIOR = 20.0


def get_term(card: dict, name: str) -> dict:
    return next(term for term in card["fact_terms"] if term["fact_name"] == name)


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
    if (sha256(args.facts) != compact["base_facts_sha256"]
            or sha256(args.reference) != compact["reference_artifact_sha256"]
            or sha256(args.compact) != cards["compact_certificate_sha256"]):
        raise ValueError("Frozen source, reference, model and cards differ")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        split = saved["split"].astype(str)
        labels = saved["labels"].astype(int)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        names = saved["names"].astype(str).tolist()
    train, dev, test = (np.flatnonzero(split == part) for part in ("train", "dev", "test"))
    if ids[test].tolist() != [row["account_id"] for row in cards["decision_cards"]]:
        raise ValueError("Official test and card order differ")
    reference = LabelBlindReference.load(args.reference)
    if reference.support_prior != 0:
        raise ValueError("This diagnostic expects the saved zero-support-prior reference")
    evidence = reference.transform(values, contexts=contexts)
    retained = compact["retained_facts"]
    keep = [names.index(name) for name in retained]
    keep += [names.index(name) + len(names) for name in retained]
    encoded = np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                               np.log1p(evidence.lower) * evidence.reliability))[:, keep]
    sample_count = values[:, names.index("tweet_sample_count")]
    if not np.isfinite(sample_count).all() or np.any(sample_count < 0):
        raise ValueError("Tweet sample counts are invalid")
    factor = sample_count / (sample_count + SUPPORT_PRIOR)
    adjusted = encoded.copy()
    for name in TWEET_FACTS:
        j = retained.index(name)
        adjusted[:, j] *= factor
        adjusted[:, j + len(retained)] *= factor
    old = LogisticRegression(C=1.0, penalty="l1", solver="liblinear",
                             max_iter=1000, random_state=0)
    old.fit(encoded[train], labels[train])
    if (abs(float(old.intercept_[0]) - compact["intercept"]) > 1e-10
            or any(abs(float(old.coef_[0, j]) - compact["coefficient_by_fact"][name]["upper"])
                   > 1e-10 for j, name in enumerate(retained))):
        raise ValueError("The saved compact model cannot be refitted")
    old_dev_p = old.predict_proba(encoded[dev])[:, 1]
    old_threshold = threshold_on_dev(old_dev_p, labels[dev])
    if abs(old_threshold - compact["threshold"]) > 1e-12:
        raise ValueError("Original dev threshold differs")
    revised = LogisticRegression(C=1.0, penalty="l1", solver="liblinear",
                                 max_iter=1000, random_state=0)
    revised.fit(adjusted[train], labels[train])
    revised_dev_p = revised.predict_proba(adjusted[dev])[:, 1]
    revised_threshold = threshold_on_dev(revised_dev_p, labels[dev])
    bins = {"zero": sample_count[test] == 0,
            "one": sample_count[test] == 1,
            "two_to_five": (sample_count[test] >= 2) & (sample_count[test] <= 5),
            "six_to_twenty": (sample_count[test] >= 6) & (sample_count[test] <= 20),
            "twenty_one_to_199": (sample_count[test] >= 21) & (sample_count[test] < 200),
            "two_hundred": sample_count[test] == 200}
    original_bin_audit = {}
    for label, mask in bins.items():
        selected = [cards["decision_cards"][int(k)] for k in np.flatnonzero(mask)]
        unique = [get_term(card, "unique_tweet_fraction") for card in selected]
        original_bin_audit[label] = {
            "accounts": len(selected),
            "distinct_unique_fraction_reliabilities": sorted(set(
                round(float(term["reliability"]), 6) for term in unique)),
            "mean_unique_fraction_logit_effect": (float(np.mean([
                term["total_logit_effect"] for term in unique])) if unique else None),
            "wrong_predictions": sum(not card["prediction_correct"] for card in selected),
        }
    one = np.flatnonzero(sample_count[test] == 1)
    full = np.flatnonzero(sample_count[test] == 200)
    same_value = values[test, names.index("unique_tweet_fraction")]
    pair = next((int(i), int(j)) for i in one for j in full
                if contexts[test[i]] == contexts[test[j]]
                and same_value[i] == same_value[j] == 1.0)
    a = get_term(cards["decision_cards"][pair[0]], "unique_tweet_fraction")
    b = get_term(cards["decision_cards"][pair[1]], "unique_tweet_fraction")
    if any(abs(float(a[key]) - float(b[key])) > 1e-12 for key in (
            "upper_tail_score", "lower_tail_score", "reliability", "total_logit_effect")):
        raise ValueError("Equal raw ratio in the same context received different original evidence")
    result = {
        "scope": "TwiBot-20 compact20 sample-size reliability audit; original official test post hoc audit, revised candidate dev only",
        "facts_sha256": sha256(args.facts),
        "reference_sha256": sha256(args.reference),
        "compact_sha256": sha256(args.compact),
        "cards_sha256": sha256(args.cards),
        "saved_reference_support_prior": reference.support_prior,
        "saved_transform_uses_per_account_tweet_support": False,
        "same_context_equal_ratio_pair": {
            "one_tweet_account_id": cards["decision_cards"][pair[0]]["account_id"],
            "two_hundred_tweet_account_id": cards["decision_cards"][pair[1]]["account_id"],
            "context": str(contexts[test[pair[0]]]),
            "unique_tweet_fraction": 1.0,
            "both_reliability": float(a["reliability"]),
            "both_total_logit_effect": float(a["total_logit_effect"]),
            "sample_counts": [1, 200],
        },
        "original_test_sample_count_bins": original_bin_audit,
        "screened_support_rule": "For six sampled-tweet facts only, multiply existing encoded upper/lower tails by n/(n+20); n is official sampled tweet count. n=20 gives half weight. The constant 20 was fixed before viewing candidate dev results; profile and graph encodings are unchanged.",
        "screened_support_prior": SUPPORT_PRIOR,
        "screened_factor_for_one": float(1 / (1 + SUPPORT_PRIOR)),
        "screened_factor_for_two_hundred": float(200 / (200 + SUPPORT_PRIOR)),
        "original_dev": metric(labels[dev], old_dev_p, old_threshold),
        "screened_dev": metric(labels[dev], revised_dev_p, revised_threshold),
        "screened_dev_threshold": float(revised_threshold),
        "screened_active_tail_terms": int(np.count_nonzero(revised.coef_)),
        "candidate_test_evaluated": False,
        "interpretation": "The existing reliability has no per-account tweet-count factor. The support-aware variant is a dev-only screen, not an independently selected or test-validated replacement. No behavioral semantic truth is inferred.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
