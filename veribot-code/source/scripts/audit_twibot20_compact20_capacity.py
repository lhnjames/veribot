"""Assess a 20-fact, no-lexicon TwiBot-20 evidence bottleneck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import metric, sha256, threshold_on_dev
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
    parser.add_argument("--no-count-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(args.no_count_control.read_text(encoding="utf-8"))
    if (sha256(args.facts) != prior["base_facts_sha256"]
            or sha256(args.reference) != prior["reference_artifact_sha256"]):
        raise ValueError("Prior frozen facts or reference differ")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        split = saved["split"].astype(str)
        labels = saved["labels"].astype(int)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        names = saved["names"].astype(str).tolist()
    train, dev, test = (np.flatnonzero(split == part) for part in ("train", "dev", "test"))
    retained = prior["retained_structured_facts"]
    if (len(retained) != 20 or set(retained) != set(names) - {"verified", "tweet_sample_count"}
            or set().union(*map(set, GROUPS.values())) != set(retained)
            or sum(map(len, GROUPS.values())) != 20):
        raise ValueError("20-fact evidence vocabulary differs")
    reference = LabelBlindReference.load(args.reference)
    if reference.names != tuple(names):
        raise ValueError("Reference names differ")
    evidence = reference.transform(values, contexts=contexts)
    encoded = np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                               np.log1p(evidence.lower) * evidence.reliability))
    keep = [names.index(name) for name in retained]
    keep += [names.index(name) + len(names) for name in retained]
    x = encoded[:, keep]
    model = LogisticRegression(C=1.0, penalty="l1", solver="liblinear",
                               max_iter=1000, random_state=0)
    model.fit(x[train], labels[train])
    dev_p = model.predict_proba(x[dev])[:, 1]
    threshold = threshold_on_dev(dev_p, labels[dev])
    coefficient = model.coef_.ravel()
    intercept = float(model.intercept_[0])
    test_logit = intercept + x[test] @ coefficient
    test_p = expit(test_logit)
    if np.max(np.abs(test_p - model.predict_proba(x[test])[:, 1])) > 1e-12:
        raise ValueError("Compact model direct prediction differs")
    prior_rows = prior["rows"]
    if [row["account_id"] for row in prior_rows] != ids[test].tolist():
        raise ValueError("Prior full-input test population differs")
    prior_class = np.asarray([row["original_class"] for row in prior_rows], dtype=int)
    prior_logit = np.asarray([row["original_logit"] for row in prior_rows], dtype=float)
    prior_threshold = float(prior["threshold"])
    if not np.array_equal(prior_class, expit(prior_logit) >= prior_threshold):
        raise ValueError("Prior full-input classes differ")
    compact_class = test_p >= threshold
    effects = {}
    for group_name, group_fields in GROUPS.items():
        cols = [k for k, name in enumerate(retained) if name in group_fields]
        cols += [k + len(retained) for k, name in enumerate(retained) if name in group_fields]
        effects[group_name] = x[test][:, cols] @ coefficient[cols]
    reconstruction_error = float(np.max(np.abs(test_logit - (intercept + sum(effects.values())))))
    if reconstruction_error > 1e-10:
        raise ValueError("Compact group effects do not reconstruct logits")
    group_mask = {}
    for group_name, effect in effects.items():
        masked_class = expit(test_logit - effect) >= threshold
        group_mask[group_name] = {
            "nonzero_score_effect_accounts": int(np.count_nonzero(np.abs(effect) > 1e-12)),
            "masked_class_changes": int(np.count_nonzero(compact_class != masked_class)),
            "masked_errors": int(np.count_nonzero(masked_class != labels[test])),
            "mean_absolute_logit_effect": float(np.mean(np.abs(effect))),
        }
    subgroups = {}
    verified_col = names.index("verified")
    for verified in (0, 1):
        for label in (0, 1):
            mask = (values[test, verified_col] == verified) & (labels[test] == label)
            subgroups[f"verified={verified},label={label}"] = {
                "n": int(np.count_nonzero(mask)),
                "compact_correct": int(np.count_nonzero(compact_class[mask] == labels[test][mask])),
                "prior_with_unigrams_correct": int(np.count_nonzero(prior_class[mask] == labels[test][mask])),
            }
    result = {
        "scope": "post hoc same-protocol TwiBot-20 test diagnostic; prior test population explored",
        "source_archive_sha256": prior["source_archive_sha256"],
        "base_facts_sha256": prior["base_facts_sha256"],
        "reference_artifact_sha256": prior["reference_artifact_sha256"],
        "prior_no_count_control_sha256": sha256(args.no_count_control),
        "protocol": "20 audited structured facts, dual-tail label-blind reference, L1 C=1 liblinear, dev Macro-F1 threshold, no lexical features",
        "retained_facts": retained,
        "active_tail_terms": int(np.count_nonzero(coefficient)),
        "threshold": threshold,
        "intercept": intercept,
        "all_zero_evidence_margin": float(intercept - logit(threshold)),
        "group_reconstruction_max_error": reconstruction_error,
        "dev": metric(labels[dev], dev_p, threshold),
        "test": metric(labels[test], test_p, threshold),
        "group_mask_diagnostics": group_mask,
        "subgroups": subgroups,
        "paired_with_prior_full_input": {
            "prior_errors": prior["test"]["errors"],
            "compact_only_correct": int(np.count_nonzero((compact_class == labels[test])
                                                             & (prior_class != labels[test]))),
            "prior_only_correct": int(np.count_nonzero((compact_class != labels[test])
                                                           & (prior_class == labels[test]))),
            "difference_test_acc_percentage_points": float(100 * (
                np.mean(compact_class == labels[test]) - np.mean(prior_class == labels[test]))),
            "difference_test_macro_f1_percentage_points": float(100 * (
                metric(labels[test], test_p, threshold)["macro_f1"]
                - prior["test"]["macro_f1"])),
        },
        "coefficient_by_fact": {
            name: {"upper": float(coefficient[k]),
                   "lower": float(coefficient[k + len(retained)])}
            for k, name in enumerate(retained)},
        "rows": [{"account_id": str(ids[i]), "label_for_evaluation": int(labels[i]),
                  "compact_logit": float(test_logit[k]), "compact_probability": float(test_p[k]),
                  "compact_class": int(compact_class[k]),
                  "prior_with_unigrams_class": int(prior_class[k]),
                  "group_effects": {name: float(effect[k]) for name, effect in effects.items()}}
                 for k, i in enumerate(test)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
