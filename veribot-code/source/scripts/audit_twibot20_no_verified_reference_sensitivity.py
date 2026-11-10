"""Compare zero masking with a label-blind observed train anchor for tweet statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


TWEET_FIELDS = ("tweet_sample_count", "retweet_fraction", "url_fraction",
                "hashtag_fraction", "mention_fraction", "mean_tweet_length",
                "unique_tweet_fraction")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = json.loads(args.control.read_text(encoding="utf-8"))
    if (sha256(args.facts) != control["base_facts_sha256"]
            or sha256(args.reference) != control["reference_artifact_sha256"]):
        raise ValueError("Frozen fact cache or reference differs")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        split = saved["split"].astype(str)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        labels = saved["labels"].astype(int)
        names = saved["names"].astype(str).tolist()
    train, test = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
    rows = control["rows"]
    if [row["account_id"] for row in rows] != ids[test].tolist():
        raise ValueError("Test population order differs")
    reference = LabelBlindReference.load(args.reference)
    if reference.names != tuple(names):
        raise ValueError("Reference fact names differ")
    evidence = reference.transform(values, contexts=contexts)
    encoded = np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                               np.log1p(evidence.lower) * evidence.reliability))
    group_cols = [names.index(name) for name in TWEET_FIELDS]
    group_cols += [j + len(names) for j in group_cols]
    train_group = encoded[train][:, group_cols]
    center = np.median(train_group, axis=0)
    distance = np.sum((train_group - center) ** 2, axis=1)
    # The first official-train occurrence breaks an exact distance tie.
    anchor_train_position = int(np.argmin(distance))
    anchor_index = int(train[anchor_train_position])
    coefficient = control["without_verified"]["structured_coefficients_by_fact"]
    group_coefficient = np.asarray(
        [coefficient[name]["upper"] for name in TWEET_FIELDS]
        + [coefficient[name]["lower"] for name in TWEET_FIELDS], dtype=float)
    anchor_effect = float(train_group[anchor_train_position] @ group_coefficient)
    test_effect = encoded[test][:, group_cols] @ group_coefficient
    saved_effect = np.asarray([row["without_verified_group_effects"]["sampled_tweet_stats"]
                               for row in rows], dtype=float)
    effect_error = float(np.max(np.abs(test_effect - saved_effect)))
    if effect_error > 1e-10:
        raise ValueError("Tweet group does not replay previous frozen certificate")
    original_logit = np.asarray([row["without_verified_logit"] for row in rows], dtype=float)
    threshold = float(control["without_verified"]["threshold"])
    original_class = expit(original_logit) >= threshold
    zero_logit = original_logit - test_effect
    anchor_logit = original_logit - test_effect + anchor_effect
    zero_class = expit(zero_logit) >= threshold
    anchor_class = expit(anchor_logit) >= threshold
    if int(np.count_nonzero(zero_class != original_class)) != control["without_verified"][
            "group_diagnostics"]["sampled_tweet_stats"]["masked_class_changes"]:
        raise ValueError("Zero-mask control differs")
    result = {
        "scope": "TwiBot-20 no-verified retrained model; official test reference sensitivity",
        "interpretation": "encoded-group substitutions only; not a source edit or natural correction",
        "selection": "label-blind observed official-train medoid of 14 encoded tweet-stat coordinates",
        "base_facts_sha256": control["base_facts_sha256"],
        "reference_artifact_sha256": control["reference_artifact_sha256"],
        "control_certificate_sha256": sha256(args.control),
        "anchor_account_id": str(ids[anchor_index]),
        "anchor_official_train_position": anchor_train_position,
        "anchor_group_effect": anchor_effect,
        "anchor_source_facts": {name: float(values[anchor_index, names.index(name)])
                                for name in TWEET_FIELDS},
        "group_effect_replay_max_error": effect_error,
        "test_accounts": len(test),
        "original_bot_predictions": int(np.count_nonzero(original_class)),
        "zero_mask_class_changes": int(np.count_nonzero(zero_class != original_class)),
        "observed_anchor_replace_class_changes": int(np.count_nonzero(anchor_class != original_class)),
        "zero_mask_errors": int(np.count_nonzero(zero_class != labels[test])),
        "observed_anchor_replace_errors": int(np.count_nonzero(anchor_class != labels[test])),
        "original_errors": int(np.count_nonzero(original_class != labels[test])),
        "test_group_effect_min": float(np.min(test_effect)),
        "test_group_effect_median": float(np.median(test_effect)),
        "test_group_effect_max": float(np.max(test_effect)),
        "rows": [
            {"account_id": str(ids[index]), "label_for_evaluation": int(labels[index]),
             "original_probability": float(expit(original_logit[k])),
             "zero_mask_probability": float(expit(zero_logit[k])),
             "observed_anchor_replace_probability": float(expit(anchor_logit[k])),
             "original_group_effect": float(test_effect[k]),
             "original_class": int(original_class[k]), "zero_mask_class": int(zero_class[k]),
             "observed_anchor_replace_class": int(anchor_class[k])}
            for k, index in enumerate(test)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
