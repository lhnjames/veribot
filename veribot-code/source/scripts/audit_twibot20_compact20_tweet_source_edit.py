"""Source-level first-tweet edit for the frozen 20-fact compact reasoner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import ijson
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256
from verifiable_bot.twibot20_facts import account_facts
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    for key, path in (("source_archive_sha256", args.zip),
                      ("base_facts_sha256", args.facts),
                      ("reference_artifact_sha256", args.reference)):
        if sha256(path) != compact[key]:
            raise ValueError(f"Source or reference differs: {key}")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        split = saved["split"].astype(str)
        positions = saved["positions"].astype(int)
        labels = saved["labels"].astype(int)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        names = saved["names"].astype(str).tolist()
    test = np.flatnonzero(split == "test")
    if [row["account_id"] for row in compact["rows"]] != ids[test].tolist():
        raise ValueError("Compact model test population differs")
    original_facts = values[test].copy()
    edited_facts = original_facts.copy()
    source_rows = []
    with zipfile.ZipFile(args.zip) as zipped, zipped.open("Twibot-20/test.json") as stream:
        for position, row in enumerate(ijson.items(stream, "item")):
            if position >= len(test):
                raise ValueError("Official test member longer than fact cache")
            i = int(test[position])
            if ids[i] != str(row["ID"]) or positions[i] != position:
                raise ValueError("Official source account order differs")
            source_facts, _, context = account_facts(row)
            if context != contexts[i] or not np.allclose(
                    source_facts, values[i, :len(source_facts)],
                    rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError("Original source facts differ")
            tweets = list(row.get("tweet") or [])
            edit_facts, _, _ = account_facts(dict(row, tweet=tweets[1:] if tweets else tweets))
            edited_facts[position, :len(edit_facts)] = edit_facts
            source_rows.append({
                "account_id": str(ids[i]),
                "first_tweet_locator": f"Twibot-20/test.json[{position}].tweet[0]" if tweets else None,
                "original_tweet_count": len(tweets),
            })
    if len(source_rows) != len(test):
        raise ValueError("Official test member shorter than fact cache")
    reference = LabelBlindReference.load(args.reference)
    if reference.names != tuple(names):
        raise ValueError("Reference fact vocabulary differs")
    retained = compact["retained_facts"]
    if len(retained) != 20:
        raise ValueError("Compact model must read exactly 20 source facts")
    columns = [names.index(name) for name in retained]
    columns += [names.index(name) + len(names) for name in retained]
    coefficient = np.asarray([
        compact["coefficient_by_fact"][name]["upper"] for name in retained]
        + [compact["coefficient_by_fact"][name]["lower"] for name in retained], dtype=float)
    intercept = float(compact["intercept"])
    threshold = float(compact["threshold"])

    def encode(facts: np.ndarray) -> np.ndarray:
        evidence = reference.transform(facts, contexts=contexts[test])
        return np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                                np.log1p(evidence.lower) * evidence.reliability))[:, columns]

    original_encoded = encode(original_facts)
    edited_encoded = encode(edited_facts)
    original_logit = intercept + original_encoded @ coefficient
    edited_logit = intercept + edited_encoded @ coefficient
    saved_logit = np.asarray([row["compact_logit"] for row in compact["rows"]], dtype=float)
    replay_error = float(np.max(np.abs(original_logit - saved_logit)))
    if replay_error > 1e-10:
        raise ValueError("Compact source facts do not replay frozen model")
    tweet_fields = ("retweet_fraction", "url_fraction", "hashtag_fraction",
                    "mention_fraction", "mean_tweet_length", "unique_tweet_fraction")
    other_columns = [k for k, name in enumerate(retained * 2) if name not in tweet_fields]
    if not np.allclose(original_encoded[:, other_columns], edited_encoded[:, other_columns],
                       rtol=0, atol=0):
        raise ValueError("Tweet source edit altered an unrelated model input")
    field_delta = {}
    for field in tweet_fields:
        j = retained.index(field)
        field_delta[field] = ((edited_encoded[:, j] - original_encoded[:, j]) * coefficient[j]
                              + (edited_encoded[:, j + len(retained)]
                                 - original_encoded[:, j + len(retained)])
                              * coefficient[j + len(retained)])
    delta_error = float(np.max(np.abs((edited_logit - original_logit)
                                      - sum(field_delta.values()))))
    if delta_error > 1e-10:
        raise ValueError("Edited logit does not decompose into six source fact changes")
    original_class = expit(original_logit) >= threshold
    edited_class = expit(edited_logit) >= threshold
    count200 = np.asarray([row["original_tweet_count"] == 200 for row in source_rows])
    eligible = np.asarray([row["original_tweet_count"] > 0 for row in source_rows])
    result = {
        "scope": "post hoc frozen compact20 source sensitivity on official TwiBot-20 test",
        "operation": "remove first sampled tweet in memory; re-extract six retained tweet statistics",
        "interpretation": "controlled observation-record edit, not a natural error correction",
        "source_archive_sha256": compact["source_archive_sha256"],
        "base_facts_sha256": compact["base_facts_sha256"],
        "reference_artifact_sha256": compact["reference_artifact_sha256"],
        "compact_certificate_sha256": sha256(args.compact),
        "frozen_original_logit_replay_max_error": replay_error,
        "source_edit_score_delta_reconstruction_max_error": delta_error,
        "test_accounts": len(test),
        "eligible_nonempty_lists": int(np.count_nonzero(eligible)),
        "original_errors": int(np.count_nonzero(original_class != labels[test])),
        "changed_score_accounts": int(np.count_nonzero(np.abs(edited_logit - original_logit) > 1e-12)),
        "changed_class_accounts": int(np.count_nonzero(edited_class != original_class)),
        "changed_class_from_200_tweet_lists": int(np.count_nonzero(
            (edited_class != original_class) & count200)),
        "edited_errors": int(np.count_nonzero(edited_class != labels[test])),
        "correct_to_wrong": int(np.count_nonzero((original_class == labels[test])
                                                   & (edited_class != labels[test]))),
        "wrong_to_correct": int(np.count_nonzero((original_class != labels[test])
                                                   & (edited_class == labels[test]))),
        "rows": [{**source_rows[k], "label_for_evaluation": int(labels[i]),
                  "original_logit": float(original_logit[k]),
                  "first_removed_logit": float(edited_logit[k]),
                  "original_class": int(original_class[k]),
                  "first_removed_class": int(edited_class[k]),
                  "tweet_fact_deltas": {name: float(effect[k]) for name, effect in field_delta.items()}}
                 for k, i in enumerate(test)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
