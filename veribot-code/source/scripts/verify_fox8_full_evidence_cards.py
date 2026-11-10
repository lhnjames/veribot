"""Verify every Fox8 evidence card by re-reading raw sampled tweets once."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from verifiable_bot.fox8_adapter import FEATURE_FIELDS, profile_features, tweet_features
from verifiable_bot.fox8_native_interaction_facts import FACT_NAMES
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def raw_matrix(args: argparse.Namespace) -> tuple[list[str], np.ndarray, int]:
    ids = pd.read_csv(args.data_dir / "features.csv")["user_id"].astype(str).tolist()
    known = set(ids)
    if len(known) != len(ids):
        raise ValueError("duplicate published Fox8 account ID")
    base = {}
    all_targets = Counter()
    internal_targets = Counter()
    event_count = Counter()
    locators = defaultdict(list)
    with gzip.open(args.raw, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid in base:
                continue
            tweets = row.get("user_tweets") or []
            profile = tweets[0]["user"] if tweets else {}
            fields = {**profile_features(profile), **tweet_features(tweets)}
            base[uid] = np.asarray([float(fields[name]) for name in FEATURE_FIELDS],
                                   dtype=np.float32)
            for position, tweet in enumerate(tweets):
                targets = []
                reply = tweet.get("in_reply_to_user_id_str") or \
                    tweet.get("in_reply_to_user_id")
                if reply is not None:
                    targets.append(("reply", str(reply),
                                    "in_reply_to_user_id_str|in_reply_to_user_id"))
                retweet = tweet.get("retweeted_status")
                if isinstance(retweet, dict):
                    user = retweet.get("user") or {}
                    target = user.get("id_str") or user.get("id")
                    if target is not None:
                        targets.append(("retweet", str(target),
                                        "retweeted_status.user.id_str|id"))
                for kind, target, field in targets:
                    all_targets[uid, kind] += 1
                    if target in known and target != uid:
                        internal_targets[uid, kind] += 1
                        event_count[uid, target, kind] += 1
                        locators[uid, target, kind].append({
                            "line": line_number, "tweet_index": position,
                            "field": field,
                            "tweet_id": str(tweet.get("id_str") or
                                            tweet.get("id") or position),
                        })
    if set(base) != known:
        raise ValueError("raw and published Fox8 account populations differ")
    event_index = json.loads(args.native_facts.with_name("interaction_event_index.json")
                             .read_text(encoding="utf-8"))
    recorded = {(row["source_id"], row["target_id"], row["relation"]): row
                for row in event_index["events"]}
    if (set(recorded) != set(event_count)
            or any(recorded[key]["count"] != event_count[key]
                   or recorded[key]["locators"] != locators[key]
                   for key in event_count)):
        raise ValueError("raw tweet target locators differ from saved event index")
    out_peers = defaultdict(set)
    in_peers = defaultdict(set)
    out_events = Counter()
    in_events = Counter()
    for (source, target, kind), count in event_count.items():
        out_peers[source, kind].add(target)
        in_peers[target, kind].add(source)
        out_events[source, kind] += count
        in_events[target, kind] += count
    native = []
    for uid in ids:
        native.append([
            np.log1p(out_events[uid, "reply"]),
            np.log1p(in_events[uid, "reply"]),
            np.log1p(len(out_peers[uid, "reply"])),
            np.log1p(len(in_peers[uid, "reply"])),
            np.log1p(out_events[uid, "retweet"]),
            np.log1p(in_events[uid, "retweet"]),
            np.log1p(len(out_peers[uid, "retweet"])),
            np.log1p(len(in_peers[uid, "retweet"])),
            np.log1p(len(out_peers[uid, "reply"] & in_peers[uid, "reply"])),
            np.log1p(len(out_peers[uid, "retweet"] & in_peers[uid, "retweet"])),
            (internal_targets[uid, "reply"] / all_targets[uid, "reply"]
             if all_targets[uid, "reply"] else np.nan),
            (internal_targets[uid, "retweet"] / all_targets[uid, "retweet"]
             if all_targets[uid, "retweet"] else np.nan),
        ])
    raw_base = np.stack([base[uid] for uid in ids])
    raw_native = np.asarray(native, dtype=np.float64)
    with np.load(args.native_facts, allow_pickle=False) as cache:
        if (cache["ids"].astype(str).tolist() != ids
                or cache["names"].astype(str).tolist() != list(FACT_NAMES)
                or not np.allclose(cache["values"], raw_native, rtol=0, atol=1e-12,
                                   equal_nan=True)):
            raise ValueError("raw native interaction values differ from cached facts")
    return ids, np.column_stack((raw_base, raw_native)), len(event_count)


def verify_seed(args: argparse.Namespace, seed: int, ids: list[str],
                matrix: np.ndarray, manifest_cards: dict) -> dict:
    directory = args.model_dir / f"seed{seed}"
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recorded = json.loads((directory / "test_predictions.json")
                          .read_text(encoding="utf-8"))["predictions"]
    x, y, loaded_ids, names, split, source = load("fox8", args.data_dir, None, seed)
    if (loaded_ids != ids or source["split_sha256"] !=
            report["manifest"]["split_sha256"]
            or not np.allclose(matrix[:, :len(FEATURE_FIELDS)], x,
                               rtol=1e-5, atol=1e-5)):
        raise ValueError("raw Fox8 base facts or split differ")
    vocabulary = names + list(FACT_NAMES)
    reference = LabelBlindReference.load(directory / "reference.json.gz")
    refit = LabelBlindReference(min_context_n=reference.min_context_n,
                                support_prior=reference.support_prior).fit(
                                    matrix[split["train"]], vocabulary)
    if (refit.version != reference.version
            or reference.version != report["manifest"]["reference_version"]):
        raise ValueError("reference does not refit from raw training facts")
    reasoner = MonotoneSplineReasoner.from_manifest(report["reasoner"])
    test = split["test"]
    evidence = reference.transform(matrix[test])
    design = reasoner._design(evidence)
    per_term = np.column_stack([
        design[:, reasoner._slices[name]] @ reasoner.coef[reasoner._slices[name]]
        for name in vocabulary])
    logits = float(reasoner.intercept) + per_term.sum(axis=1)
    probabilities = expit(logits)
    path = args.cards_dir / f"seed{seed}_evidence_cards.jsonl.gz"
    if sha256_file(path) != manifest_cards["runs"][seed]["cards_sha256"]:
        raise ValueError("Fox8 card archive differs from manifest")
    group_for = {name: group for group, members in reasoner.groups.items()
                 for name in members}
    max_value = max_evidence = max_term = max_probability = 0.0
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for offset, line in enumerate(stream):
            if offset >= len(test):
                raise ValueError("too many Fox8 evidence cards")
            card = json.loads(line)
            index = int(test[offset])
            uid = ids[index]
            saved = recorded[offset]
            if (card["account_id"] != uid or card["seed"] != seed
                    or card["evaluation_label"] != int(y[index])
                    or saved["account_id"] != uid
                    or card["source_hashes"] != manifest_cards["source_hashes"]
                    or card["source_level"] !=
                    "raw_profile_sampled_tweets_and_native_targets"
                    or card["report_sha256"] != sha256_file(report_path)
                    or card["reference_version"] != reference.version
                    or abs(card["threshold"] - report["manifest"]["threshold"]) > 1e-12
                    or len(card["terms"]) != len(vocabulary)):
                raise ValueError("Fox8 card population, model or vocabulary differs")
            for j, term in enumerate(card["terms"]):
                name = vocabulary[j]
                locator = term["source_locator"]
                sl = reasoner._slices[name]
                if (term["name"] != name or term["group"] != group_for[name]
                        or locator["raw_account_key"] != uid
                        or locator["fact_name"] != name
                        or (j >= len(FEATURE_FIELDS) and
                            locator["event_index_sha256"] !=
                            manifest_cards["source_hashes"]["event_index_sha256"])
                        or term["selected_tail"] != reasoner.tail[j]
                        or term["fixed_direction"] != int(reasoner.sign[j])
                        or term["knots"] != list(reasoner.knots[name])
                        or not np.allclose(term["nonnegative_slopes"],
                                           reasoner.coef[sl], rtol=0, atol=1e-12)):
                    raise ValueError("Fox8 source locator or trained term differs")
                actual = matrix[index, j]
                if np.isfinite(actual):
                    if term["source_value"] is None:
                        raise ValueError("Fox8 finite source was marked missing")
                    max_value = max(max_value, abs(term["source_value"] - actual))
                elif term["source_value"] is not None:
                    raise ValueError("Fox8 missing source was marked finite")
                if term["available"] != bool(evidence.available[offset, j]):
                    raise ValueError("Fox8 evidence availability differs")
                max_evidence = max(max_evidence,
                                   abs(term["upper_tail_score"] - evidence.upper[offset, j]),
                                   abs(term["lower_tail_score"] - evidence.lower[offset, j]),
                                   abs(term["reliability"] - evidence.reliability[offset, j]))
                max_term = max(max_term,
                               abs(term["logit_contribution"] - per_term[offset, j]))
            groups = {group: float(sum(per_term[offset, j]
                                       for j, name in enumerate(vocabulary)
                                       if group_for[name] == group))
                      for group in reasoner.groups}
            if any(abs(card["group_contributions"][group] - groups[group]) > 1e-12
                   or abs(saved["group_contributions"][group] - groups[group]) > 1e-12
                   for group in groups):
                raise ValueError("Fox8 card or saved group contribution differs")
            max_probability = max(max_probability,
                                  abs(card["probability"] - probabilities[offset]),
                                  abs(saved["probability"] - probabilities[offset]))
            if (abs(card["intercept"] - reasoner.intercept) > 1e-12
                    or abs(card["logit"] - logits[offset]) > 1e-12
                    or card["prediction"] !=
                    int(probabilities[offset] >= report["manifest"]["threshold"])):
                raise ValueError("Fox8 card logit or decision differs")
            count += 1
    if count != len(test) or max(max_value, max_evidence, max_term,
                                 max_probability) > 1e-12:
        raise ValueError("Fox8 card count or numeric reconstruction differs")
    return {"seed": seed, "test_accounts": count,
            "fact_cells": count * len(vocabulary),
            "max_raw_fact_difference": max_value,
            "max_reference_evidence_difference": max_evidence,
            "max_term_contribution_difference": max_term,
            "max_probability_difference": max_probability,
            "reference_refit_from_raw_train": True,
            "status": "verified_all_source_to_logit_cards"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--native-facts", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cards-dir", type=Path, required=True)
    args = parser.parse_args()
    cards = json.loads((args.cards_dir / "evidence_cards_manifest.json")
                       .read_text(encoding="utf-8"))
    if (cards["dataset"] != "fox8"
            or cards["source_hashes"]["raw_sha256"] != sha256_file(args.raw)
            or cards["source_hashes"]["native_facts_sha256"] !=
            sha256_file(args.native_facts)
            or cards["source_hashes"]["event_index_sha256"] !=
            sha256_file(args.native_facts.with_name("interaction_event_index.json"))):
        raise ValueError("Fox8 card source hashes differ")
    ids, matrix, pairs = raw_matrix(args)
    result = {"dataset": "fox8", "scope": "all_test_cards_raw_source_recomputation",
              "typed_internal_pairs_recomputed": pairs,
              "runs": [verify_seed(args, seed, ids, matrix, cards)
                       for seed in (0, 1, 2)]}
    output = args.cards_dir / "verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
