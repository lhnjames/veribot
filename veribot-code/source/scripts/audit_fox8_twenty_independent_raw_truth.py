"""Recompute every Fox8 model-used fact directly from raw JSON without adapter formulas."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.fox8_targeted_twenty_runtime import Fox8TargetedTwentyRuntime


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
FOLDER = ROOT / "output/v3_fox8_targeted_twenty_ab_group"
OUT = ROOT / "research/certificates/fox8_twenty_independent_raw_truth.json"
PROFILE_NAMES = (
    "followers_count", "friends_count", "statuses_count", "favourites_count",
    "listed_count", "description_len", "name_len", "screen_name_len",
    "has_location", "has_url", "has_description", "ff_ratio")
TWEET_NAMES = (
    "n_tweets", "retweet_ratio", "reply_ratio", "avg_tweet_len",
    "avg_favorites", "avg_retweets", "avg_score", "unique_lang_ratio")
NATIVE_NAMES = (
    "native_reply_out_tweet_count", "native_reply_in_tweet_count",
    "native_reply_out_unique_targets", "native_reply_in_unique_sources",
    "native_retweet_out_tweet_count", "native_retweet_in_tweet_count",
    "native_retweet_out_unique_targets", "native_retweet_in_unique_sources",
    "native_reply_reciprocal_neighbors", "native_retweet_reciprocal_neighbors",
    "native_reply_internal_target_fraction", "native_retweet_internal_target_fraction")
NEIGHBOR_NAME = "native_interaction_neighbor_mean_statuses_count"
NAMES = PROFILE_NAMES + TWEET_NAMES + NATIVE_NAMES + (NEIGHBOR_NAME,)


def account_facts(user: dict, tweets: list[dict]) -> dict[str, float]:
    facts: dict[str, float] = {}
    for field in PROFILE_NAMES[:5]:
        facts[field] = float(user.get(field, 0) or 0)
    for field, source in (("description_len", "description"),
                          ("name_len", "name"),
                          ("screen_name_len", "screen_name")):
        facts[field] = float(len(user.get(source, "") or ""))
    for field, source in (("has_location", "location"),
                          ("has_url", "url"),
                          ("has_description", "description")):
        facts[field] = float(bool(user.get(source, "")))
    friends = float(user.get("friends_count", 0) or 0)
    facts["ff_ratio"] = facts["followers_count"] / max(friends + 1, 1)
    count = len(tweets)
    facts["n_tweets"] = float(count)
    if not count:
        facts.update({name: 0.0 for name in TWEET_NAMES[1:]})
        return facts
    facts["retweet_ratio"] = sum(
        (tweet.get("text", "") or "").startswith("RT ")
        for tweet in tweets) / count
    facts["reply_ratio"] = sum(
        tweet.get("in_reply_to_status_id") is not None
        for tweet in tweets) / count
    lengths = [len(tweet.get("text", "") or "") for tweet in tweets]
    favorite = [tweet.get("favorite_count", 0) or 0 for tweet in tweets]
    retweet = [tweet.get("retweet_count", 0) or 0 for tweet in tweets]
    facts["avg_tweet_len"] = sum(lengths) / count
    facts["avg_favorites"] = sum(favorite) / count
    facts["avg_retweets"] = sum(retweet) / count
    facts["avg_score"] = sum(a + b for a, b in zip(favorite, retweet)) / count
    facts["unique_lang_ratio"] = len({tweet.get("lang", "und")
                                      for tweet in tweets}) / count
    return facts


def raw_truth() -> tuple[list[str], np.ndarray, int]:
    first = Fox8TargetedTwentyRuntime(ROOT, 0)
    with (first.source.data_dir / "features.csv").open(
            encoding="utf-8", newline="") as stream:
        ids = [str(row["user_id"]) for row in csv.DictReader(stream)]
    if len(ids) != 2279 or len(set(ids)) != len(ids):
        raise ValueError("Fox8 published account index differs")
    known = set(ids)
    facts: dict[str, dict[str, float]] = {}
    all_targets: Counter = Counter()
    internal_targets: Counter = Counter()
    event_count: Counter = Counter()
    locators: defaultdict = defaultdict(list)
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid not in known or uid in facts:
                continue
            tweets = row.get("user_tweets") or []
            user = tweets[0].get("user") or {} if tweets else {}
            facts[uid] = account_facts(user, tweets)
            for ordinal, tweet in enumerate(tweets):
                targets = []
                reply = (tweet.get("in_reply_to_user_id_str") or
                         tweet.get("in_reply_to_user_id"))
                if reply is not None:
                    targets.append(("reply", str(reply),
                                    "in_reply_to_user_id_str|in_reply_to_user_id"))
                retweeted = tweet.get("retweeted_status")
                if isinstance(retweeted, dict):
                    author = retweeted.get("user") or {}
                    target = author.get("id_str") or author.get("id")
                    if target is not None:
                        targets.append(("retweet", str(target),
                                        "retweeted_status.user.id_str|id"))
                for kind, target, field in targets:
                    all_targets[uid, kind] += 1
                    if target in known and target != uid:
                        internal_targets[uid, kind] += 1
                        event_count[uid, target, kind] += 1
                        locators[uid, target, kind].append({
                            "line": line_number, "tweet_index": ordinal,
                            "field": field,
                            "tweet_id": str(tweet.get("id_str") or
                                            tweet.get("id") or ordinal)})
    if set(facts) != known:
        raise ValueError("Fox8 raw and model account populations differ")
    recorded = json.loads(EVENTS.read_text(encoding="utf-8"))["events"]
    event_map = {(str(row["source_id"]), str(row["target_id"]),
                  row["relation"]): row for row in recorded}
    if (set(event_map) != set(event_count)
            or any(event_map[key]["count"] != event_count[key]
                   or event_map[key]["locators"] != locators[key]
                   for key in event_count)):
        raise ValueError("Native event index differs from independent raw parser")
    out_peers: defaultdict = defaultdict(set)
    in_peers: defaultdict = defaultdict(set)
    out_total: Counter = Counter()
    in_total: Counter = Counter()
    adjacent: defaultdict = defaultdict(set)
    for (source, target, kind), count in event_count.items():
        out_peers[source, kind].add(target)
        in_peers[target, kind].add(source)
        out_total[source, kind] += count
        in_total[target, kind] += count
        adjacent[source].add(target)
        adjacent[target].add(source)
    for uid in ids:
        row = facts[uid]
        for kind in ("reply", "retweet"):
            row[f"native_{kind}_out_tweet_count"] = float(np.log1p(
                out_total[uid, kind]))
            row[f"native_{kind}_in_tweet_count"] = float(np.log1p(
                in_total[uid, kind]))
            row[f"native_{kind}_out_unique_targets"] = float(np.log1p(
                len(out_peers[uid, kind])))
            row[f"native_{kind}_in_unique_sources"] = float(np.log1p(
                len(in_peers[uid, kind])))
            row[f"native_{kind}_reciprocal_neighbors"] = float(np.log1p(
                len(out_peers[uid, kind] & in_peers[uid, kind])))
            total = all_targets[uid, kind]
            row[f"native_{kind}_internal_target_fraction"] = (
                internal_targets[uid, kind] / total if total else np.nan)
        neighbors = adjacent[uid]
        row[NEIGHBOR_NAME] = (float(np.float32(np.mean([
            facts[peer]["statuses_count"] for peer in neighbors])))
            if neighbors else np.nan)
    result = np.asarray([[float(np.float32(facts[uid][name]))
                          if name in PROFILE_NAMES + TWEET_NAMES or name == NEIGHBOR_NAME
                          else facts[uid][name]
                          for name in NAMES] for uid in ids], dtype=float)
    return ids, result, len(event_count)


def main() -> None:
    ids, truth, event_pairs = raw_truth()
    runs = []
    for seed in (0, 1, 2):
        runtime = Fox8TargetedTwentyRuntime(ROOT, seed)
        if runtime.ids != ids or tuple(runtime.reference.names[i]
                                       for i in runtime.kept) != NAMES:
            raise ValueError("Selected Fox8 fact vocabulary differs")
        used = runtime.source.facts[:, runtime.kept]
        if not np.array_equal(np.isfinite(used), np.isfinite(truth)):
            raise ValueError("Selected Fox8 fact availability differs from raw")
        max_fact_difference = float(np.nanmax(np.abs(used - truth)))
        if max_fact_difference > 1e-6:
            raise ValueError("Selected Fox8 fact value differs from raw")
        saved_path = FOLDER / f"seed{seed}/test_predictions.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))["predictions"]
        cards = runtime.predict_many([row["account_id"] for row in saved])
        winners = 0
        max_winner_difference = max_probability_difference = 0.0
        for card, recorded in zip(cards, saved, strict=True):
            if (card["account_id"] != recorded["account_id"]
                    or card["prediction"] != recorded["prediction"]):
                raise ValueError("Fox8 raw card and frozen decision differ")
            max_probability_difference = max(max_probability_difference,
                abs(card["probability"] - recorded["probability"]))
            i = runtime.index[card["account_id"]]
            for concept in card["concepts"]:
                name = concept["winning_source_fact"]
                if name is None:
                    continue
                expected = truth[i, NAMES.index(name)]
                if not np.isfinite(expected) or concept["source_locator"] is None:
                    raise ValueError("Concept winner has no raw source")
                max_winner_difference = max(max_winner_difference,
                    abs(float(concept["winning_source_value"]) - expected))
                winners += 1
        if max_winner_difference > 1e-6 or max_probability_difference > 1e-12:
            raise ValueError("Raw winner-to-frozen-decision chain differs")
        runs.append({
            "seed": seed,
            "account_fact_cells_checked": int(truth.size),
            "max_selected_fact_difference": max_fact_difference,
            "test_accounts_checked": len(cards),
            "test_concept_slots_checked": 20 * len(cards),
            "available_winners_checked": winners,
            "max_winner_difference": max_winner_difference,
            "max_probability_difference": max_probability_difference,
            "model_sha256": runtime.record["model_sha256"],
            "test_predictions_sha256": sha256_file(saved_path),
        })
        print(runs[-1], flush=True)
    OUT.write_text(json.dumps({
        "status": "all_model_used_fox8_twenty_facts_independently_recomputed_from_raw",
        "raw_source_sha256": sha256_file(RAW),
        "native_event_index_sha256": sha256_file(EVENTS),
        "native_event_pairs_checked": event_pairs,
        "feature_preprocessing_functions_imported": False,
        "semantic_bot_behavior_validity_inferred": False,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
