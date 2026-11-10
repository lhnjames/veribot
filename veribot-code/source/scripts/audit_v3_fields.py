"""Count which raw evidence fields are available in the three target releases."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import pandas as pd
import torch


def audit_fox8(path: Path) -> dict:
    counts = Counter()
    account_ids = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            account = json.loads(line)
            counts["released_rows"] += 1
            uid = str(account.get("user_id"))
            if uid in account_ids:
                counts["duplicate_account_rows"] += 1
            account_ids.add(uid)
            tweets = account.get("user_tweets") or []
            counts["tweets"] += len(tweets)
            if tweets and isinstance(tweets[0].get("user"), dict):
                user = tweets[0]["user"]
                for field in ("created_at", "followers_count", "friends_count", "statuses_count",
                              "description", "verified"):
                    counts[f"profiles_with_{field}"] += int(user.get(field) is not None)
            for tweet in tweets:
                counts["tweets_with_id"] += int(tweet.get("id") is not None or tweet.get("id_str") is not None)
                counts["tweets_with_text"] += int(tweet.get("text") is not None)
                counts["tweets_with_created_at"] += int(tweet.get("created_at") is not None)
                counts["tweets_with_retweeted_status"] += int(tweet.get("retweeted_status") is not None)
                counts["tweets_with_reply_status_id"] += int(tweet.get("in_reply_to_status_id") is not None)
                entities = tweet.get("entities") or {}
                urls = entities.get("urls") or []
                counts["tweets_with_url"] += int(bool(urls))
                counts["url_objects"] += len(urls)
                counts["url_objects_with_expanded_url"] += sum(
                    item.get("expanded_url") is not None for item in urls)
                hashtags = entities.get("hashtags") or []
                counts["tweets_with_hashtag"] += int(bool(hashtags))
    counts["unique_accounts"] = len(account_ids)
    return dict(counts)


def audit_botsim(users_path: Path, activity_path: Path) -> dict:
    users = pd.read_csv(users_path)
    counts = Counter({"released_rows": len(users),
                      "unique_accounts": users["user_id"].astype(str).nunique()})
    for field in ("description", "submission_num", "comment_num", "comment_num_1",
                  "comment_num_2", "subreddit", "character_setting"):
        counts[f"users_with_{field}"] = int(users[field].notna().sum())
    with activity_path.open("r", encoding="utf-8") as stream:
        activity = json.load(stream)
    counts["activity_accounts"] = len(activity)
    for account in activity.values():
        for kind in ("posts", "comment_1", "comment_2"):
            records = account.get(kind, [])
            counts[kind] += len(records)
            for record in records:
                counts[f"{kind}_with_created_utc"] += int(record.get("created_utc") is not None)
                counts[f"{kind}_with_subreddit"] += int(record.get("subreddit") is not None)
                counts[f"{kind}_with_id"] += int(
                    record.get("submission_id") is not None or record.get("comment_id") is not None)
                if kind.startswith("comment"):
                    counts[f"{kind}_with_parent_id"] += int(record.get("parent_id") is not None)
                    counts[f"{kind}_with_link_id"] += int(record.get("link_id") is not None)
    return dict(counts)


def audit_mgtab(path: Path) -> dict:
    load = lambda name: torch.load(path / name, map_location="cpu", weights_only=True)
    features = load("features.pt")
    edge = load("edge_index.pt")
    relation = load("edge_type.pt")
    labels = load("labels_bot.pt")
    values, counts = torch.unique(relation, return_counts=True)
    return {
        "accounts": int(features.shape[0]), "feature_columns": int(features.shape[1]),
        "published_profile_feature_columns": 20,
        "published_averaged_tweet_embedding_columns": int(features.shape[1] - 20),
        "edges": int(edge.shape[1]),
        "edge_counts_by_published_type_index": {str(int(v)): int(c) for v, c in zip(values, counts)},
        "label_rows": int(labels.numel()),
        "raw_tweets_in_local_release": False,
        "raw_profile_records_in_local_release": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fox8", type=Path, required=True)
    parser.add_argument("--botsim-users", type=Path, required=True)
    parser.add_argument("--botsim-activity", type=Path, required=True)
    parser.add_argument("--mgtab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"fox8": audit_fox8(args.fox8),
              "botsim": audit_botsim(args.botsim_users, args.botsim_activity),
              "mgtab": audit_mgtab(args.mgtab)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
