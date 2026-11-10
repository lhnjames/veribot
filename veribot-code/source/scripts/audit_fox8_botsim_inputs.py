"""Stream the two WWW target datasets and audit raw fields needed for new models."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import json
from pathlib import Path


def split_audit(path: Path, ids: list[str]) -> dict:
    saved = json.loads(path.read_text(encoding="utf-8"))
    assignments: dict[int, str] = {}
    id_sets = {}
    for name, part in saved["splits"].items():
        indices = part["indices"]
        if len(indices) != len(part["account_ids"]):
            raise ValueError(f"{path}: {name} index/ID length mismatch")
        for i, uid in zip(indices, part["account_ids"]):
            if ids[i] != uid or i in assignments:
                raise ValueError(f"{path}: {name} source alignment or overlap failed")
            assignments[i] = name
        id_sets[name] = set(part["account_ids"])
    if len(assignments) != len(ids):
        raise ValueError(f"{path}: not all source rows assigned")
    overlap = {f"{a}_{b}": len(id_sets[a] & id_sets[b])
               for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))}
    if any(overlap.values()):
        raise ValueError(f"{path}: account IDs overlap: {overlap}")
    return {"sizes": {name: len(part["indices"]) for name, part in saved["splits"].items()},
            "cross_split_account_id_overlap": overlap}


def parse_subreddits(value: str) -> list[str]:
    try:
        result = ast.literal_eval(value or "[]")
    except (SyntaxError, ValueError):
        return []
    return [str(item) for item in result] if isinstance(result, list) else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.root / "data/raw"
    splits = args.root / "data/splits"

    fox_ids = []
    fox_labels = Counter()
    fox_empty = Counter()
    with (raw / "fox8_23/fox8_23_dataset.ndjson").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            fox_ids.append(str(row["user_id"]))
            fox_labels[row["label"]] += 1
            tweets = row.get("user_tweets") or []
            if not tweets:
                fox_empty["no_tweets"] += 1
            profile = tweets[0].get("user") if tweets else None
            if not profile:
                fox_empty["no_profile"] += 1
            if not profile or not str(profile.get("description") or "").strip():
                fox_empty["no_description"] += 1
            if not any(str(t.get("text") or "").strip() for t in tweets):
                fox_empty["no_tweet_text"] += 1
    fox = {"rows": len(fox_ids), "unique_ids": len(set(fox_ids)),
           "labels": dict(fox_labels), "empty_fields": dict(fox_empty),
           "split": split_audit(splits / "fox8_23_seed0.json", fox_ids)}

    with (raw / "botsim_24/Users.csv").open(encoding="utf-8-sig", newline="") as stream:
        users = list(csv.DictReader(stream))
    bot_ids = [row["user_id"] for row in users]
    records = json.loads((raw / "botsim_24/user_post_comment.json").read_text(encoding="utf-8"))
    sub_lists = [parse_subreddits(row.get("subreddit") or "") for row in users]
    bot = {
        "rows": len(users), "unique_ids": len(set(bot_ids)),
        "json_records": len(records),
        "users_missing_json_record": sum(uid not in records for uid in bot_ids),
        "users_without_subreddit": sum(not items for items in sub_lists),
        "unique_subreddits": len({item for items in sub_lists for item in items}),
        "users_without_description": sum(not str(row.get("description") or "").strip()
                                         for row in users),
        "posts": sum(len((records.get(uid) or {}).get("posts") or []) for uid in bot_ids),
        "comment_1": sum(len((records.get(uid) or {}).get("comment_1") or []) for uid in bot_ids),
        "comment_2": sum(len((records.get(uid) or {}).get("comment_2") or []) for uid in bot_ids),
        "split": split_audit(splits / "botsim_24_seed0.json", bot_ids),
    }
    report = {"fox8_23": fox, "botsim_24": bot}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
