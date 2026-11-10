"""Audit native graph edges in BotSim and TwiBot-20 without inventing kNN ties.

TwiBot-20 is read directly from its ZIP, including support IDs via streaming;
the 5.3 GB uncompressed support.json is never extracted or loaded as one object.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import ijson


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def twibot20(archive: Path) -> dict:
    members = {name: f"Twibot-20/{name}.json"
               for name in ("train", "dev", "test", "support")}
    ids_by_split: dict[str, set[str]] = {}
    with zipfile.ZipFile(archive) as zipped:
        if any(member not in zipped.namelist() for member in members.values()):
            raise ValueError("TwiBot-20 ZIP lacks an official split member")
        for split, member in members.items():
            seen: set[str] = set()
            with zipped.open(member) as stream:
                for identifier in ijson.items(stream, "item.ID"):
                    uid = str(identifier)
                    if uid in seen:
                        raise ValueError(f"duplicate account ID inside {split}: {uid}")
                    seen.add(uid)
            ids_by_split[split] = seen
        if any(ids_by_split[a] & ids_by_split[b] for i, a in enumerate(members)
               for b in list(members)[i + 1:]):
            raise ValueError("TwiBot-20 account ID appears in two official splits")
        all_ids = set.union(*ids_by_split.values())
        labeled_ids = ids_by_split["train"] | ids_by_split["dev"] | ids_by_split["test"]
        counts: Counter = Counter()
        labels = {split: Counter() for split in ("train", "dev", "test")}
        edges: set[tuple[str, str]] = set()
        accounts_with_resolved_edge: set[str] = set()
        for split in ("train", "dev", "test"):
            with zipped.open(members[split]) as stream:
                for row in ijson.items(stream, "item"):
                    uid = str(row["ID"])
                    labels[split][str(row["label"])] += 1
                    profile = row.get("profile") or {}
                    counts[f"{split}_tweet_texts"] += len(row.get("tweet") or [])
                    for field in ("followers_count", "friends_count", "statuses_count",
                                  "listed_count", "created_at", "description"):
                        counts[f"{split}_profiles_with_{field}"] += int(profile.get(field) is not None)
                    neighbor = row.get("neighbor")
                    if neighbor is None:
                        counts[f"{split}_null_neighbor"] += 1
                        continue
                    for target in neighbor.get("following") or []:
                        target = str(target)
                        counts["following_claims"] += 1
                        if target in all_ids and target != uid:
                            edges.add((uid, target))
                            accounts_with_resolved_edge.update((uid, target))
                            counts["following_resolved"] += 1
                            counts["following_target_labeled"] += int(target in labeled_ids)
                    for source in neighbor.get("follower") or []:
                        source = str(source)
                        counts["follower_claims"] += 1
                        if source in all_ids and source != uid:
                            edges.add((source, uid))
                            accounts_with_resolved_edge.update((source, uid))
                            counts["follower_resolved"] += 1
                            counts["follower_source_labeled"] += int(source in labeled_ids)
    return {
        "dataset": "twibot20", "archive": str(archive.resolve()),
        "archive_sha256": sha256(archive),
        "member_metadata": {split: {"uncompressed_bytes": zipped_info.file_size,
                                    "crc32": f"{zipped_info.CRC:08x}"}
                            for split, zipped_info in _member_info(archive, members).items()},
        "accounts_by_split": {split: len(ids) for split, ids in ids_by_split.items()},
        "label_counts": {split: dict(counter) for split, counter in labels.items()},
        "edge_claim_counts": dict(counts),
        "unique_directed_follow_edges": len(edges),
        "accounts_touching_resolved_edge": len(accounts_with_resolved_edge),
        "graph_source": "official_neighbor_following_and_follower_lists",
        "support_used_for": "endpoint_identity_only",
    }


def _member_info(archive: Path, members: dict[str, str]) -> dict:
    with zipfile.ZipFile(archive) as zipped:
        return {split: zipped.getinfo(member) for split, member in members.items()}


def botsim(users_csv: Path, activity_json: Path) -> dict:
    with users_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        users = list(csv.DictReader(stream))
    by_id = {row["user_id"]: row for row in users}
    if len(by_id) != len(users):
        raise ValueError("BotSim account IDs are not unique")
    with activity_json.open("r", encoding="utf-8") as stream:
        activity = json.load(stream)
    objects: dict[str, str] = {}
    counts: Counter = Counter()
    for uid, record in activity.items():
        if uid not in by_id:
            raise ValueError("activity account missing from Users.csv")
        for kind, identifier in (("posts", "submission_id"),
                                 ("comment_1", "comment_id"),
                                 ("comment_2", "comment_id")):
            author_key = "author_name" if kind == "posts" else "comment_author_name"
            for item in record.get(kind, []):
                if item.get(author_key) != by_id[uid]["name"]:
                    raise ValueError("BotSim record owner differs from author name")
                object_id = str(item[identifier]).split("_", 1)[-1]
                if object_id in objects:
                    if objects[object_id] != uid:
                        raise ValueError("BotSim object ID has conflicting authors")
                    counts["duplicate_same_author_object_id"] += 1
                objects[object_id] = uid
                counts[f"{kind}_objects"] += 1
    edges: set[tuple[str, str]] = set()
    involved: set[str] = set()
    for uid, record in activity.items():
        for kind in ("comment_1", "comment_2"):
            for item in record.get(kind, []):
                counts["comments"] += 1
                parent = str(item.get("parent_id") or "").split("_", 1)[-1]
                target = objects.get(parent)
                if target is None:
                    continue
                counts["known_parent_comments"] += 1
                if target != uid:
                    counts["cross_account_reply_comments"] += 1
                    edges.add((uid, target))
                    involved.update((uid, target))
    return {
        "dataset": "botsim", "accounts": len(users),
        "native_relation": "comment_author_replies_to_parent_object_author",
        "object_and_comment_counts": dict(counts),
        "unique_directed_reply_edges": len(edges),
        "accounts_touching_reply_edge": len(involved),
        "files_sha256": {str(path.resolve()): sha256(path)
                         for path in (users_csv, activity_json)},
    }


def fox8(raw_path: Path, features_csv: Path) -> dict:
    with features_csv.open("r", encoding="utf-8", newline="") as stream:
        feature_ids = {str(row["user_id"]) for row in csv.DictReader(stream)}
    counts: Counter = Counter()
    typed_pairs: set[tuple[str, str, str]] = set()
    involved: set[str] = set()
    seen: set[str] = set()
    opener = gzip.open if raw_path.suffix == ".gz" else open
    with opener(raw_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid in seen:
                counts["duplicate_account_rows_skipped"] += 1
                continue
            seen.add(uid)
            for tweet in row.get("user_tweets") or []:
                counts["tweets"] += 1
                reply = tweet.get("in_reply_to_user_id_str") or tweet.get("in_reply_to_user_id")
                if reply is not None:
                    counts["reply_targets_recorded"] += 1
                    target = str(reply)
                    if target in feature_ids and target != uid:
                        counts["internal_reply_tweets"] += 1
                        typed_pairs.add((uid, target, "reply"))
                        involved.update((uid, target))
                retweet = tweet.get("retweeted_status")
                if isinstance(retweet, dict):
                    counts["retweet_objects"] += 1
                    author = retweet.get("user") or {}
                    target = author.get("id_str") or author.get("id")
                    if target is not None and str(target) in feature_ids and str(target) != uid:
                        counts["internal_retweet_tweets"] += 1
                        typed_pairs.add((uid, str(target), "retweet"))
                        involved.update((uid, str(target)))
    if seen != feature_ids:
        raise ValueError("Fox8 raw and feature account populations differ")
    return {
        "dataset": "fox8", "accounts": len(feature_ids),
        "relation": "sampled_reply_and_retweet_targets_within_labeled_account_population",
        "counts": dict(counts),
        "unique_typed_internal_account_pairs": len(typed_pairs),
        "accounts_touching_internal_pair": len(involved),
        "follower_graph_published": False,
        "current_main_model_uses_native_interaction_edges": False,
        "files_sha256": {str(path.resolve()): sha256(path)
                         for path in (raw_path, features_csv)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("twibot20", "botsim", "fox8"), required=True)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--users", type=Path)
    parser.add_argument("--activity", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset == "twibot20":
        if args.zip is None:
            parser.error("TwiBot-20 requires --zip")
        report = twibot20(args.zip)
    elif args.dataset == "botsim":
        if args.users is None or args.activity is None:
            parser.error("BotSim requires --users and --activity")
        report = botsim(args.users, args.activity)
    else:
        if args.raw is None or args.features is None:
            parser.error("Fox8 requires --raw and --features")
        report = fox8(args.raw, args.features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
