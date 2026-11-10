"""Verify one BotSim graph fact against raw comments, parents, and saved edge traces."""

from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .botsim_native_facts import FACT_NAMES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_id(value: object) -> str:
    return str(value).split("_", 1)[-1]


def certificate(users_path: Path, activity_path: Path, facts_dir: Path,
                account_id: str, fact_name: str) -> dict:
    if fact_name not in FACT_NAMES:
        raise ValueError("unknown BotSim native graph fact")
    manifest = json.loads((facts_dir / "manifest.json").read_text(encoding="utf-8"))
    facts_path = facts_dir / "native_graph_facts.npz"
    traces_path = facts_dir / "reply_sources.jsonl.gz"
    for path in (users_path, activity_path):
        if manifest["input_sha256"].get(str(path.resolve())) != sha256(path):
            raise ValueError("raw source differs from fact manifest")
    if manifest["facts_sha256"] != sha256(facts_path) or \
            manifest["reply_sources_sha256"] != sha256(traces_path):
        raise ValueError("fact or source trace artifact differs from manifest")
    with users_path.open("r", encoding="utf-8-sig", newline="") as stream:
        users = list(csv.DictReader(stream))
    by_id = {row["user_id"]: row for row in users}
    if len(by_id) != len(users) or account_id not in by_id:
        raise ValueError("account ID absent or duplicated in Users.csv")
    activity = json.loads(activity_path.read_text(encoding="utf-8"))
    if set(activity) != set(by_id):
        raise ValueError("activity and user populations differ")
    owners = {}
    comments = {}
    for uid, record in activity.items():
        for kind, identifier, author_field in (
            ("posts", "submission_id", "author_name"),
            ("comment_1", "comment_id", "comment_author_name"),
            ("comment_2", "comment_id", "comment_author_name"),
        ):
            for item in record.get(kind, []):
                if item.get(author_field) != by_id[uid]["name"]:
                    raise ValueError("raw object author differs from Users.csv")
                oid = object_id(item[identifier])
                if oid in owners and owners[oid] != uid:
                    raise ValueError("raw object ID has conflicting authors")
                owners[oid] = uid
                if kind != "posts":
                    comments[(uid, kind, str(item["comment_id"]))] = item
    out_neighbors, in_neighbors = set(), set()
    out_comments = in_comments = 0
    used = []
    expected = []
    for source, record in activity.items():
        for kind in ("comment_1", "comment_2"):
            for item in record.get(kind, []):
                parent = str(item.get("parent_id") or "")
                target = owners.get(object_id(parent))
                if target is None or source == target or account_id not in (source, target):
                    continue
                expected.append((source, target, kind, str(item["comment_id"]), parent))
    with gzip.open(traces_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            source, target = row["source_account_id"], row["target_account_id"]
            if account_id not in (source, target):
                continue
            item = comments.get((source, row["activity_kind"], row["comment_id"]))
            if item is None or str(item.get("parent_id")) != row["parent_id"] or \
                    owners.get(object_id(row["parent_id"])) != target or source == target:
                raise ValueError("saved reply trace does not resolve to raw comment and parent")
            used.append((source, target, row["activity_kind"], row["comment_id"], row["parent_id"]))
            if source == account_id:
                out_neighbors.add(target)
                out_comments += 1
            if target == account_id:
                in_neighbors.add(source)
                in_comments += 1
    if Counter(used) != Counter(expected):
        missing = Counter(expected) - Counter(used)
        extra = Counter(used) - Counter(expected)
        raise ValueError(
            f"saved reply trace is incomplete or duplicated for account: "
            f"missing={sum(missing.values())}, extra={sum(extra.values())}")
    counts = (len(out_neighbors), len(in_neighbors), len(out_neighbors & in_neighbors),
              out_comments, in_comments)
    selected = FACT_NAMES.index(fact_name)
    source_value = float(np.log1p(np.float32(counts[selected])))
    with np.load(facts_path, allow_pickle=False) as cached:
        ids = cached["ids"].astype(str).tolist()
        if ids != [row["user_id"] for row in users] or \
                cached["names"].astype(str).tolist() != list(FACT_NAMES):
            raise ValueError("fact cache account order or vocabulary differs")
        stored_value = float(cached["values"][ids.index(account_id), selected])
    if source_value != stored_value:
        raise ValueError("recomputed raw reply fact differs from training cache")
    source_bytes = json.dumps(sorted(used), separators=(",", ":")).encode()
    return {
        "dataset": "botsim24", "account_id": account_id, "fact_name": fact_name,
        "source_value": source_value, "cached_value": stored_value,
        "raw_count": counts[selected],
        "provenance_level": "raw_comment_id_and_parent_object_author",
        "source_query": "reply_sources.jsonl.gz rows with source_account_id or target_account_id equal to account; each row checked against raw comment and parent object",
        "source_reply_records": len(used),
        "raw_matching_reply_records": len(expected),
        "source_reply_records_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_reply_records_first_5": used[:5],
        "source_files_sha256": {"Users.csv": sha256(users_path),
                                "user_post_comment.json": sha256(activity_path),
                                "native_graph_facts.npz": sha256(facts_path),
                                "reply_sources.jsonl.gz": sha256(traces_path)},
        "status": "verified",
    }
