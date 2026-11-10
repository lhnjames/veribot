"""Source-addressable BotSim account facts from comment-to-parent replies."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


FACT_NAMES = (
    "native_reply_out_unique_log1p",
    "native_reply_in_unique_log1p",
    "native_reply_mutual_unique_log1p",
    "native_reply_out_comments_log1p",
    "native_reply_in_comments_log1p",
)


def _object_id(value: object) -> str:
    return str(value).split("_", 1)[-1]


def extract(users: list[dict], activity: dict) -> tuple[list[str], np.ndarray, list[dict], dict]:
    """Resolve only published parent objects; labels and character settings are unused."""
    ids = [str(user["user_id"]) for user in users]
    by_id = {str(user["user_id"]): user for user in users}
    if len(by_id) != len(ids) or set(activity) != set(ids):
        raise ValueError("Users and activity IDs must be unique and identical")
    object_author: dict[str, str] = {}
    duplicate_same_author = 0
    for uid, record in activity.items():
        for kind, id_key, author_key in (
            ("posts", "submission_id", "author_name"),
            ("comment_1", "comment_id", "comment_author_name"),
            ("comment_2", "comment_id", "comment_author_name"),
        ):
            for item in record.get(kind, []):
                if item.get(author_key) != by_id[uid]["name"]:
                    raise ValueError("activity author differs from Users.csv")
                oid = _object_id(item[id_key])
                if oid in object_author:
                    if object_author[oid] != uid:
                        raise ValueError("object ID has conflicting authors")
                    duplicate_same_author += 1
                object_author[oid] = uid
    out_neighbors: dict[str, set[str]] = defaultdict(set)
    in_neighbors: dict[str, set[str]] = defaultdict(set)
    out_count: Counter = Counter()
    in_count: Counter = Counter()
    trace: list[dict] = []
    total_comments = known_parent = 0
    for uid, record in activity.items():
        for kind in ("comment_1", "comment_2"):
            for item in record.get(kind, []):
                total_comments += 1
                parent = _object_id(item.get("parent_id") or "")
                target = object_author.get(parent)
                if target is None:
                    continue
                known_parent += 1
                if target == uid:
                    continue
                out_neighbors[uid].add(target)
                in_neighbors[target].add(uid)
                out_count[uid] += 1
                in_count[target] += 1
                trace.append({
                    "source_account_id": uid,
                    "target_account_id": target,
                    "comment_id": str(item["comment_id"]),
                    "parent_id": str(item["parent_id"]),
                    "activity_kind": kind,
                })
    values = np.asarray([
        [len(out_neighbors[uid]), len(in_neighbors[uid]),
         len(out_neighbors[uid] & in_neighbors[uid]), out_count[uid], in_count[uid]]
        for uid in ids
    ], dtype=np.float32)
    values = np.log1p(values)
    edges = {(row["source_account_id"], row["target_account_id"]) for row in trace}
    touched = {uid for edge in edges for uid in edge}
    counts = {
        "accounts": len(ids), "comments": total_comments,
        "known_parent_comments": known_parent,
        "cross_account_reply_comments": len(trace),
        "unique_directed_reply_edges": len(edges),
        "accounts_touching_reply_edge": len(touched),
        "duplicate_same_author_object_id": duplicate_same_author,
    }
    return ids, values, trace, counts


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build(users_path: Path, activity_path: Path, output_dir: Path) -> dict:
    with users_path.open("r", encoding="utf-8-sig", newline="") as stream:
        users = list(csv.DictReader(stream))
    with activity_path.open("r", encoding="utf-8") as stream:
        activity = json.load(stream)
    ids, values, trace, counts = extract(users, activity)
    output_dir.mkdir(parents=True, exist_ok=True)
    facts_path = output_dir / "native_graph_facts.npz"
    np.savez_compressed(facts_path, ids=np.asarray(ids), names=np.asarray(FACT_NAMES), values=values)
    trace_path = output_dir / "reply_sources.jsonl.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in trace:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "relation": "comment_author_replies_to_parent_object_author",
        "fact_names": list(FACT_NAMES), "counts": counts,
        "input_sha256": {str(p.resolve()): _sha256(p) for p in (users_path, activity_path)},
        "facts_sha256": _sha256(facts_path), "reply_sources_sha256": _sha256(trace_path),
        "record_address": "user_post_comment.json[account_id][activity_kind][comment_id], parent_id",
        "excluded_fields": ["character_setting", "labels"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
