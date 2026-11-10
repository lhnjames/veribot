"""Full-population source audit for BotSim native reply-graph facts."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .botsim_native_facts import FACT_NAMES
from .botsim_source_certificate import object_id, sha256


def audit_all(users_path: Path, activity_path: Path, facts_dir: Path) -> dict:
    manifest = json.loads((facts_dir / "manifest.json").read_text(encoding="utf-8"))
    facts_path = facts_dir / "native_graph_facts.npz"
    trace_path = facts_dir / "reply_sources.jsonl.gz"
    for path in (users_path, activity_path):
        if manifest["input_sha256"].get(str(path.resolve())) != sha256(path):
            raise ValueError(f"raw source differs from manifest: {path}")
    if manifest["facts_sha256"] != sha256(facts_path) or \
            manifest["reply_sources_sha256"] != sha256(trace_path):
        raise ValueError("fact cache or source trace differs from manifest")

    with users_path.open("r", encoding="utf-8-sig", newline="") as stream:
        users = list(csv.DictReader(stream))
    ids = [row["user_id"] for row in users]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate account ID in Users.csv")
    by_id = {row["user_id"]: row for row in users}
    activity = json.loads(activity_path.read_text(encoding="utf-8"))
    if set(activity) != set(ids):
        raise ValueError("raw activity and Users.csv populations differ")

    owners = {}
    for uid, record in activity.items():
        for kind, key, author_key in (
            ("posts", "submission_id", "author_name"),
            ("comment_1", "comment_id", "comment_author_name"),
            ("comment_2", "comment_id", "comment_author_name"),
        ):
            for item in record.get(kind, []):
                if item.get(author_key) != by_id[uid]["name"]:
                    raise ValueError("raw object author differs from Users.csv")
                oid = object_id(item[key])
                if oid in owners and owners[oid] != uid:
                    raise ValueError("raw object ID has conflicting authors")
                owners[oid] = uid

    expected = Counter()
    for source, record in activity.items():
        for kind in ("comment_1", "comment_2"):
            for item in record.get(kind, []):
                parent = str(item.get("parent_id") or "")
                target = owners.get(object_id(parent))
                if target is None or source == target:
                    continue
                expected[(source, target, kind, str(item["comment_id"]), parent)] += 1

    observed = Counter()
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["source_account_id"], row["target_account_id"],
                   row["activity_kind"], row["comment_id"], row["parent_id"])
            observed[key] += 1
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise ValueError(f"reply trace differs from raw comments: "
                         f"missing={sum(missing.values())}, extra={sum(extra.values())}")

    outgoing = defaultdict(set)
    incoming = defaultdict(set)
    out_count = Counter()
    in_count = Counter()
    for (source, target, _, _, _), count in expected.items():
        outgoing[source].add(target)
        incoming[target].add(source)
        out_count[source] += count
        in_count[target] += count
    raw_counts = np.asarray([
        [len(outgoing[uid]), len(incoming[uid]),
         len(outgoing[uid] & incoming[uid]), out_count[uid], in_count[uid]]
        for uid in ids
    ], dtype=np.int64)
    recomputed = np.log1p(raw_counts.astype(np.float32)).astype(np.float32)
    with np.load(facts_path, allow_pickle=False) as cached:
        if cached["ids"].astype(str).tolist() != ids or \
                cached["names"].astype(str).tolist() != list(FACT_NAMES):
            raise ValueError("fact cache account order or vocabulary differs")
        saved = cached["values"].copy()
    if recomputed.shape != saved.shape or not np.array_equal(recomputed, saved):
        raise ValueError(f"full-population source facts differ from cache: "
                         f"mismatching_cells={int(np.count_nonzero(recomputed != saved))}")
    digest = hashlib.sha256(raw_counts.tobytes()).hexdigest()
    return {
        "dataset": "botsim24",
        "status": "verified_full_population_against_raw_comments",
        "accounts": len(ids),
        "fact_names": list(FACT_NAMES),
        "verified_fact_cells": int(saved.size),
        "raw_cross_account_reply_comments": sum(expected.values()),
        "verified_trace_records": sum(observed.values()),
        "unique_directed_reply_pairs": len({(source, target) for source, target, *_ in expected}),
        "raw_counts_sha256": digest,
        "input_sha256": {"Users.csv": sha256(users_path),
                         "user_post_comment.json": sha256(activity_path)},
        "cache_sha256": sha256(facts_path),
        "trace_sha256": sha256(trace_path),
    }
