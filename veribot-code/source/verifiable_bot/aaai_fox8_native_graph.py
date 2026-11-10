"""Source-indexed Fox8 reply/retweet graph for the VeriBot HGT/RelNet pair."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import torch


RELATIONS = ("reply", "rev_reply", "retweet", "rev_retweet")


def sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def native_relation_edges(
    *, raw_path: Path, event_index_path: Path, account_ids: list[str],
    excluded_events: frozenset[tuple[str, str, str]] = frozenset(),
) -> tuple[dict[str, torch.Tensor], dict]:
    """Map raw target-user pointers to typed edges shared by both branches.

    An event remains a graph edge while at least one of its raw pointers
    remains. Callers may exclude an event only after verifying all supporting
    raw pointers were edited.
    """
    payload = json.loads(event_index_path.read_text(encoding="utf-8"))
    if payload["raw_source_sha256"] != sha256_file(raw_path):
        raise ValueError("event index does not match the raw Fox8 records")
    index = {uid: position for position, uid in enumerate(account_ids)}
    if len(index) != len(account_ids):
        raise ValueError("graph accounts must have unique IDs")
    pairs: dict[str, set[tuple[int, int]]] = {name: set() for name in RELATIONS}
    event_count = 0
    pointer_count = 0
    for event in payload["events"]:
        key = (str(event["source_id"]), str(event["target_id"]), event["relation"])
        if event["relation"] not in ("reply", "retweet"):
            raise ValueError("unexpected raw relation type")
        if len(event["locators"]) != int(event["count"]):
            raise ValueError("relation pointer multiplicity differs")
        if key in excluded_events:
            continue
        source = index.get(key[0])
        target = index.get(key[1])
        if source is None or target is None or source == target:
            continue
        pairs[key[2]].add((source, target))
        pairs[f"rev_{key[2]}"].add((target, source))
        event_count += 1
        pointer_count += int(event["count"])
    edges = {}
    digest = sha256()
    for name in RELATIONS:
        ordered = sorted(pairs[name])
        tensor = (torch.tensor(ordered, dtype=torch.long).T.contiguous()
                  if ordered else torch.empty((2, 0), dtype=torch.long))
        edges[name] = tensor
        digest.update(name.encode())
        digest.update(len(ordered).to_bytes(8, "little"))
        digest.update(tensor.numpy().tobytes())
    if sum(edge.shape[1] for edge in edges.values()) == 0:
        raise ValueError("source-backed Fox8 graph is empty")
    return edges, {
        "graph_sha256": digest.hexdigest(),
        "raw_source_sha256": payload["raw_source_sha256"],
        "event_index_sha256": sha256_file(event_index_path),
        "included_event_records": event_count,
        "included_raw_pointers": pointer_count,
        "operation": "deduplicated native reply/retweet target-user edges with explicit reverse relations",
    }
