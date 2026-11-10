"""Build typed, directed Fox8 reply/retweet relations from native events."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import torch


RELATIONS = ("reply", "rev_reply", "retweet", "rev_retweet")


def typed_native_edges(
    index_path: Path, account_ids: list[str],
    excluded_events: frozenset[tuple[str, str, str]] = frozenset(),
) -> tuple[dict[str, torch.Tensor], str]:
    index = {uid: i for i, uid in enumerate(account_ids)}
    if len(index) != len(account_ids):
        raise ValueError("account IDs must be unique")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if set(item["relation"] for item in payload["events"]) - {"reply", "retweet"}:
        raise ValueError("unrecognized native relation")
    pairs: dict[str, set[tuple[int, int]]] = {name: set() for name in RELATIONS}
    for item in payload["events"]:
        if (str(item["source_id"]), str(item["target_id"]),
                item["relation"]) in excluded_events:
            continue
        a = index.get(str(item["source_id"]))
        b = index.get(str(item["target_id"]))
        if a is None or b is None or a == b:
            continue
        name = item["relation"]
        pairs[name].add((a, b))
        pairs[f"rev_{name}"].add((b, a))
    result = {}
    h = sha256()
    for name in RELATIONS:
        ordered = sorted(pairs[name])
        edges = torch.tensor(ordered, dtype=torch.long).T.contiguous()
        if not ordered:
            edges = torch.empty((2, 0), dtype=torch.long)
        result[name] = edges
        h.update(name.encode())
        h.update(len(ordered).to_bytes(8, "little"))
        h.update(edges.numpy().tobytes())
    if sum(value.shape[1] for value in result.values()) == 0:
        raise ValueError("native relational graph is empty")
    return result, h.hexdigest()
