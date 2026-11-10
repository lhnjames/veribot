"""List every raw source in a two-layer RelNet account computation closure."""

from __future__ import annotations

from hashlib import sha256
import json

import torch

from .fox8_two_stage_inputs import Fox8TwoStageInputs


def relnet_source_closure(
    batch: Fox8TwoStageInputs, account_id: str,
    hard_selection: dict[str, torch.Tensor], class_index: int,
) -> dict:
    """Return the selected two-hop source records used by one class score."""
    if class_index not in (0, 1):
        raise ValueError("class index must be Human=0 or Bot=1")
    target = batch.ids.index(account_id)
    selected = {name: mask[:, class_index].detach().cpu().bool().tolist()
                for name, mask in hard_selection.items()}
    directed = []
    for j, atom in enumerate(batch.relations):
        if not selected["relation"][j]:
            continue
        source = int(batch.relation_source[j])
        destination = int(batch.relation_target[j])
        directed.extend(((source, destination, j), (destination, source, j)))
    nodes = {target}
    frontier = {target}
    relation_indices = set()
    for _ in range(2):
        incoming = [(source, relation) for source, destination, relation in directed
                    if destination in frontier]
        relation_indices.update(relation for _, relation in incoming)
        frontier = {source for source, _ in incoming}
        nodes.update(frontier)
    source_records = []
    for i in sorted(nodes):
        account = batch.accounts[i]
        p0, p1 = batch.profile_offsets[i:i + 2]
        t0, t1 = batch.tweet_offsets[i:i + 2]
        for field, keep in zip(account.profile_field_ids, selected["profile"][p0:p1]):
            if not keep:
                continue
            value = account.profile[field]
            serialized = json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"))
            source_records.append({
                "kind": "profile_field", "source_id": account.source_id("profile", field),
                "raw_location": f"line={account.line_number};tweet=0;user.{field}",
                "value": value,
                "value_sha256": sha256(serialized.encode("utf-8")).hexdigest(),
            })
        for ordinal, keep in zip(account.tweet_ordinals, selected["tweet"][t0:t1]):
            if not keep:
                continue
            atom = account.tweets[ordinal]
            source_records.append({
                "kind": "tweet", "source_id": account.source_id("tweet", ordinal),
                "raw_location": f"line={atom.line_number};tweet={ordinal}",
                "tweet_id": atom.tweet_id,
                "record_sha256": atom.record_sha256,
            })
    for j in sorted(relation_indices):
        atom = batch.relations[j]
        source_records.append({
            "kind": "native_relation_event", "source_id": atom.source_id,
            "source_account_id": atom.source_account_id,
            "target_account_id": atom.target_account_id,
            "relation": atom.relation,
            "raw_locators": [{"line": line, "tweet": ordinal, "field": field}
                             for line, ordinal, field in atom.raw_locators],
        })
    counts = {kind: sum(item["kind"] == kind for item in source_records)
              for kind in ("profile_field", "tweet", "native_relation_event")}
    return {
        "account_id": account_id,
        "class_hypothesis": "Bot" if class_index == 1 else "Human",
        "relnet_layers": 2,
        "receptive_field_accounts": [batch.ids[i] for i in sorted(nodes)],
        "source_counts": counts,
        "source_records": source_records,
    }
