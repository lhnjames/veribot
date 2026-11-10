"""Rebuild the provided sample graph including support-row neighbor declarations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import zipfile

import ijson
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256


GRAPH_NAMES = ("graph_out_degree", "graph_in_degree", "graph_reciprocal_degree")


def account_edges(uid: str, neighbor: dict | None,
                  part: str, ordinal: int):
    neighbor = neighbor or {}
    for field, reverse in (("following", False), ("follower", True)):
        for position, other in enumerate(neighbor.get(field) or []):
            other = str(other)
            if other == uid:
                continue
            edge = (other, uid) if reverse else (uid, other)
            yield edge, f"Twibot-20/{part}.json[{ordinal}].neighbor.{field}[{position}]"


def graph_degrees(edges: set[tuple[str, str]], known: set[str]) -> dict[str, tuple[int, int, int]]:
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        if source in known:
            outgoing[source].add(target)
        if target in known:
            incoming[target].add(source)
    return {uid: (len(outgoing[uid]), len(incoming[uid]),
                  len(outgoing[uid] & incoming[uid])) for uid in known}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--neighbor-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    contract = json.loads(args.neighbor_contract.read_text(encoding="utf-8"))
    archive_hash = sha256(args.zip)
    if (archive_hash != source_audit["source_archive_sha256"]
            or archive_hash != contract["archive_sha256"]
            or sha256(args.facts) != source_audit["fact_cache_sha256"]):
        raise ValueError("Official archive or source facts differ")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        split = saved["split"].astype(str)
        positions = saved["positions"].astype(int)
        values = saved["values"].astype(float)
        names = saved["names"].astype(str).tolist()
    known = set(ids)
    old_edges: set[tuple[str, str]] = set()
    support_edges: set[tuple[str, str]] = set()
    support_ids: set[str] = set()
    added_declarations: dict[tuple[str, str], list[str]] = defaultdict(list)
    support_nonnull = 0
    support_declarations = 0
    with zipfile.ZipFile(args.zip) as zipped:
        for part in ("train", "dev", "test"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for ordinal, row in enumerate(ijson.items(stream, "item")):
                    for edge, _ in account_edges(str(row["ID"]), row.get("neighbor"),
                                                 part, ordinal):
                        old_edges.add(edge)
        if len(old_edges) != source_audit["distinct_directed_follow_edges"]:
            raise ValueError("Labeled-only graph differs from old source audit")
        with zipped.open("Twibot-20/support.json") as stream:
            for ordinal, row in enumerate(ijson.items(stream, "item")):
                uid = str(row["ID"])
                if uid in known or uid in support_ids:
                    raise ValueError("Support ID duplicates a known or support account")
                support_ids.add(uid)
                neighbor = row.get("neighbor")
                if neighbor is not None:
                    support_nonnull += 1
                for edge, locator in account_edges(uid, neighbor, "support", ordinal):
                    support_declarations += 1
                    support_edges.add(edge)
                    if edge[0] in known or edge[1] in known:
                        added_declarations[edge].append(locator)
                if (ordinal + 1) % 50000 == 0:
                    print(json.dumps({"support_rows_scanned": ordinal + 1,
                                      "nonnull_neighbor_rows": support_nonnull,
                                      "support_edge_declarations": support_declarations}),
                          flush=True)
    if (len(support_ids) != contract["support_rows_scanned"]
            or support_nonnull != contract["support_neighbor_nonnull_rows"]
            or support_declarations != contract["support_neighbor_edge_declarations"]):
        raise ValueError("Support scan differs from neighbor contract")
    added = support_edges - old_edges
    combined = old_edges | support_edges
    old_degrees = graph_degrees(old_edges, known)
    new_degrees = graph_degrees(combined, known)
    incident_added: dict[str, list[dict]] = defaultdict(list)
    for source, target in sorted(added):
        if source not in known and target not in known:
            continue
        item = {"source": source, "target": target,
                "support_declarations": sorted(added_declarations.get((source, target), []))}
        if source in known:
            incident_added[source].append(item)
        if target in known:
            incident_added[target].append(item)
    changed = []
    changed_by_split = {}
    for part in ("train", "dev", "test"):
        count = 0
        for i in np.flatnonzero(split == part):
            uid = str(ids[i])
            cached = tuple(int(values[i, names.index(name)]) for name in GRAPH_NAMES)
            if cached != old_degrees[uid]:
                raise ValueError(f"Cached old graph degree differs: {uid}")
            if cached == new_degrees[uid]:
                continue
            count += 1
            changed.append({"account_id": uid, "split": part,
                            "source_ordinal": int(positions[i]),
                            "old_labeled_only_degrees": dict(zip(GRAPH_NAMES, cached, strict=True)),
                            "all_provided_degrees": dict(zip(GRAPH_NAMES, new_degrees[uid], strict=True)),
                            "new_support_edge_declarations_incident": incident_added[uid]})
        changed_by_split[part] = count
    added_with_known = sum(source in known or target in known for source, target in added)
    report = {
        "scope": "TwiBot-20 officially provided sampled graph including support-row neighbor declarations",
        "source_archive_sha256": archive_hash,
        "fact_cache_sha256": sha256(args.facts),
        "source_audit_sha256": sha256(args.source_audit),
        "neighbor_contract_sha256": sha256(args.neighbor_contract),
        "labeled_only_unique_edges": len(old_edges),
        "support_nonnull_neighbor_rows": support_nonnull,
        "support_edge_declarations": support_declarations,
        "support_unique_edges": len(support_edges),
        "new_unique_edges_from_support": len(added),
        "new_unique_support_edges_incident_to_labeled_accounts": added_with_known,
        "union_unique_edges": len(combined),
        "labeled_accounts_with_degree_change_by_split": changed_by_split,
        "labeled_accounts_with_degree_change": len(changed),
        "changed_labeled_account_rows": changed,
        "interpretation": "The old cache correctly describes the union of labeled-row declarations but omitted available support-row declarations. The revised degrees describe all provided sampled declarations, not the complete Twitter follower graph. Whether this changes the frozen classifier must be checked by regenerating evidence; this certificate only audits source graph values.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "changed_labeled_account_rows"}, indent=2))


if __name__ == "__main__":
    main()
