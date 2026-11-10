"""Audit which held-out sampled edge declarations enter train graph degrees."""

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
from audit_twibot20_support_edge_omission import account_edges, graph_degrees
from train_v3_twibot20 import sha256


FIELDS = ("graph_out_degree", "graph_in_degree", "graph_reciprocal_degree")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.support_audit.read_text(encoding="utf-8"))
    manifest = json.loads(args.facts.with_suffix(".json").read_text(encoding="utf-8"))
    if (sha256(args.zip) != audit["source_archive_sha256"]
            or sha256(args.facts) != manifest["fact_cache_sha256"]
            or audit["union_unique_edges"] != manifest["directed_follow_edges"]):
        raise ValueError("Official archive or corrected facts differ")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        split = saved["split"].astype(str)
        values = saved["values"].astype(float)
        positions = saved["positions"].astype(int)
        names = saved["names"].astype(str).tolist()
    train_indices = np.flatnonzero(split == "train")
    dev_indices = np.flatnonzero(split == "dev")
    train_ids = set(ids[train_indices])
    dev_ids = set(ids[dev_indices])
    by_part = {part: set() for part in ("train", "dev", "test", "support")}
    heldout_locators = defaultdict(list)
    with zipfile.ZipFile(args.zip) as zipped:
        for part in by_part:
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for ordinal, row in enumerate(ijson.items(stream, "item")):
                    uid = str(row["ID"])
                    for edge, locator in account_edges(uid, row.get("neighbor"),
                                                       part, ordinal):
                        by_part[part].add(edge)
                        if part in ("dev", "test") and (
                                edge[0] in train_ids or edge[1] in train_ids):
                            heldout_locators[edge].append(locator)
    full = set().union(*by_part.values())
    train_and_support = by_part["train"] | by_part["support"]
    if (len(full) != audit["union_unique_edges"]
            or len(by_part["train"] | by_part["dev"] | by_part["test"])
            != audit["labeled_only_unique_edges"]):
        raise ValueError("Source declaration graph size differs")
    full_degrees = graph_degrees(full, train_ids)
    limited_degrees = graph_degrees(train_and_support, train_ids)
    dev_visible_edges = train_and_support | by_part["dev"]
    full_dev_degrees = graph_degrees(full, dev_ids)
    dev_visible_degrees = graph_degrees(dev_visible_edges, dev_ids)
    heldout_only = full - train_and_support
    incident_heldout_only = {}
    for source, target in sorted(heldout_only):
        if source in train_ids or target in train_ids:
            incident_heldout_only[(source, target)] = sorted(
                heldout_locators[(source, target)])
    incident_by_account = defaultdict(list)
    incident_edges_by_declaring_split = {"dev_only": 0, "test_only": 0,
                                         "dev_and_test": 0}
    train_accounts_by_declaring_split = {"dev_only": set(), "test_only": set(),
                                         "dev_and_test": set()}
    for (source, target), locators in incident_heldout_only.items():
        declared_in_dev = any("/dev.json" in locator for locator in locators)
        declared_in_test = any("/test.json" in locator for locator in locators)
        category = ("dev_and_test" if declared_in_dev and declared_in_test
                    else "dev_only" if declared_in_dev else "test_only")
        incident_edges_by_declaring_split[category] += 1
        item = {"source": source, "target": target,
                "heldout_declarations": locators}
        if source in train_ids:
            incident_by_account[source].append(item)
            train_accounts_by_declaring_split[category].add(source)
        if target in train_ids:
            incident_by_account[target].append(item)
            train_accounts_by_declaring_split[category].add(target)
    changed = []
    changed_dev = []
    changed_by_field = dict.fromkeys(FIELDS, 0)
    maximum_increase = dict.fromkeys(FIELDS, 0)
    for i in train_indices:
        uid = str(ids[i])
        observed = full_degrees[uid]
        cached = tuple(int(values[i, names.index(name)]) for name in FIELDS)
        if observed != cached:
            raise ValueError(f"Corrected graph cache differs from source: {uid}")
        limited = limited_degrees[uid]
        if observed == limited:
            continue
        for j, name in enumerate(FIELDS):
            changed_by_field[name] += observed[j] != limited[j]
            maximum_increase[name] = max(maximum_increase[name],
                                         observed[j] - limited[j])
        changed.append({
            "account_id": uid, "train_ordinal": int(positions[i]),
            "full_graph_degrees": dict(zip(FIELDS, observed, strict=True)),
            "train_plus_support_declaration_degrees": dict(
                zip(FIELDS, limited, strict=True)),
            "exclusively_dev_or_test_declared_incident_edges": incident_by_account[uid],
        })
    for i in dev_indices:
        uid = str(ids[i])
        observed = full_dev_degrees[uid]
        cached = tuple(int(values[i, names.index(name)]) for name in FIELDS)
        if observed != cached:
            raise ValueError(f"Corrected dev graph cache differs from source: {uid}")
        limited = dev_visible_degrees[uid]
        if observed != limited:
            changed_dev.append({
                "account_id": uid, "dev_ordinal": int(positions[i]),
                "full_graph_degrees": dict(zip(FIELDS, observed, strict=True)),
                "train_support_dev_declaration_degrees": dict(
                    zip(FIELDS, limited, strict=True)),
            })
    report = {
        "scope": "TwiBot-20 sampled graph transductive source-visibility audit, train account features only",
        "source_archive_sha256": sha256(args.zip),
        "corrected_fact_cache_sha256": sha256(args.facts),
        "support_edge_audit_sha256": sha256(args.support_audit),
        "labels_read_from_zip": False,
        "train_accounts": len(train_indices),
        "unique_edges_by_declaring_split": {part: len(edges)
                                             for part, edges in by_part.items()},
        "all_provided_unique_edges": len(full),
        "train_plus_support_unique_edges": len(train_and_support),
        "train_support_dev_unique_edges": len(dev_visible_edges),
        "edges_exclusively_declared_in_dev_or_test": len(heldout_only),
        "such_edges_incident_to_train_accounts": len(incident_heldout_only),
        "incident_edges_by_exclusive_declaring_split": incident_edges_by_declaring_split,
        "train_accounts_by_exclusive_declaring_split": {
            key: len(ids) for key, ids in train_accounts_by_declaring_split.items()},
        "train_accounts_whose_degree_uses_dev_or_test_declarations": len(changed),
        "changed_train_accounts_by_degree_field": changed_by_field,
        "max_degree_increase_due_to_dev_or_test_declarations": maximum_increase,
        "changed_train_account_rows": changed,
        "dev_accounts_whose_degree_uses_test_declarations": len(changed_dev),
        "changed_dev_account_rows": changed_dev,
        "interpretation": "The corrected graph merges all published sampled declarations before the train-only reference and A/B classifier are fitted. Dev/test account declarations are unlabeled sources but can alter train graph facts. The resulting protocol is transductive in feature availability; it cannot be presented as an inductive future-account deployment evaluation without a separate graph-visibility split.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("changed_train_account_rows",
                                     "changed_dev_account_rows")}, indent=2))


if __name__ == "__main__":
    main()
