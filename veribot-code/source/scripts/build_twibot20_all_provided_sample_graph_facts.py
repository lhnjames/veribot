"""Apply independently audited support-edge degree corrections to TwiBot-20 facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256


GRAPH_NAMES = ("graph_out_degree", "graph_in_degree", "graph_reciprocal_degree")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.support_audit.read_text(encoding="utf-8"))
    if sha256(args.facts) != audit["fact_cache_sha256"]:
        raise ValueError("Old fact cache differs from support-edge audit")
    with np.load(args.facts, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    names = arrays["names"].astype(str).tolist()
    ids = arrays["ids"].astype(str)
    split = arrays["split"].astype(str)
    positions = arrays["positions"].astype(int)
    original = arrays["values"].astype(float).copy()
    revised = original.copy()
    index = {uid: i for i, uid in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("Fact cache contains duplicate IDs")
    for row in audit["changed_labeled_account_rows"]:
        uid = row["account_id"]
        i = index[uid]
        if split[i] != row["split"] or positions[i] != row["source_ordinal"]:
            raise ValueError("Changed graph account locator differs")
        for name in GRAPH_NAMES:
            j = names.index(name)
            if original[i, j] != row["old_labeled_only_degrees"][name]:
                raise ValueError("Old graph degree differs from source audit")
            revised[i, j] = row["all_provided_degrees"][name]
    changed_mask = np.any(~np.isclose(original, revised, rtol=0, atol=0,
                                      equal_nan=True), axis=1)
    if int(np.count_nonzero(changed_mask)) != audit["labeled_accounts_with_degree_change"]:
        raise ValueError("Changed account count differs from source audit")
    for j, name in enumerate(names):
        if name not in GRAPH_NAMES and not np.array_equal(
                original[:, j], revised[:, j], equal_nan=True):
            raise ValueError("Non-graph source fact changed")
    arrays["values"] = revised
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    old_manifest = json.loads(args.facts.with_suffix(".json").read_text(encoding="utf-8"))
    manifest = {
        **old_manifest,
        "base_fact_cache_sha256": sha256(args.facts),
        "support_edge_audit_sha256": sha256(args.support_audit),
        "fact_cache_sha256": sha256(args.output),
        "directed_follow_edges": audit["union_unique_edges"],
        "support_edge_declarations": audit["support_edge_declarations"],
        "support_neighbor_edges_included": True,
        "graph_scope": "officially provided sampled-neighbor declarations",
        "graph_degree_query": "deduplicated union of all provided train/dev/test/support sampled neighbor declarations; support labels never used",
        "graph_degree_scope": "observed random-neighbor sample graph, not complete Twitter follower/following network",
        "graph_fields_changed": GRAPH_NAMES,
        "accounts_with_graph_degree_change": int(np.count_nonzero(changed_mask)),
        "all_other_fact_fields_unchanged": True,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()),
                      "sha256": sha256(args.output),
                      "changed_accounts": int(np.count_nonzero(changed_mask))}, indent=2))


if __name__ == "__main__":
    main()
