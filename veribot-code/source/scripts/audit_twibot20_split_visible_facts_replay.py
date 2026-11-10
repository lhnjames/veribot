"""Replay all split-visible TwiBot-20 facts without opening the test member."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import zipfile

import ijson
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256
from verifiable_bot.twibot20_facts import FACT_NAMES
from verifiable_bot.twibot20_full_source_audit import direct_facts


GRAPH_NAMES = ("graph_out_degree", "graph_in_degree", "graph_reciprocal_degree")


def degree_map(edges: set[tuple[str, str]], ids: set[str]) -> dict[str, tuple[int, int, int]]:
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        if source in ids:
            outgoing[source].add(target)
        if target in ids:
            incoming[target].add(source)
    return {uid: (len(outgoing[uid]), len(incoming[uid]),
                  len(outgoing[uid] & incoming[uid])) for uid in ids}


def source_value_equal(source: float, card: float | None) -> bool:
    return (card is None and np.isnan(source)) or (
        card is not None and np.isfinite(source) and float(source) == float(card))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    archive_hash = sha256(args.zip)
    facts_hash = sha256(args.facts)
    if (cards["source_archive_sha256"] != archive_hash
            or cards["fact_cache_sha256"] != facts_hash
            or cards["graph_scope"]
            != "train/support/dev sampled neighbor declarations without test"):
        raise ValueError("Input artifacts are not the split-visible decision version")
    with np.load(args.facts, allow_pickle=False) as saved:
        names = saved["names"].astype(str).tolist()
        ids = saved["ids"].astype(str)
        split = saved["split"].astype(str)
        positions = saved["positions"].astype(int)
        contexts = saved["contexts"].astype(str)
        values = saved["values"].astype(float)
    if names != list(FACT_NAMES) or set(split) != {"train", "dev"}:
        raise ValueError("Fact vocabulary or train/dev population differs")
    index = {(str(part), int(ordinal)): i
             for i, (part, ordinal) in enumerate(zip(split, positions, strict=True))}
    if len(index) != len(ids):
        raise ValueError("Duplicate source row locator in cache")
    by_part: dict[str, set[tuple[str, str]]] = {
        part: set() for part in ("train", "dev", "support")}
    row_counts = Counter()
    direct_cells = 0
    direct_exact_diff_cells = 0
    max_direct_abs_diff = 0.0
    card_rows = cards["decision_cards"]
    dev_card_index = 0
    with zipfile.ZipFile(args.zip) as zipped:
        for part in by_part:
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for ordinal, row in enumerate(ijson.items(stream, "item")):
                    row_counts[part] += 1
                    owner = str(row["ID"])
                    if part != "support":
                        i = index.get((part, ordinal))
                        if i is None or ids[i] != owner:
                            raise ValueError(f"Source account row differs: {part}[{ordinal}]")
                        direct = direct_facts(row)
                        context = str(row.get("domain") or "unknown")
                        expected_direct = values[i, :len(direct)]
                        close = np.isclose(direct, expected_direct,
                                           rtol=0, atol=1e-10, equal_nan=True)
                        if (context != contexts[i]
                                or not np.all(close)):
                            raise ValueError(f"Direct raw facts differ: {part}[{ordinal}]")
                        exact = np.isclose(direct, expected_direct,
                                           rtol=0, atol=0, equal_nan=True)
                        direct_exact_diff_cells += int(np.count_nonzero(~exact))
                        finite = np.isfinite(direct) & np.isfinite(expected_direct)
                        if finite.any():
                            max_direct_abs_diff = max(
                                max_direct_abs_diff,
                                float(np.max(np.abs(direct[finite]
                                                    - expected_direct[finite]))))
                        direct_cells += len(direct)
                        if part == "dev":
                            if card_rows[dev_card_index]["account_id"] != owner:
                                raise ValueError("Dev card order differs")
                            dev_card_index += 1
                    neighbor = row.get("neighbor") or {}
                    for field, reverse in (("following", False), ("follower", True)):
                        for other in neighbor.get(field) or []:
                            other = str(other)
                            if other != owner:
                                by_part[part].add((other, owner) if reverse
                                                  else (owner, other))
    if (row_counts["train"] != int(np.count_nonzero(split == "train"))
            or row_counts["dev"] != int(np.count_nonzero(split == "dev"))
            or dev_card_index != len(card_rows)):
        raise ValueError("Raw source and cache populations differ")

    train_indices = np.flatnonzero(split == "train")
    dev_indices = np.flatnonzero(split == "dev")
    train_edges = by_part["train"] | by_part["support"]
    dev_edges = train_edges | by_part["dev"]
    if len(dev_edges) != cards["official_graph_edges_recomputed"]:
        raise ValueError("Visible raw edge population differs")
    train_degrees = degree_map(train_edges, set(ids[train_indices]))
    dev_degrees = degree_map(dev_edges, set(ids[dev_indices]))
    graph_cells = 0
    card_cells = 0
    for part, indices, degrees in (("train", train_indices, train_degrees),
                                   ("dev", dev_indices, dev_degrees)):
        for k, i in enumerate(indices):
            uid = str(ids[i])
            observed = degrees[uid]
            cached = tuple(int(values[i, names.index(name)]) for name in GRAPH_NAMES)
            if observed != cached:
                raise ValueError(f"Raw graph degree differs: {part} {uid}")
            graph_cells += len(GRAPH_NAMES)
            if part != "dev":
                continue
            card = card_rows[k]
            for term in card["fact_terms"]:
                name = term["fact_name"]
                if not source_value_equal(values[i, names.index(name)],
                                          term["source_value"]):
                    raise ValueError(f"Decision card source value differs: {uid} {name}")
                card_cells += 1

    report = {
        "scope": "independent train/dev source-to-fact and dev fact-to-card replay",
        "source_archive_sha256": archive_hash,
        "facts_sha256": facts_hash,
        "decision_cards_sha256": sha256(args.cards),
        "zip_members_opened": ["Twibot-20/train.json", "Twibot-20/dev.json",
                               "Twibot-20/support.json"],
        "test_member_opened": False,
        "label_field_accessed": False,
        "source_rows_by_part": dict(row_counts),
        "train_visible_unique_edges": len(train_edges),
        "dev_visible_unique_edges": len(dev_edges),
        "direct_source_fact_cells_matched": direct_cells,
        "direct_source_fact_atol": 1e-10,
        "direct_source_exact_diff_cells": direct_exact_diff_cells,
        "max_direct_source_abs_diff": max_direct_abs_diff,
        "graph_source_fact_cells_matched": graph_cells,
        "dev_card_fact_values_matched": card_cells,
        "all_value_differences": 0,
        "status": "exact_split_visible_raw_source_replay",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
