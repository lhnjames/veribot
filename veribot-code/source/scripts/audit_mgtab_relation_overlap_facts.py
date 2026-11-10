"""Independently verify every cross-relation overlap fact using edge-ID sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_relation_overlap_facts import NAMES, PAIRS
from verifiable_bot.mgtab_source_certificate import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.facts.with_suffix(".json").read_text(encoding="utf-8"))
    files = {name: args.data_dir / name
             for name in ("features.pt", "edge_index.pt", "edge_type.pt")}
    if (manifest["facts_sha256"] != sha256(args.facts)
            or manifest["input_sha256"] != {name: sha256(path)
                                              for name, path in files.items()}
            or tuple(manifest["fact_names"]) != NAMES):
        raise ValueError("cached facts or published source hashes differ")
    n = len(torch.load(files["features.pt"], map_location="cpu", weights_only=True))
    edge = torch.load(files["edge_index.pt"], map_location="cpu", weights_only=True).numpy()
    kind = torch.load(files["edge_type.pt"], map_location="cpu", weights_only=True).numpy()
    with np.load(args.facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(int)
        names = tuple(cache["names"].astype(str))
        values = cache["values"].copy()
        supports = cache["supports"].copy()
    if (n != 10199 or not np.array_equal(ids, np.arange(n)) or names != NAMES
            or values.shape != (n, 4) or supports.shape != (n, 4)):
        raise ValueError("cached account population or evidence vocabulary differs")
    max_difference = 0.0
    missing_mismatch = support_mismatch = 0
    checked = 0
    for pair_index, (left, right) in enumerate(PAIRS):
        sets = {(relation, direction): [set() for _ in range(n)]
                for relation in (left, right) for direction in ("out", "in")}
        selected = np.isin(kind, (left, right))
        for source, target, relation in zip(edge[0, selected], edge[1, selected],
                                            kind[selected], strict=True):
            a, b, r = int(source), int(target), int(relation)
            if a == b:
                continue
            sets[r, "out"][a].add(b)
            sets[r, "in"][b].add(a)
        for direction_index, direction in enumerate(("out", "in")):
            column = pair_index * 2 + direction_index
            for account in range(n):
                first = sets[left, direction][account]
                second = sets[right, direction][account]
                union = len(first | second)
                intersection = len(first & second)
                if int(supports[account, column]) != union:
                    support_mismatch += 1
                expected = intersection / union if union else float("nan")
                actual = float(values[account, column])
                if np.isnan(expected) != np.isnan(actual):
                    missing_mismatch += 1
                elif np.isfinite(expected) and np.isfinite(actual):
                    max_difference = max(max_difference, abs(expected - actual))
                checked += 1
    if support_mismatch or missing_mismatch or max_difference > 1e-12:
        raise ValueError("relation overlap values differ from independent edge-set query")
    result = {
        "dataset": "mgtab", "status": "all_relation_overlap_fact_cells_replayed",
        "source_level": "published_typed_edge_tensor",
        "labels_read": False, "accounts": n, "fact_names": NAMES,
        "fact_cells_checked": checked, "support_cells_checked": checked,
        "nonmissing_fact_cells": int(np.isfinite(values).sum()),
        "max_absolute_difference": max_difference,
        "missing_mismatch_count": missing_mismatch,
        "support_mismatch_count": support_mismatch,
        "facts_sha256": sha256(args.facts),
        "source_sha256": manifest["input_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
