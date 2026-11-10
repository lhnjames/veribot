"""Independently recompute all MGTAB cross-relation overlap cells from edge rows."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_source_certificate import RELATIONS, sha256


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    cache_path = root / "data/processed/mgtab_v3/cross_relation_facts.npz"
    manifest = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    if (manifest["facts_sha256"] != sha256(cache_path)
            or manifest["label_access"] != "none"
            or manifest["fact_count"] != 84
            or any(sha256(data_dir / name) != digest
                   for name, digest in manifest["input_sha256"].items())):
        raise ValueError("cross-relation fact source binding differs")
    with np.load(cache_path, allow_pickle=False) as cache:
        ids = cache["ids"].astype(int)
        names = cache["names"].astype(str).tolist()
        values = cache["values"].astype(np.float32)
        supports = cache["supports"].astype(np.int32)
    n = len(ids)
    if (n != 10199 or not np.array_equal(ids, np.arange(n))
            or names != manifest["feature_names"]
            or values.shape != (n, 84) or supports.shape != values.shape):
        raise ValueError("cross-relation account or fact population differs")
    edge = torch.load(data_dir / "edge_index.pt", map_location="cpu",
                      weights_only=True).numpy()
    kind = torch.load(data_dir / "edge_type.pt", map_location="cpu",
                      weights_only=True).numpy()
    if edge.shape != (2, len(kind)) or np.any(edge < 0) or np.any(edge >= n):
        raise ValueError("published edge endpoints differ")
    # This replay uses Python sets, independent of the sparse-matrix builder.
    neighbor_sets = {
        "out": [[set() for _ in range(n)] for _ in RELATIONS],
        "in": [[set() for _ in range(n)] for _ in RELATIONS],
    }
    for row in range(len(kind)):
        source, target, relation = int(edge[0, row]), int(edge[1, row]), int(kind[row])
        if source == target:
            continue
        neighbor_sets["out"][relation][source].add(target)
        neighbor_sets["in"][relation][target].add(source)
    cells = 0
    max_value_difference = 0.0
    for direction_index, direction in enumerate(("out", "in")):
        for pair_index, (left, right) in enumerate(itertools.combinations(
                range(len(RELATIONS)), 2)):
            col = 42 * direction_index + 2 * pair_index
            base = f"published_{RELATIONS[left]}_{RELATIONS[right]}_{direction}"
            if names[col:col + 2] != [f"{base}_unique_overlap_log1p",
                                       f"{base}_neighbor_jaccard"]:
                raise ValueError("fact name and relation pairing differ")
            for account in range(n):
                a = neighbor_sets[direction][left][account]
                b = neighbor_sets[direction][right][account]
                intersection = len(a & b)
                union = len(a | b)
                expected_overlap = np.float32(np.log1p(intersection))
                expected_jaccard = (np.float32(intersection / union)
                                    if union else np.float32(np.nan))
                if (supports[account, col] != union
                        or supports[account, col + 1] != union
                        or np.isnan(values[account, col])
                        or (np.isnan(values[account, col + 1]) != (union == 0))):
                    raise ValueError("source support or missingness differs")
                diff = abs(float(values[account, col]) - float(expected_overlap))
                max_value_difference = max(max_value_difference, diff)
                if diff > 1e-7:
                    raise ValueError("unique overlap count differs from edge sets")
                if union:
                    diff = abs(float(values[account, col + 1]) -
                               float(expected_jaccard))
                    max_value_difference = max(max_value_difference, diff)
                    if diff > 1e-7:
                        raise ValueError("neighbor Jaccard differs from edge sets")
                cells += 2
    certificate = {
        "dataset": "mgtab", "status": "verified_all_cross_relation_fact_cells",
        "account_count": n, "fact_count": len(names), "cells_checked": cells,
        "max_value_difference": max_value_difference,
        "fact_cache_sha256": sha256(cache_path),
        "fact_manifest_sha256": sha256(cache_path.with_suffix(".json")),
        "input_sha256": manifest["input_sha256"],
        "verification_operation": "independent per-account Python neighbor sets and set intersection/union",
    }
    path = root / "research/certificates/mgtab_cross_relation_full_source_audit.json"
    path.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(certificate["cells_checked"], certificate["max_value_difference"])


if __name__ == "__main__":
    main()
