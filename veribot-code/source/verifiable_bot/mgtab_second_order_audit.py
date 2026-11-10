"""Independent all-account audit of typed-neighbor degree evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .mgtab_second_order_facts import RELATION_INDICES, TARGET_DEGREE_SPECS
from .mgtab_source_certificate import RELATIONS, sha256


def recompute(edge_index: np.ndarray, edge_type: np.ndarray, n: int
              ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    edge = np.asarray(edge_index, dtype=np.int64)
    kind = np.asarray(edge_type, dtype=np.int64)
    if edge.ndim != 2 or edge.shape[0] != 2 or edge.shape[1] != len(kind) or \
            np.any(edge < 0) or np.any(edge >= n) or \
            not np.array_equal(np.unique(kind), np.arange(len(RELATIONS))):
        raise ValueError("published typed edge dimensions differ")
    degree = {}
    for relation_index, direction in TARGET_DEGREE_SPECS:
        endpoint = 0 if direction == "out" else 1
        counts = Counter(map(int, edge[endpoint, kind == relation_index]))
        degree[(relation_index, direction)] = np.log1p(
            np.asarray([counts[i] for i in range(n)], dtype=np.float32))
    values = np.full((n, 24), np.nan, dtype=np.float32)
    supports = np.zeros((n, 24), dtype=np.int32)
    names = []
    offset = 0
    for relation_index in RELATION_INDICES:
        source, target = edge[:, kind == relation_index]
        pairs = {(int(a), int(b)) for a, b in zip(source, target) if a != b}
        outgoing: dict[int, set[int]] = defaultdict(set)
        incoming: dict[int, set[int]] = defaultdict(set)
        for a, b in pairs:
            outgoing[a].add(b)
            incoming[b].add(a)
        for direction, owners in (("out", outgoing), ("in", incoming)):
            for target_relation, target_direction in TARGET_DEGREE_SPECS:
                names.append(f"published_{RELATIONS[relation_index]}_{direction}_"
                             f"neighbor_{RELATIONS[target_relation]}_"
                             f"{target_direction}_degree_mean")
                target_values = degree[(target_relation, target_direction)]
                for account, neighbors in owners.items():
                    ordered = sorted(neighbors)
                    values[account, offset] = np.float32(np.mean(
                        target_values[ordered], dtype=np.float64))
                    supports[account, offset] = len(ordered)
                offset += 1
    return values, supports, names


def audit(data_dir: Path, cache_path: Path) -> dict:
    manifest = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256(cache_path) != manifest["facts_sha256"]:
        raise ValueError("second-order cache hash differs")
    for name, expected in manifest["input_sha256"].items():
        if sha256(data_dir / name) != expected:
            raise ValueError(f"published source differs: {name}")
    n = len(torch.load(data_dir / "features.pt", map_location="cpu", weights_only=True))
    edge = torch.load(data_dir / "edge_index.pt", map_location="cpu",
                      weights_only=True).numpy()
    kind = torch.load(data_dir / "edge_type.pt", map_location="cpu",
                      weights_only=True).numpy()
    values, supports, names = recompute(edge, kind, n)
    with np.load(cache_path, allow_pickle=False) as saved:
        saved_ids = saved["ids"].astype(int)
        saved_names = saved["names"].astype(str).tolist()
        saved_values = saved["values"].copy()
        saved_supports = saved["supports"].copy()
    if not np.array_equal(saved_ids, np.arange(n)) or \
            saved_names != names or saved_names != manifest["feature_names"]:
        raise ValueError("second-order fact population or vocabulary differs")
    if not np.array_equal(supports, saved_supports):
        raise ValueError("second-order support counts differ")
    if not np.all(np.isclose(values, saved_values, rtol=0, atol=1e-7,
                             equal_nan=True)):
        raise ValueError("second-order fact values differ")
    return {
        "dataset": "mgtab", "status": "verified_all_second_order_fact_cells",
        "provenance_level": "published_typed_edge_tensor",
        "account_count": n, "fact_count": len(names),
        "fact_cells_including_missing": int(values.size),
        "nonmissing_fact_cells": int(np.count_nonzero(np.isfinite(values))),
        "support_cells": int(supports.size),
        "source_sha256": manifest["input_sha256"],
        "fact_cache_sha256": manifest["facts_sha256"],
        "label_fields_read": False,
    }
