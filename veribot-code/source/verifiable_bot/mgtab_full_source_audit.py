"""Independent population audit of MGTAB degree, structure, and neighbor facts."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .mgtab_neighbor_facts import PROFILE_INDICES, RELATION_INDICES
from .mgtab_profile_schema import PROFILE_SCHEMA
from .mgtab_source_certificate import RELATIONS, sha256


def recompute(features: np.ndarray, edge_index: np.ndarray, edge_type: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    profile = np.asarray(features, dtype=np.float32)
    edge = np.asarray(edge_index, dtype=np.int64)
    kinds = np.asarray(edge_type, dtype=np.int64)
    n = len(profile)
    if profile.ndim != 2 or profile.shape[1] < 20 or \
            edge.ndim != 2 or edge.shape[0] != 2 or edge.shape[1] != len(kinds) or \
            np.any(edge < 0) or np.any(edge >= n):
        raise ValueError("published feature or edge dimensions differ")
    if not np.array_equal(np.unique(kinds), np.arange(len(RELATIONS))):
        raise ValueError("published relation vocabulary differs")
    base = np.zeros((n, 14), dtype=np.float32)
    structural = np.zeros((n, 28), dtype=np.float32)
    neighbor = np.full((n, 24), np.nan, dtype=np.float32)
    supports = np.zeros((n, 24), dtype=np.int32)
    relation_summary = {}
    neighbor_offset = 0
    for relation_index, relation in enumerate(RELATIONS):
        src, dst = edge[:, kinds == relation_index]
        out_rows = Counter(map(int, src))
        in_rows = Counter(map(int, dst))
        self_rows = Counter(int(a) for a, b in zip(src, dst) if a == b)
        pairs = {(int(a), int(b)) for a, b in zip(src, dst) if a != b}
        outgoing: dict[int, set[int]] = defaultdict(set)
        incoming: dict[int, set[int]] = defaultdict(set)
        for source, target in pairs:
            outgoing[source].add(target)
            incoming[target].add(source)
        for account in range(n):
            counts = (out_rows[account], in_rows[account])
            base[account, 2*relation_index:2*relation_index+2] = np.log1p(
                np.asarray(counts, dtype=np.float32))
            structural_counts = (
                len(outgoing[account]), len(incoming[account]),
                len(outgoing[account] & incoming[account]), self_rows[account])
            structural[account, 4*relation_index:4*relation_index+4] = np.log1p(
                np.asarray(structural_counts, dtype=np.float32))
        if relation_index in RELATION_INDICES:
            for owner_to_neighbors in (outgoing, incoming):
                for account, neighbor_set in owner_to_neighbors.items():
                    if not neighbor_set:
                        continue
                    sorted_neighbors = sorted(neighbor_set)
                    for field_index, profile_index in enumerate(PROFILE_INDICES):
                        column = neighbor_offset + field_index
                        neighbor[account, column] = np.float32(np.mean(
                            profile[sorted_neighbors, profile_index], dtype=np.float64))
                        supports[account, column] = len(sorted_neighbors)
                neighbor_offset += len(PROFILE_INDICES)
        relation_summary[relation] = {
            "edge_rows": len(src), "self_edge_rows": sum(self_rows.values()),
            "unique_nonself_directed_pairs": len(pairs),
        }
    if neighbor_offset != neighbor.shape[1]:
        raise AssertionError("neighbor concept order differs from fixed vocabulary")
    return base, structural, neighbor, supports, relation_summary


def audit(data_dir: Path, structural_path: Path, neighbor_path: Path) -> dict:
    structural_manifest = json.loads(structural_path.with_suffix(".json").read_text(
        encoding="utf-8"))
    neighbor_manifest = json.loads(neighbor_path.with_suffix(".json").read_text(
        encoding="utf-8"))
    for path, manifest in ((structural_path, structural_manifest),
                           (neighbor_path, neighbor_manifest)):
        if sha256(path) != manifest["facts_sha256"]:
            raise ValueError("MGTAB cache differs from source manifest")
        for source_name, expected in manifest["input_sha256"].items():
            if sha256(data_dir / source_name) != expected:
                raise ValueError(f"MGTAB source differs: {source_name}")
    feature_path = data_dir / "features.pt"
    edge_path = data_dir / "edge_index.pt"
    type_path = data_dir / "edge_type.pt"
    features = torch.load(feature_path, map_location="cpu", weights_only=True).numpy()
    edge = torch.load(edge_path, map_location="cpu", weights_only=True).numpy()
    kind = torch.load(type_path, map_location="cpu", weights_only=True).numpy()
    base, structure, neighbor, supports, summary = recompute(features, edge, kind)
    with np.load(structural_path, allow_pickle=False) as cached:
        structural_ids = cached["ids"].astype(int)
        structural_names = cached["names"].astype(str).tolist()
        saved_structure = cached["values"].copy()
    with np.load(neighbor_path, allow_pickle=False) as cached:
        neighbor_ids = cached["ids"].astype(int)
        neighbor_names = cached["names"].astype(str).tolist()
        saved_neighbor = cached["values"].copy()
        saved_supports = cached["supports"].copy()
    n = len(features)
    expected_structure_names = [
        f"published_{relation}_{suffix}" for relation in RELATIONS
        for suffix in ("nonself_out_unique_log1p", "nonself_in_unique_log1p",
                       "reciprocal_unique_log1p", "self_edge_rows_log1p")]
    expected_neighbor_names = [
        f"published_{RELATIONS[relation]}_{direction}_neighbor_"
        f"{PROFILE_SCHEMA[profile_index][0]}_mean"
        for relation in RELATION_INDICES
        for direction in ("out", "in")
        for profile_index in PROFILE_INDICES]
    if not np.array_equal(structural_ids, np.arange(n)) or \
            not np.array_equal(neighbor_ids, np.arange(n)) or \
            structural_names != expected_structure_names or \
            neighbor_names != expected_neighbor_names:
        raise ValueError("MGTAB cache account order or vocabulary differs")
    if not np.array_equal(structure, saved_structure):
        raise ValueError(f"MGTAB structural facts differ: "
                         f"{int(np.count_nonzero(structure != saved_structure))} cells")
    if not np.array_equal(supports, saved_supports):
        raise ValueError("MGTAB neighbor support counts differ")
    equal_neighbor = np.isclose(neighbor, saved_neighbor, rtol=0,
                                atol=1e-7, equal_nan=True)
    if not np.all(equal_neighbor):
        raise ValueError(f"MGTAB neighbor facts differ: "
                         f"{int(np.count_nonzero(~equal_neighbor))} cells")
    # The production model's degree columns are computed by relation_degrees.
    try:
        from source.scripts.gate0_shallow_tree import relation_degrees
    except ModuleNotFoundError as exc:
        if exc.name != "source":
            raise
        from scripts.gate0_shallow_tree import relation_degrees
    model_base, model_names = relation_degrees(edge, kind, n)
    expected_base_names = [f"published_{relation}_{direction}_degree_log1p"
                           for relation in RELATIONS for direction in ("out", "in")]
    if model_names != expected_base_names or not np.array_equal(base, model_base):
        raise ValueError("MGTAB model degree facts differ from independent row counts")
    return {
        "dataset": "mgtab",
        "status": "verified_all_published_tensor_derived_fact_cells",
        "provenance_level": "published_feature_and_typed_edge_tensors",
        "raw_profile_or_tweet_records_available": False,
        "account_count": n,
        "base_relation_degree_cells": int(base.size),
        "structural_fact_cells": int(structure.size),
        "neighbor_fact_cells_including_missing": int(neighbor.size),
        "neighbor_nonmissing_cells": int(np.count_nonzero(np.isfinite(neighbor))),
        "neighbor_support_cells": int(supports.size),
        "total_checked_cells_excluding_support": int(base.size + structure.size + neighbor.size),
        "relation_summary": summary,
        "source_sha256": {name: sha256(data_dir / name)
                          for name in ("features.pt", "edge_index.pt", "edge_type.pt")},
        "structural_facts_sha256": sha256(structural_path),
        "neighbor_facts_sha256": sha256(neighbor_path),
        "label_fields_read": False,
    }
