"""Requery one MGTAB neighbor mean from published edges and profiles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .mgtab_neighbor_facts import PROFILE_INDICES, RELATION_INDICES
from .mgtab_profile_schema import PROFILE_SCHEMA, PAPER_TABLE, PAPER_URL
from .mgtab_source_certificate import RELATIONS, sha256


def certificate(data_dir: Path, facts_path: Path, account_index: int,
                fact_name: str) -> dict:
    manifest = json.loads(facts_path.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256(facts_path) != manifest["facts_sha256"]:
        raise ValueError("neighbor fact cache differs from manifest")
    for name, expected in manifest["input_sha256"].items():
        if sha256(data_dir / name) != expected:
            raise ValueError(f"published MGTAB source changed: {name}")
    with np.load(facts_path, allow_pickle=False) as saved:
        ids = saved["ids"].astype(int)
        names = saved["names"].astype(str).tolist()
        if not 0 <= account_index < len(ids) or ids[account_index] != account_index or \
                names != manifest["feature_names"] or fact_name not in names:
            raise ValueError("account or fact absent from fixed neighbor vocabulary")
        position = names.index(fact_name)
        cached_value = float(saved["values"][account_index, position])
        cached_support = int(saved["supports"][account_index, position])
    parsed = [(relation_index, direction, profile_index)
              for relation_index in RELATION_INDICES
              for direction in ("out", "in")
              for profile_index in PROFILE_INDICES
              if fact_name == f"published_{RELATIONS[relation_index]}_{direction}_"
                              f"neighbor_{PROFILE_SCHEMA[profile_index][0]}_mean"]
    if len(parsed) != 1:
        raise ValueError("neighbor fact name does not match fixed vocabulary")
    relation_index, direction, profile_index = parsed[0]
    features = torch.load(data_dir / "features.pt", map_location="cpu",
                          weights_only=True).numpy()
    edge = torch.load(data_dir / "edge_index.pt", map_location="cpu",
                      weights_only=True).numpy()
    edge_type = torch.load(data_dir / "edge_type.pt", map_location="cpu",
                           weights_only=True).numpy()
    if len(features) != len(ids) or edge.shape[0] != 2 or \
            edge.shape[1] != len(edge_type):
        raise ValueError("published source population or edge dimensions differ")
    endpoint, neighbor_endpoint = (0, 1) if direction == "out" else (1, 0)
    edge_rows = np.flatnonzero((edge_type == relation_index) &
                               (edge[endpoint] == account_index) &
                               (edge[neighbor_endpoint] != account_index))
    neighbors = sorted(set(edge[neighbor_endpoint, edge_rows].astype(int).tolist()))
    support = len(neighbors)
    value = float(np.float32(np.mean(features[neighbors, profile_index],
                                      dtype=np.float64))) if support else None
    matched = (support == cached_support and
               ((support == 0 and np.isnan(cached_value)) or
                (support > 0 and value == cached_value)))
    if not matched:
        raise ValueError("neighbor mean or support differs from raw edge/profile query")
    edge_digest = hashlib.sha256(json.dumps(edge_rows.tolist(),
                                            separators=(",", ":")).encode()).hexdigest()
    return {
        "dataset": "mgtab", "account_index": account_index,
        "fact_name": fact_name, "value": value,
        "cached_value": cached_value if support else None,
        "unique_nonself_neighbors": support,
        "published_neighbor_indices_first_10": neighbors[:10],
        "matching_edge_rows": int(len(edge_rows)),
        "matching_edge_row_indices_sha256": edge_digest,
        "paper_profile_dimension": profile_index + 1,
        "paper_profile_name": PROFILE_SCHEMA[profile_index][0],
        "paper_mapping": {"source": PAPER_TABLE, "paper_url": PAPER_URL,
                          "raw_profile_value_recomputed": False},
        "provenance_level": "published_typed_edges_and_normalized_profile_tensor",
        "source_files_sha256": manifest["input_sha256"],
        "facts_sha256": manifest["facts_sha256"],
        "status": "verified_against_published_tensors",
    }
