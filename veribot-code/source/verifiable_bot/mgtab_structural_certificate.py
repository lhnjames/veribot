"""Verify an MGTAB structural fact against typed edge rows and its cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .mgtab_source_certificate import RELATIONS, sha256


SUFFIXES = (
    "nonself_out_unique_log1p",
    "nonself_in_unique_log1p",
    "reciprocal_unique_log1p",
    "self_edge_rows_log1p",
)


def certificate(data_dir: Path, facts_path: Path, account_index: int,
                fact_name: str) -> dict:
    manifest = json.loads(facts_path.with_suffix(".json").read_text(encoding="utf-8"))
    for name, expected in manifest["input_sha256"].items():
        if sha256(data_dir / name) != expected:
            raise ValueError(f"published source differs from structural fact manifest: {name}")
    if sha256(facts_path) != manifest["facts_sha256"]:
        raise ValueError("structural fact cache differs from manifest")
    with np.load(facts_path, allow_pickle=False) as cached:
        ids = cached["ids"].astype(int)
        names = cached["names"].astype(str).tolist()
        if not 0 <= account_index < len(ids) or ids[account_index] != account_index:
            raise ValueError("account index absent from structural cache")
        if names != manifest["feature_names"] or fact_name not in names:
            raise ValueError("structural fact vocabulary differs")
        stored = float(cached["values"][account_index, names.index(fact_name)])
    parsed = [(index, relation, suffix)
              for index, relation in enumerate(RELATIONS)
              for suffix in SUFFIXES
              if fact_name == f"published_{relation}_{suffix}"]
    if len(parsed) != 1:
        raise ValueError("fact name is not in fixed structural vocabulary")
    relation_index, relation, suffix = parsed[0]
    edge = torch.load(data_dir / "edge_index.pt", map_location="cpu",
                      weights_only=True).numpy()
    relation_type = torch.load(data_dir / "edge_type.pt", map_location="cpu",
                               weights_only=True).numpy()
    if edge.ndim != 2 or edge.shape[0] != 2 or edge.shape[1] != len(relation_type):
        raise ValueError("published edge dimensions differ")
    selected = np.flatnonzero((relation_type == relation_index) &
                              ((edge[0] == account_index) |
                               (edge[1] == account_index)))
    source, target = edge[:, selected]
    out_neighbors = set(target[(source == account_index) &
                               (target != account_index)].tolist())
    in_neighbors = set(source[(target == account_index) &
                              (source != account_index)].tolist())
    counts = {
        "nonself_out_unique_log1p": len(out_neighbors),
        "nonself_in_unique_log1p": len(in_neighbors),
        "reciprocal_unique_log1p": len(out_neighbors & in_neighbors),
        "self_edge_rows_log1p": int(np.sum((source == account_index) &
                                           (target == account_index))),
    }
    count = counts[suffix]
    value = float(np.float32(np.log1p(np.float32(count))))
    if value != stored:
        raise ValueError("structural fact cache differs from edge-row query")
    digest = hashlib.sha256(json.dumps(selected.tolist(),
                                       separators=(",", ":")).encode()).hexdigest()
    return {
        "dataset": "mgtab", "account_index": account_index,
        "fact_name": fact_name, "relation_type": relation_index,
        "raw_count": count, "value": value, "cached_value": stored,
        "provenance_level": "published_typed_edge_tensor",
        "source_query": f"edge_type == {relation_index} and account {account_index} is an endpoint; {suffix}",
        "matching_edge_rows": int(len(selected)),
        "matching_edge_row_indices_sha256": digest,
        "matching_edge_row_indices_first_10": selected[:10].tolist(),
        "source_files_sha256": manifest["input_sha256"],
        "fact_cache_sha256": manifest["facts_sha256"],
        "status": "verified_against_published_typed_edges",
    }
