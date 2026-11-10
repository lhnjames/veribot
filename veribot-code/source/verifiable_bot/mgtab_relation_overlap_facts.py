"""Four label-free typed-neighbor overlap facts from published MGTAB edges."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
import torch

from .mgtab_source_certificate import RELATIONS, sha256


PAIRS = ((0, 1), (2, 3))  # followers/friends; mention/reply
NAMES = tuple(f"published_{RELATIONS[left]}_{RELATIONS[right]}_{direction}_"
              "unique_neighbor_jaccard"
              for left, right in PAIRS for direction in ("out", "in"))


def _adjacency(edge: np.ndarray, kind: np.ndarray, n: int,
               relation: int, direction: str) -> csr_matrix:
    selected = kind == relation
    source, target = edge[:, selected]
    nonself = source != target
    row, col = ((source[nonself], target[nonself]) if direction == "out"
                else (target[nonself], source[nonself]))
    matrix = csr_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(n, n))
    matrix.sum_duplicates()
    matrix.data[:] = 1
    return matrix


def extract(edge_index: np.ndarray, edge_type: np.ndarray, n: int
            ) -> tuple[np.ndarray, np.ndarray]:
    edge = np.asarray(edge_index, dtype=np.int64)
    kind = np.asarray(edge_type, dtype=np.int64)
    if (edge.ndim != 2 or edge.shape[0] != 2 or edge.shape[1] != len(kind)
            or n < 1 or np.any(edge < 0) or np.any(edge >= n)
            or not np.array_equal(np.unique(kind), np.arange(len(RELATIONS)))):
        raise ValueError("published typed edge tensor differs")
    matrices = {(relation, direction): _adjacency(edge, kind, n, relation, direction)
                for pair in PAIRS for relation in pair for direction in ("out", "in")}
    values, supports = [], []
    for left, right in PAIRS:
        for direction in ("out", "in"):
            first, second = matrices[left, direction], matrices[right, direction]
            overlap = np.asarray(first.multiply(second).sum(axis=1)).ravel().astype(int)
            union = np.asarray(first.sum(axis=1) + second.sum(axis=1)).ravel().astype(int)
            union -= overlap
            value = np.full(n, np.nan, dtype=np.float64)
            valid = union > 0
            value[valid] = overlap[valid] / union[valid]
            values.append(value)
            supports.append(union)
    return np.column_stack(values), np.column_stack(supports)


def build(data_dir: Path, output: Path) -> dict:
    files = [data_dir / name for name in ("features.pt", "edge_index.pt", "edge_type.pt")]
    n = len(torch.load(files[0], map_location="cpu", weights_only=True))
    edge = torch.load(files[1], map_location="cpu", weights_only=True).numpy()
    kind = torch.load(files[2], map_location="cpu", weights_only=True).numpy()
    values, supports = extract(edge, kind, n)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.arange(n), names=np.asarray(NAMES),
                        values=values, supports=supports)
    manifest = {
        "dataset": "mgtab", "account_count": n,
        "fact_names": NAMES,
        "definition": "Jaccard of distinct non-self outgoing/incoming neighbor ID sets under each fixed relation pair; zero union is missing",
        "relation_pairs": [[RELATIONS[a], RELATIONS[b]] for a, b in PAIRS],
        "source_level": "published_typed_edge_tensor",
        "label_access": "none", "input_sha256": {p.name: sha256(p) for p in files},
        "facts_sha256": sha256(output),
        "nonmissing_cells": int(np.isfinite(values).sum()),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    return manifest
