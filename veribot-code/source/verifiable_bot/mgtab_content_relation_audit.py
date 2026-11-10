"""Independent full-population audit of MGTAB relation-conditioned cosine facts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .mgtab_content_relation_facts import (RELATION_INDICES, VECTOR_START,
                                           VECTOR_WIDTH)
from .mgtab_source_certificate import RELATIONS, sha256


def recompute(features: np.ndarray, edge_index: np.ndarray, edge_type: np.ndarray,
              *, chunk_size: int = 8192) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw = np.asarray(features)
    edge = np.asarray(edge_index, dtype=np.int64)
    kind = np.asarray(edge_type, dtype=np.int64)
    n = len(raw)
    if raw.ndim != 2 or raw.shape[1] < VECTOR_START + VECTOR_WIDTH or \
            edge.ndim != 2 or edge.shape[0] != 2 or edge.shape[1] != len(kind) or \
            not np.array_equal(np.unique(kind), np.arange(len(RELATIONS))):
        raise ValueError("published tensors differ")
    vectors = raw[:, VECTOR_START:VECTOR_START + VECTOR_WIDTH].astype(np.float64)
    norms = np.sqrt(np.sum(vectors * vectors, axis=1))
    present = norms > 0
    values = np.full((n, 6), np.nan, dtype=np.float32)
    supports = np.zeros((n, 6), dtype=np.int32)
    names = []
    column = 0
    for relation_index in RELATION_INDICES:
        src, dst = edge[:, kind == relation_index]
        pairs = {(int(a), int(b)) for a, b in zip(src, dst)
                 if a != b and present[a] and present[b]}
        ordered = sorted(pairs)
        source = np.asarray([a for a, _ in ordered], dtype=np.int64)
        target = np.asarray([b for _, b in ordered], dtype=np.int64)
        similarities = np.empty(len(ordered), dtype=np.float64)
        for start in range(0, len(ordered), chunk_size):
            stop = min(start + chunk_size, len(ordered))
            a, b = source[start:stop], target[start:stop]
            numer = np.sum(vectors[a] * vectors[b], axis=1)
            similarities[start:stop] = np.clip(numer / (norms[a] * norms[b]),
                                               -1.0, 1.0)
        for direction, owner in (("out", source), ("in", target)):
            count = np.bincount(owner, minlength=n).astype(np.int32)
            total = np.bincount(owner, weights=similarities, minlength=n)
            observed = count > 0
            values[observed, column] = (total[observed] / count[observed]).astype(
                np.float32)
            supports[:, column] = count
            names.append(f"published_{RELATIONS[relation_index]}_{direction}_"
                         "neighbor_tweet_embedding_cosine_mean")
            column += 1
    return values, supports, names


def audit(data_dir: Path, cache_path: Path) -> dict:
    manifest = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256(cache_path) != manifest["facts_sha256"]:
        raise ValueError("content relation cache differs from manifest")
    for name, digest in manifest["input_sha256"].items():
        if sha256(data_dir / name) != digest:
            raise ValueError(f"published source differs: {name}")
    features = torch.load(data_dir / "features.pt", map_location="cpu",
                          weights_only=True).numpy()
    edge = torch.load(data_dir / "edge_index.pt", map_location="cpu",
                      weights_only=True).numpy()
    kind = torch.load(data_dir / "edge_type.pt", map_location="cpu",
                      weights_only=True).numpy()
    values, supports, names = recompute(features, edge, kind)
    with np.load(cache_path, allow_pickle=False) as saved:
        ids = saved["ids"].astype(int)
        cached_names = saved["names"].astype(str).tolist()
        cached_values = saved["values"].copy()
        cached_supports = saved["supports"].copy()
    if not np.array_equal(ids, np.arange(len(features))) or \
            cached_names != names or cached_names != manifest["feature_names"]:
        raise ValueError("content relation population or vocabulary differs")
    if not np.array_equal(supports, cached_supports):
        raise ValueError("content relation supports differ")
    equal = np.isclose(values, cached_values, rtol=0, atol=1e-6,
                       equal_nan=True)
    if not np.all(equal):
        raise ValueError(f"content relation facts differ in {np.count_nonzero(~equal)} cells")
    return {
        "dataset": "mgtab", "status": "verified_all_content_relation_fact_cells",
        "provenance_level": "published_averaged_tweet_vector_and_typed_edge_tensors",
        "account_count": len(features), "fact_count": len(names),
        "fact_cells_including_missing": int(values.size),
        "nonmissing_fact_cells": int(np.count_nonzero(np.isfinite(values))),
        "support_cells": int(supports.size),
        "source_sha256": manifest["input_sha256"],
        "fact_cache_sha256": manifest["facts_sha256"],
        "label_fields_read": False, "raw_tweets_available": False,
    }
