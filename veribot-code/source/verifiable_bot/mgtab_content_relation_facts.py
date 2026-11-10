"""Typed-neighbor similarity of published MGTAB averaged-tweet vectors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .mgtab_source_certificate import RELATIONS, sha256


RELATION_INDICES = (0, 1, 3)
VECTOR_START = 20
VECTOR_WIDTH = 768
SUPPORT_PRIOR = 5.0


def extract(features: np.ndarray, edge_index: np.ndarray, edge_type: np.ndarray,
            *, chunk_size: int = 16_384
            ) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    raw = np.asarray(features)
    edge = np.asarray(edge_index, dtype=np.int64)
    kind = np.asarray(edge_type, dtype=np.int64)
    n = len(raw)
    if raw.ndim != 2 or raw.shape[1] < VECTOR_START + VECTOR_WIDTH or \
            not np.isfinite(raw[:, VECTOR_START:VECTOR_START + VECTOR_WIDTH]).all() or \
            edge.ndim != 2 or edge.shape[0] != 2 or edge.shape[1] != len(kind) or \
            np.any(edge < 0) or np.any(edge >= n) or \
            not np.array_equal(np.unique(kind), np.arange(len(RELATIONS))) or \
            chunk_size < 1:
        raise ValueError("published MGTAB feature or typed edge dimensions differ")
    vectors = raw[:, VECTOR_START:VECTOR_START + VECTOR_WIDTH].astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    present = norms > 0
    vectors[present] /= norms[present, None]
    values, supports, names = [], [], []
    summary = {"accounts_with_nonzero_vector": int(np.count_nonzero(present))}
    for relation_index in RELATION_INDICES:
        source, target = edge[:, kind == relation_index]
        nonself = source != target
        keys = np.unique(source[nonself] * n + target[nonself])
        source, target = keys // n, keys % n
        usable = present[source] & present[target]
        source, target = source[usable], target[usable]
        similarity = np.empty(len(source), dtype=np.float64)
        for start in range(0, len(source), chunk_size):
            stop = min(start + chunk_size, len(source))
            similarity[start:stop] = np.einsum(
                "ij,ij->i", vectors[source[start:stop]], vectors[target[start:stop]])
        similarity = np.clip(similarity, -1.0, 1.0)
        relation = RELATIONS[relation_index]
        summary[relation] = {
            "unique_nonself_directed_pairs": int(len(keys)),
            "pairs_with_two_nonzero_vectors": int(len(source)),
        }
        for direction, owner in (("out", source), ("in", target)):
            count = np.bincount(owner, minlength=n).astype(np.int32)
            total = np.bincount(owner, weights=similarity, minlength=n)
            mean = np.full(n, np.nan, dtype=np.float32)
            observed = count > 0
            mean[observed] = (total[observed] / count[observed]).astype(np.float32)
            values.append(mean)
            supports.append(count)
            names.append(f"published_{relation}_{direction}_neighbor_"
                         "tweet_embedding_cosine_mean")
    return (np.stack(values, axis=1), np.stack(supports, axis=1),
            names, summary)


def build(data_dir: Path, output: Path) -> dict:
    feature_path = data_dir / "features.pt"
    edge_path = data_dir / "edge_index.pt"
    type_path = data_dir / "edge_type.pt"
    features = torch.load(feature_path, map_location="cpu", weights_only=True).numpy()
    edge = torch.load(edge_path, map_location="cpu", weights_only=True).numpy()
    kind = torch.load(type_path, map_location="cpu", weights_only=True).numpy()
    values, supports, names, summary = extract(features, edge, kind)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.arange(len(features)),
                        names=np.asarray(names), values=values, supports=supports)
    manifest = {
        "dataset": "mgtab", "account_count": len(features),
        "feature_names": names, "relation_indices": list(RELATION_INDICES),
        "embedding_coordinates": [VECTOR_START, VECTOR_START + VECTOR_WIDTH],
        "source_level": "published_averaged_tweet_vector_and_typed_edge_tensors",
        "source_operation": "mean cosine similarity to unique non-self typed neighbors with nonzero published vectors; missing if none",
        "zero_vector_handling": "exclude pair; account concept missing without valid neighbor",
        "support_definition": "number of unique non-self directed neighbors with two nonzero vectors",
        "recommended_support_prior": SUPPORT_PRIOR,
        "raw_tweets_available": False, "label_access": "none",
        "relation_summary": summary,
        "input_sha256": {path.name: sha256(path)
                         for path in (feature_path, edge_path, type_path)},
        "facts_sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    return manifest
