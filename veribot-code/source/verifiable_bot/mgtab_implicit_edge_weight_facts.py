"""Account facts from published weighted URL and hashtag co-occurrence edges."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .mgtab_source_certificate import sha256


IMPLICIT_RELATIONS = ((5, "url"), (6, "hashtag"))
STRONG_WEIGHT = 0.7


def extract(edge_index: np.ndarray, edge_type: np.ndarray,
            edge_weight: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray,
                                                        list[str], dict]:
    edge = np.asarray(edge_index, dtype=np.int64)
    kind = np.asarray(edge_type, dtype=np.int64).ravel()
    weight = np.asarray(edge_weight, dtype=np.float64).ravel()
    if (edge.shape != (2, len(kind)) or len(weight) != len(kind) or
            n != 10199 or np.any(edge < 0) or np.any(edge >= n) or
            not np.isfinite(weight).all() or np.any(weight <= 0) or
            np.any(weight > 1) or not np.array_equal(np.unique(kind),
                                                      np.arange(7))):
        raise ValueError("MGTAB typed weighted edge tensor differs")
    values, supports, names = [], [], []
    counts = {}
    for relation, title in IMPLICIT_RELATIONS:
        selected = kind == relation
        left, right = edge[:, selected]
        w = weight[selected]
        if np.any(left >= right) or len(np.unique(left * n + right)) != len(w):
            raise ValueError("implicit edges must be unique undirected pairs")
        owner = np.concatenate((left, right))
        incident_weight = np.concatenate((w, w))
        degree = np.bincount(owner, minlength=n).astype(np.int32)
        total = np.bincount(owner, weights=incident_weight,
                            minlength=n)
        high = np.bincount(owner, weights=(incident_weight >= STRONG_WEIGHT),
                           minlength=n)
        maximum = np.full(n, -np.inf)
        np.maximum.at(maximum, owner, incident_weight)
        mean = np.full(n, np.nan)
        active = degree > 0
        mean[active] = total[active] / degree[active]
        maximum[~active] = np.nan
        for suffix, array in (
            ("incident_weight_sum", total),
            ("incident_weight_mean", mean),
            ("incident_weight_max", maximum),
            ("incident_weight_ge_0p7_count", high),
        ):
            values.append(array.astype(np.float32))
            supports.append(degree)
            names.append(f"published_{title}_{suffix}")
        counts[title] = {"unique_undirected_pairs": len(w),
                         "accounts_with_incident_edges": int(active.sum()),
                         "minimum_weight": float(w.min()),
                         "maximum_weight": float(w.max())}
    return (np.stack(values, axis=1), np.stack(supports, axis=1),
            names, counts)


def build(data_dir: Path, output: Path) -> dict:
    paths = [data_dir / name for name in
             ("edge_index.pt", "edge_type.pt", "edge_weight.pt")]
    edge, kind, weight = [torch.load(path, map_location="cpu",
                                     weights_only=True).numpy()
                          for path in paths]
    values, supports, names, counts = extract(edge, kind, weight, 10199)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.arange(10199),
                        values=values, supports=supports,
                        names=np.asarray(names))
    result = {
        "dataset": "mgtab", "accounts": 10199,
        "source_level": "published_implicit_relation_edge_and_weight_tensors",
        "source_definition": "For each published URL or hashtag undirected edge, add its published weight to both incident accounts; summarize sum, mean, maximum, and count of weights at least 0.7. No URL/hashtag string is available in the release.",
        "support_definition": "number of incident published edges of that type",
        "names": names, "relation_summary": counts,
        "input_sha256": {path.name: sha256(path) for path in paths},
        "facts_sha256": sha256(output), "label_access": "none",
    }
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
