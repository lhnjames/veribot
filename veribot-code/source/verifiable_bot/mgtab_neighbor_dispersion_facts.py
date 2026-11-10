"""Label-blind neighbor profile dispersion on published MGTAB typed edges."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .mgtab_profile_schema import PROFILE_SCHEMA
from .mgtab_source_certificate import RELATIONS, sha256


RELATION_INDICES = (0, 1, 3)
PROFILE_INDICES = (13, 14)
SUPPORT_PRIOR = 5.0


def extract(features: np.ndarray, edge_index: np.ndarray,
            edge_type: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    profile = np.asarray(features, dtype=np.float64)
    edge = np.asarray(edge_index, dtype=np.int64)
    kind = np.asarray(edge_type, dtype=np.int64)
    n = len(profile)
    if (profile.ndim != 2 or profile.shape[1] < len(PROFILE_SCHEMA) or
            not np.isfinite(profile[:, PROFILE_INDICES]).all() or
            edge.shape[0] != 2 or edge.shape[1] != len(kind) or
            np.any(edge < 0) or np.any(edge >= n) or
            not np.array_equal(np.unique(kind), np.arange(len(RELATIONS)))):
        raise ValueError("published MGTAB feature or typed edge schema differs")
    values, supports, names = [], [], []
    summary = {}
    for relation_index in RELATION_INDICES:
        relation = RELATIONS[relation_index]
        src, dst = edge[:, kind == relation_index]
        keep = src != dst
        keys = np.unique(src[keep] * n + dst[keep])
        source, target = keys // n, keys % n
        relation_summary = {"unique_nonself_directed_pairs": int(len(keys))}
        for direction, owner, neighbor in (("out", source, target),
                                           ("in", target, source)):
            count = np.bincount(owner, minlength=n).astype(np.int32)
            relation_summary[f"accounts_with_{direction}_neighbors"] = int(
                np.count_nonzero(count))
            present = count > 0
            for profile_index in PROFILE_INDICES:
                dimension = profile[neighbor, profile_index]
                sum_values = np.bincount(owner, weights=dimension,
                                         minlength=n)
                sum_squares = np.bincount(owner, weights=dimension ** 2,
                                          minlength=n)
                dispersion = np.full(n, np.nan, dtype=np.float32)
                mean = sum_values[present] / count[present]
                variance = np.maximum(
                    sum_squares[present] / count[present] - mean ** 2, 0.0)
                dispersion[present] = np.sqrt(variance).astype(np.float32)
                values.append(dispersion)
                supports.append(count)
                names.append(
                    f"published_{relation}_{direction}_neighbor_"
                    f"{PROFILE_SCHEMA[profile_index][0]}_std")
        summary[relation] = relation_summary
    return (np.stack(values, axis=1), np.stack(supports, axis=1),
            names, summary)


def build(data_dir: Path, output: Path) -> dict:
    feature_path = data_dir / "features.pt"
    edge_path = data_dir / "edge_index.pt"
    type_path = data_dir / "edge_type.pt"
    features = torch.load(feature_path, map_location="cpu",
                          weights_only=True).numpy()
    edge = torch.load(edge_path, map_location="cpu",
                      weights_only=True).numpy()
    kind = torch.load(type_path, map_location="cpu",
                      weights_only=True).numpy()
    values, supports, names, summary = extract(features, edge, kind)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.arange(len(features)),
                        names=np.asarray(names), values=values,
                        supports=supports)
    manifest = {
        "dataset": "mgtab", "account_count": len(features),
        "feature_names": names,
        "relation_indices": list(RELATION_INDICES),
        "profile_indices": list(PROFILE_INDICES),
        "source_level": "published_normalized_profile_and_typed_edge_tensors",
        "source_operation": "population standard deviation of paper-named normalized profile coordinate over unique non-self directed neighbors; missing when no neighbors",
        "support_definition": "number of unique non-self directed neighbors",
        "recommended_support_prior": SUPPORT_PRIOR,
        "label_access": "none", "relation_summary": summary,
        "input_sha256": {path.name: sha256(path)
                         for path in (feature_path, edge_path, type_path)},
        "facts_sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
