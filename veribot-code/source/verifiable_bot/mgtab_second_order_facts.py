"""Two-hop, typed-neighbor degree facts derived only from MGTAB edge tensors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .mgtab_source_certificate import RELATIONS, sha256


RELATION_INDICES = (0, 1, 3)
TARGET_DEGREE_SPECS = ((0, "out"), (0, "in"), (3, "out"), (3, "in"))
SUPPORT_PRIOR = 5.0


def extract(edge_index: np.ndarray, edge_type: np.ndarray, n_accounts: int
            ) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    edge = np.asarray(edge_index, dtype=np.int64)
    kind = np.asarray(edge_type, dtype=np.int64)
    if edge.ndim != 2 or edge.shape[0] != 2 or edge.shape[1] != len(kind) or \
            n_accounts < 1 or np.any(edge < 0) or np.any(edge >= n_accounts) or \
            not np.array_equal(np.unique(kind), np.arange(len(RELATIONS))):
        raise ValueError("published typed edge dimensions differ")
    target_degrees = {}
    for relation_index, direction in TARGET_DEGREE_SPECS:
        endpoint = 0 if direction == "out" else 1
        counts = np.bincount(edge[endpoint, kind == relation_index],
                             minlength=n_accounts)
        target_degrees[(relation_index, direction)] = np.log1p(
            counts.astype(np.float32)).astype(np.float32)
    values, supports, names = [], [], []
    summary = {}
    for relation_index in RELATION_INDICES:
        relation = RELATIONS[relation_index]
        src, dst = edge[:, kind == relation_index]
        nonself = src != dst
        keys = np.unique(src[nonself] * n_accounts + dst[nonself])
        source, target = keys // n_accounts, keys % n_accounts
        detail = {"unique_nonself_directed_pairs": int(len(keys))}
        for direction, owner, neighbor in (("out", source, target),
                                           ("in", target, source)):
            count = np.bincount(owner, minlength=n_accounts).astype(np.int32)
            detail[f"accounts_with_{direction}_neighbors"] = int(np.count_nonzero(count))
            for target_relation, target_direction in TARGET_DEGREE_SPECS:
                degree = target_degrees[(target_relation, target_direction)]
                sums = np.bincount(owner, weights=degree[neighbor],
                                   minlength=n_accounts)
                mean = np.full(n_accounts, np.nan, dtype=np.float32)
                present = count > 0
                mean[present] = (sums[present] / count[present]).astype(np.float32)
                values.append(mean)
                supports.append(count)
                names.append(
                    f"published_{relation}_{direction}_neighbor_"
                    f"{RELATIONS[target_relation]}_{target_direction}_degree_mean")
        summary[relation] = detail
    return (np.stack(values, axis=1), np.stack(supports, axis=1), names, summary)


def build(data_dir: Path, output: Path) -> dict:
    feature_path = data_dir / "features.pt"
    edge_path = data_dir / "edge_index.pt"
    type_path = data_dir / "edge_type.pt"
    n_accounts = len(torch.load(feature_path, map_location="cpu", weights_only=True))
    edge = torch.load(edge_path, map_location="cpu", weights_only=True).numpy()
    kind = torch.load(type_path, map_location="cpu", weights_only=True).numpy()
    values, supports, names, summary = extract(edge, kind, n_accounts)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.arange(n_accounts),
                        names=np.asarray(names), values=values, supports=supports)
    manifest = {
        "dataset": "mgtab", "account_count": n_accounts,
        "feature_names": names,
        "relation_indices": list(RELATION_INDICES),
        "target_degree_specs": [list(spec) for spec in TARGET_DEGREE_SPECS],
        "source_level": "published_typed_edge_tensor",
        "source_operation": "mean log1p target relation row degree over unique non-self typed neighbors; missing when none",
        "support_definition": "number of unique non-self directed neighbors",
        "recommended_support_prior": SUPPORT_PRIOR,
        "label_access": "none", "relation_summary": summary,
        "input_sha256": {path.name: sha256(path)
                         for path in (feature_path, edge_path, type_path)},
        "facts_sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    return manifest
