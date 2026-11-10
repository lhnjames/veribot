"""Label-free, source-addressable structural facts from MGTAB typed edges."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .mgtab_source_certificate import RELATIONS, sha256


def extract(edge_index: np.ndarray, edge_type: np.ndarray, n_accounts: int
            ) -> tuple[np.ndarray, list[str], dict]:
    edge = np.asarray(edge_index, dtype=np.int64)
    relation_type = np.asarray(edge_type, dtype=np.int64)
    if edge.ndim != 2 or edge.shape[0] != 2 or edge.shape[1] != len(relation_type):
        raise ValueError("typed edge dimensions differ")
    if n_accounts < 1 or np.any(edge < 0) or np.any(edge >= n_accounts):
        raise ValueError("edge endpoint outside account population")
    if not np.array_equal(np.unique(relation_type), np.arange(len(RELATIONS))):
        raise ValueError("relation type vocabulary differs from published MGTAB")

    columns = []
    names = []
    summary = {}
    for index, relation in enumerate(RELATIONS):
        src, dst = edge[:, relation_type == index]
        self_mask = src == dst
        self_rows = np.bincount(src[self_mask], minlength=n_accounts)
        keys = np.unique(src[~self_mask] * n_accounts + dst[~self_mask])
        unique_src, unique_dst = keys // n_accounts, keys % n_accounts
        out_unique = np.bincount(unique_src, minlength=n_accounts)
        in_unique = np.bincount(unique_dst, minlength=n_accounts)
        reverse_keys = unique_dst * n_accounts + unique_src
        reciprocal = np.isin(keys, reverse_keys, assume_unique=True)
        reciprocal_unique = np.bincount(unique_src[reciprocal],
                                        minlength=n_accounts)
        for suffix, count in (
            ("nonself_out_unique_log1p", out_unique),
            ("nonself_in_unique_log1p", in_unique),
            ("reciprocal_unique_log1p", reciprocal_unique),
            ("self_edge_rows_log1p", self_rows),
        ):
            columns.append(np.log1p(count.astype(np.float32)).astype(np.float32))
            names.append(f"published_{relation}_{suffix}")
        summary[relation] = {
            "edge_rows": int(len(src)),
            "self_edge_rows": int(np.sum(self_mask)),
            "unique_nonself_directed_pairs": int(len(keys)),
            "unique_reciprocal_directed_pairs": int(np.sum(reciprocal)),
            "accounts_with_nonself_edge": int(np.count_nonzero(out_unique + in_unique)),
        }
    return np.stack(columns, axis=1), names, summary


def build(data_dir: Path, output: Path) -> dict:
    features_path = data_dir / "features.pt"
    edge_path = data_dir / "edge_index.pt"
    type_path = data_dir / "edge_type.pt"
    n_accounts = len(torch.load(features_path, map_location="cpu", weights_only=True))
    edge = torch.load(edge_path, map_location="cpu", weights_only=True).numpy()
    relation_type = torch.load(type_path, map_location="cpu", weights_only=True).numpy()
    values, names, summary = extract(edge, relation_type, n_accounts)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.arange(n_accounts), names=np.asarray(names),
                        values=values)
    manifest = {
        "dataset": "mgtab",
        "provenance_level": "published_typed_edge_tensor",
        "label_access": "none",
        "account_count": n_accounts,
        "feature_names": names,
        "relation_summary": summary,
        "formula": "per relation: log1p of unique non-self out/in neighbors, unique reciprocal neighbors, or self-edge rows; float32",
        "input_sha256": {path.name: sha256(path)
                         for path in (features_path, edge_path, type_path)},
        "facts_sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    return manifest
