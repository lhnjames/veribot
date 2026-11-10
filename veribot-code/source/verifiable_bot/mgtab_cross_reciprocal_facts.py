"""Label-free cross-type reciprocal-edge facts from MGTAB typed edge rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix

from .mgtab_source_certificate import RELATIONS, sha256


def extract(edge_index: np.ndarray, edge_type: np.ndarray,
            account_count: int) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    edge = np.asarray(edge_index, dtype=np.int64)
    kind = np.asarray(edge_type, dtype=np.int64)
    if (edge.ndim != 2 or edge.shape[0] != 2 or edge.shape[1] != len(kind)
            or np.any(edge < 0) or np.any(edge >= account_count)
            or not np.array_equal(np.unique(kind), np.arange(len(RELATIONS)))):
        raise ValueError("MGTAB typed edge population differs")
    directed = []
    for relation in range(len(RELATIONS)):
        keep = (kind == relation) & (edge[0] != edge[1])
        src, dst = edge[:, keep]
        adjacency = csr_matrix((np.ones(len(src), dtype=np.int32), (src, dst)),
                               shape=(account_count, account_count))
        adjacency.sum_duplicates()
        adjacency.data[:] = 1
        directed.append(adjacency)
    reverse = [matrix.transpose().tocsr() for matrix in directed]
    out_degree = [np.diff(matrix.indptr).astype(np.int32) for matrix in directed]
    in_degree = [np.diff(matrix.indptr).astype(np.int32) for matrix in reverse]
    values, supports, names = [], [], []
    coverage = {}
    for out_type, out_name in enumerate(RELATIONS):
        for in_type, in_name in enumerate(RELATIONS):
            if out_type == in_type:
                continue  # Same-type reciprocity is already in structural facts.
            overlap = np.asarray(directed[out_type].multiply(
                reverse[in_type]).sum(axis=1)).ravel().astype(np.int32)
            union = out_degree[out_type] + in_degree[in_type] - overlap
            if np.any(overlap > union) or np.any(union < 0):
                raise ValueError("Cross-type reciprocal neighbor union differs")
            name = f"published_{out_name}_out_{in_name}_in_reciprocal"
            count = np.log1p(overlap).astype(np.float32)
            ratio = np.full(account_count, np.nan, dtype=np.float32)
            present = union > 0
            ratio[present] = (overlap[present] / union[present]).astype(np.float32)
            values.extend((count, ratio))
            supports.extend((union, union))
            names.extend((name + "_unique_log1p", name + "_jaccard"))
            coverage[name] = {
                "accounts_with_both_relations": int(np.count_nonzero(
                    (out_degree[out_type] > 0) & (in_degree[in_type] > 0))),
                "accounts_with_reciprocal_overlap": int(np.count_nonzero(overlap)),
            }
    return (np.stack(values, axis=1), np.stack(supports, axis=1),
            names, {"coverage_by_ordered_relation_pair": coverage})


def build(data_dir: Path, destination: Path) -> dict:
    feature_path = data_dir / "features.pt"
    edge_path = data_dir / "edge_index.pt"
    type_path = data_dir / "edge_type.pt"
    features = torch.load(feature_path, map_location="cpu", weights_only=True)
    edge = torch.load(edge_path, map_location="cpu", weights_only=True).numpy()
    kind = torch.load(type_path, map_location="cpu", weights_only=True).numpy()
    values, supports, names, summary = extract(edge, kind, len(features))
    if (values.shape != (10199, 84) or supports.shape != values.shape
            or not np.isfinite(supports).all()
            or not np.isfinite(values[:, ::2]).all()):
        raise ValueError("Cross-type reciprocal fact size differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, ids=np.arange(len(features)),
                        names=np.asarray(names), values=values, supports=supports)
    manifest = {
        "dataset": "mgtab", "account_count": len(features),
        "fact_count": len(names), "feature_names": names,
        "source_level": "published_typed_edge_tensor",
        "source_operation": "For ordered distinct types i,j, intersect unique nonself outgoing-i neighbors with incoming-j neighbors; report log1p intersection and Jaccard over their union",
        "support_definition": "unique out-i and in-j neighbor-set union size; Jaccard missing when union is empty",
        "label_access": "none", "summary": summary,
        "input_sha256": {path.name: sha256(path) for path in
                         (feature_path, edge_path, type_path)},
        "facts_sha256": sha256(destination),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    source_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    root = Path(__file__).resolve().parents[2]
    path = root / "data/processed/mgtab_v3/cross_reciprocal_facts.npz"
    result = build(source_dir, path)
    print(result["fact_count"], result["facts_sha256"])
