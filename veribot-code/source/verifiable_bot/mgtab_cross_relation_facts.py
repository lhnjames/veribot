"""Label-free cross-relation neighbor overlap from published MGTAB edge rows."""

from __future__ import annotations

import itertools
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
        raise ValueError("published typed edge population differs")
    adjacency = []
    counts_by_relation = {}
    for index, relation in enumerate(RELATIONS):
        src, dst = edge[:, (kind == index) & (edge[0] != edge[1])]
        matrix = csr_matrix((np.ones(len(src), dtype=np.int32), (src, dst)),
                            shape=(account_count, account_count))
        matrix.sum_duplicates()
        matrix.data[:] = 1
        adjacency.append(matrix)
        counts_by_relation[relation] = int(matrix.nnz)
    values, supports, names = [], [], []
    coverage = {}
    for direction in ("out", "in"):
        directed = (adjacency if direction == "out" else
                    [matrix.transpose().tocsr() for matrix in adjacency])
        degrees = [np.diff(matrix.indptr).astype(np.int32) for matrix in directed]
        for left, right in itertools.combinations(range(len(RELATIONS)), 2):
            intersection = np.asarray(directed[left].multiply(
                directed[right]).sum(axis=1)).ravel().astype(np.int32)
            union = degrees[left] + degrees[right] - intersection
            if np.any(intersection > union) or np.any(union < 0):
                raise ValueError("cross-relation overlap is outside neighbor union")
            base = f"published_{RELATIONS[left]}_{RELATIONS[right]}_{direction}"
            overlap = np.log1p(intersection).astype(np.float32)
            jaccard = np.full(account_count, np.nan, dtype=np.float32)
            present = union > 0
            jaccard[present] = (intersection[present] / union[present]).astype(
                np.float32)
            values.extend((overlap, jaccard))
            supports.extend((union, union))
            names.extend((f"{base}_unique_overlap_log1p",
                          f"{base}_neighbor_jaccard"))
            coverage[base] = {
                "accounts_with_any_neighbor": int(np.count_nonzero(present)),
                "accounts_with_cross_relation_overlap": int(np.count_nonzero(
                    intersection)),
            }
    return (np.stack(values, axis=1), np.stack(supports, axis=1),
            names, {"unique_nonself_edges_by_relation": counts_by_relation,
                    "coverage_by_relation_pair": coverage})


def build(data_dir: Path, destination: Path) -> dict:
    features_path = data_dir / "features.pt"
    edge_path = data_dir / "edge_index.pt"
    type_path = data_dir / "edge_type.pt"
    features = torch.load(features_path, map_location="cpu", weights_only=True)
    edge = torch.load(edge_path, map_location="cpu", weights_only=True).numpy()
    kind = torch.load(type_path, map_location="cpu", weights_only=True).numpy()
    values, supports, names, summary = extract(edge, kind, len(features))
    if values.shape != (10199, 84) or not np.isfinite(supports).all():
        raise ValueError("cross-relation fact size differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, ids=np.arange(len(features)),
                        names=np.asarray(names), values=values, supports=supports)
    manifest = {
        "dataset": "mgtab", "account_count": len(features),
        "fact_count": len(names), "feature_names": names,
        "source_level": "published_typed_edge_tensor",
        "source_operation": "unique nonself directed neighbor-set intersection and Jaccard for each pair of published relation types, both edge directions",
        "support_definition": "unique neighbor-set union size; Jaccard missing when union is empty",
        "label_access": "none", "summary": summary,
        "input_sha256": {path.name: sha256(path) for path in
                         (features_path, edge_path, type_path)},
        "facts_sha256": sha256(destination),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    source_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    root = Path(__file__).resolve().parents[2]
    output = root / "data/processed/mgtab_v3/cross_relation_facts.npz"
    result = build(source_dir, output)
    print(result["fact_count"], result["summary"]["unique_nonself_edges_by_relation"])
