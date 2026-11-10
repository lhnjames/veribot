"""Independent set-based spot audit of MGTAB cross-type reciprocal facts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.mgtab_source_certificate import RELATIONS


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    artifact = root / "data/processed/mgtab_v3/cross_reciprocal_facts.npz"
    manifest = json.loads(artifact.with_suffix(".json").read_text())
    if manifest["facts_sha256"] != sha256_file(artifact):
        raise ValueError("Reciprocal fact artifact differs from manifest")
    edge = torch.load(data / "edge_index.pt", map_location="cpu",
                      weights_only=True).numpy()
    kind = torch.load(data / "edge_type.pt", map_location="cpu",
                      weights_only=True).numpy()
    for name, path in (("edge_index.pt", data / "edge_index.pt"),
                       ("edge_type.pt", data / "edge_type.pt")):
        if manifest["input_sha256"][name] != sha256_file(path):
            raise ValueError("Published typed edges differ")
    with np.load(artifact, allow_pickle=False) as saved:
        ids = saved["ids"]
        names = saved["names"].astype(str).tolist()
        values = saved["values"]
        supports = saved["supports"]
    if (not np.array_equal(ids, np.arange(10199))
            or names != manifest["feature_names"]
            or values.shape != (10199, 84)):
        raise ValueError("Reciprocal fact schema differs")
    rng = np.random.default_rng(20261103)
    sampled = np.sort(rng.choice(len(ids), size=300, replace=False))
    wanted = set(map(int, sampled))
    outgoing = {u: [set() for _ in RELATIONS] for u in wanted}
    incoming = {u: [set() for _ in RELATIONS] for u in wanted}
    for (src, dst), relation in zip(edge.T, kind):
        u, v, t = int(src), int(dst), int(relation)
        if u == v:
            continue
        if u in wanted:
            outgoing[u][t].add(v)
        if v in wanted:
            incoming[v][t].add(u)
    max_error = 0.0
    checked = 0
    missing_checked = 0
    for u in sampled:
        col = 0
        for out_type in range(len(RELATIONS)):
            for in_type in range(len(RELATIONS)):
                if out_type == in_type:
                    continue
                a = outgoing[int(u)][out_type]
                b = incoming[int(u)][in_type]
                overlap = len(a & b)
                union = len(a | b)
                expected_count = np.float32(np.log1p(overlap))
                if int(supports[u, col]) != union or int(supports[u, col + 1]) != union:
                    raise ValueError("Reciprocal union support differs from source sets")
                max_error = max(max_error, abs(float(values[u, col]) -
                                               float(expected_count)))
                expected_ratio = np.float32(overlap / union) if union else np.nan
                if union:
                    max_error = max(max_error, abs(float(values[u, col + 1]) -
                                                   float(expected_ratio)))
                elif not np.isnan(values[u, col + 1]):
                    raise ValueError("Undefined Jaccard should be missing")
                else:
                    missing_checked += 1
                checked += 2
                col += 2
        if col != 84:
            raise ValueError("Reciprocal ordered-pair count differs")
    if max_error > 1e-7:
        raise ValueError("Reciprocal fact values differ from raw edge sets")
    result = {
        "dataset": "mgtab",
        "scope": "independent_set_reconstruction_of_sampled_cross_type_reciprocal_facts",
        "published_edge_index_sha256": sha256_file(data / "edge_index.pt"),
        "published_edge_type_sha256": sha256_file(data / "edge_type.pt"),
        "fact_artifact_sha256": sha256_file(artifact),
        "sample_seed": 20261103,
        "sample_account_count": len(sampled),
        "sample_account_ids": sampled.tolist(),
        "source_fact_cells_checked": checked,
        "missing_jaccard_cells_checked": missing_checked,
        "maximum_absolute_value_difference": max_error,
        "status": "sampled_source_facts_verified",
    }
    path = root / "research/certificates/mgtab_cross_reciprocal_sample_source_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(checked, max_error, missing_checked)


if __name__ == "__main__":
    main()
