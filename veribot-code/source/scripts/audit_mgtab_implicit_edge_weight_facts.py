"""Independently rebuild all implicit-edge facts from published edge rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_source_certificate import sha256
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir


ROOT = Path(__file__).resolve().parents[2]
FACTS = ROOT / "data/processed/mgtab_v3/implicit_edge_weight_facts.npz"
OUTPUT = ROOT / "research/certificates/mgtab_implicit_edge_weight_full_source_audit.json"


def main() -> None:
    data = resolve_mgtab_source_dir(ROOT)
    edge = torch.load(data / "edge_index.pt", map_location="cpu",
                      weights_only=True).numpy()
    kind = torch.load(data / "edge_type.pt", map_location="cpu",
                      weights_only=True).numpy()
    weight = torch.load(data / "edge_weight.pt", map_location="cpu",
                        weights_only=True).numpy()
    manifest = json.loads(FACTS.with_suffix(".json").read_text(
        encoding="utf-8"))
    with np.load(FACTS, allow_pickle=False) as cache:
        values = cache["values"]
        supports = cache["supports"]
        names = cache["names"].astype(str).tolist()
        ids = cache["ids"]
    if (values.shape != (10199, 8) or supports.shape != values.shape or
            not np.array_equal(ids, np.arange(10199)) or
            names != manifest["names"] or sha256(FACTS) !=
            manifest["facts_sha256"] or any(
                sha256(data / name) != digest for name, digest in
                manifest["input_sha256"].items())):
        raise ValueError("published sources or cached facts changed")
    expected = np.full((10199, 8), np.nan, dtype=float)
    expected_support = np.zeros((10199, 8), dtype=int)
    for k, relation in ((5, "url"), (6, "hashtag")):
        adjacency: list[list[float]] = [[] for _ in range(10199)]
        selected = np.flatnonzero(kind == k)
        for edge_row in selected:
            left, right = map(int, edge[:, edge_row])
            observed_weight = float(weight[edge_row])
            adjacency[left].append(observed_weight)
            adjacency[right].append(observed_weight)
        offset = 0 if k == 5 else 4
        for account, incident in enumerate(adjacency):
            count = len(incident)
            expected_support[account, offset:offset + 4] = count
            expected[account, offset] = sum(incident)
            expected[account, offset + 3] = sum(
                value >= 0.7 for value in incident)
            if count:
                expected[account, offset + 1] = sum(incident) / count
                expected[account, offset + 2] = max(incident)
    if (not np.array_equal(np.isnan(values), np.isnan(expected)) or
            not np.array_equal(supports, expected_support)):
        raise ValueError("missingness or support differs from edge rows")
    finite = np.isfinite(expected)
    error = float(np.max(np.abs(values[finite] - expected[finite])))
    scaled_error = float(np.max(np.abs(values[finite] - expected[finite]) /
                                (1.0 + np.abs(expected[finite]))))
    if scaled_error > 5e-7:
        raise ValueError(f"implicit edge-weight values differ: "
                         f"absolute={error}, scaled={scaled_error}")
    result = {
        "status": "all_published_edge_weight_fact_cells_rebuilt",
        "account_count": 10199, "fact_count": 8,
        "fact_cells_checked": int(values.size),
        "missing_cells_checked": int(np.isnan(values).sum()),
        "max_absolute_value_error": error,
        "max_scaled_value_error": scaled_error,
        "reconstruction": "Independent per-account edge-row adjacency lists; each published undirected edge weight appended to both endpoints.",
        "source_limit": "edge_weight.pt is published; original shared URL/hashtag strings are unavailable",
        "facts_sha256": sha256(FACTS),
        "source_sha256": manifest["input_sha256"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n",
                      encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
