"""Independently rebuild every neighbor-dispersion value from typed edges."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_neighbor_dispersion_facts import (
    PROFILE_INDICES, RELATION_INDICES)
from verifiable_bot.mgtab_profile_schema import PROFILE_SCHEMA
from verifiable_bot.mgtab_source_certificate import RELATIONS, sha256
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir


ROOT = Path(__file__).resolve().parents[2]
FACTS = ROOT / "data/processed/mgtab_v3/neighbor_dispersion_facts.npz"
OUTPUT = ROOT / "research/certificates/mgtab_neighbor_dispersion_full_source_audit.json"


def main() -> None:
    data = resolve_mgtab_source_dir(ROOT)
    manifest = json.loads(FACTS.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest["facts_sha256"] != sha256(FACTS):
        raise ValueError("derived fact artifact changed")
    for file in ("features.pt", "edge_index.pt", "edge_type.pt"):
        if manifest["input_sha256"][file] != sha256(data / file):
            raise ValueError(f"published source {file} changed")
    profile = torch.load(data / "features.pt", map_location="cpu",
                         weights_only=True).numpy().astype(float)
    edges = torch.load(data / "edge_index.pt", map_location="cpu",
                       weights_only=True).numpy().astype(int)
    kinds = torch.load(data / "edge_type.pt", map_location="cpu",
                       weights_only=True).numpy().astype(int)
    with np.load(FACTS, allow_pickle=False) as cache:
        values = cache["values"].astype(float)
        supports = cache["supports"].astype(int)
        names = cache["names"].astype(str).tolist()
        ids = cache["ids"].astype(int)
    n = len(profile)
    if (values.shape != (n, 12) or supports.shape != values.shape or
            not np.array_equal(ids, np.arange(n)) or
            names != manifest["feature_names"]):
        raise ValueError("derived fact schema differs")
    max_error = 0.0
    checked, missing = 0, 0
    columns = []
    for relation_index in RELATION_INDICES:
        relation = RELATIONS[relation_index]
        out_sets = [set() for _ in range(n)]
        in_sets = [set() for _ in range(n)]
        relevant = np.flatnonzero(kinds == relation_index)
        for e in relevant:
            source, target = int(edges[0, e]), int(edges[1, e])
            if source != target:
                out_sets[source].add(target)
                in_sets[target].add(source)
        for direction, owners in (("out", out_sets), ("in", in_sets)):
            for feature in PROFILE_INDICES:
                name = (f"published_{relation}_{direction}_neighbor_"
                        f"{PROFILE_SCHEMA[feature][0]}_std")
                if names[len(columns)] != name:
                    raise ValueError("dispersion column order differs")
                j = len(columns)
                present = 0
                for account, neighbors in enumerate(owners):
                    count = len(neighbors)
                    if supports[account, j] != count:
                        raise ValueError("neighbor support count differs")
                    actual = values[account, j]
                    if count == 0:
                        if not np.isnan(actual):
                            raise ValueError("missing neighbor value was filled")
                        missing += 1
                    else:
                        expected = float(np.std(profile[list(neighbors), feature],
                                                ddof=0))
                        error = abs(actual - expected)
                        max_error = max(max_error, error)
                        if error > 1e-6:
                            raise ValueError("neighbor dispersion differs from edges")
                        present += 1
                    checked += 1
                columns.append({"name": names[j], "nonmissing_accounts": present})
        print({"relation": relation, "checked_columns": len(columns)}, flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "scope": "full_independent_edge_set_and_profile_std_recalculation",
        "account_count": n,
        "fact_count": len(columns),
        "fact_cells_checked": checked,
        "missing_cells_checked": missing,
        "max_absolute_difference": max_error,
        "facts_sha256": sha256(FACTS),
        "source_hashes": manifest["input_sha256"],
        "columns": columns,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
