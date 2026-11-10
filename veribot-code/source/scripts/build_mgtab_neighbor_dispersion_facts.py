"""Build the 12 published-source MGTAB neighbor dispersion facts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_neighbor_dispersion_facts import build
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    result = build(resolve_mgtab_source_dir(ROOT),
                   ROOT / "data/processed/mgtab_v3/neighbor_dispersion_facts.npz")
    print({"account_count": result["account_count"],
           "fact_count": len(result["feature_names"]),
           "facts_sha256": result["facts_sha256"]})
