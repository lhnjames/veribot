"""Build eight label-blind, source-bound MGTAB implicit-edge weight facts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_implicit_edge_weight_facts import build
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    manifest = build(resolve_mgtab_source_dir(ROOT),
                     ROOT / "data/processed/mgtab_v3/implicit_edge_weight_facts.npz")
    print(manifest["relation_summary"])
