"""Build the compact TwiBot-20 fact table directly from the official ZIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.twibot20_facts import build, save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    graph_scope = parser.add_mutually_exclusive_group()
    graph_scope.add_argument("--include-support-neighbor-edges", action="store_true",
                             help="include unlabeled support accounts' provided sampled neighbor declarations (default)")
    graph_scope.add_argument("--labeled-only-neighbor-edges", action="store_true",
                             help="rebuild the earlier labeled-row-only graph variant")
    args = parser.parse_args()
    bundle = build(args.zip,
                   include_support_neighbor_edges=not args.labeled_only_neighbor_edges)
    save(bundle, args.output)
    print(json.dumps({"output": str(args.output), "shape": bundle["values"].shape,
                      "manifest": bundle["manifest"]}, indent=2))


if __name__ == "__main__":
    main()
