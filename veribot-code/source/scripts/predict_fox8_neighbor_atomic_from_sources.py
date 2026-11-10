"""Return one frozen Fox8 prediction with its source-linked evidence ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("account_id")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--top", type=int, default=12,
                        help="number of evidence terms to show; full ledger is always computed")
    args = parser.parse_args()
    if args.top < 1 or args.top > 74:
        raise ValueError("--top must lie between 1 and 74")
    root = Path(__file__).resolve().parents[2]
    result = Fox8NeighborAtomicRuntime(root, args.seed).predict(args.account_id)
    result["evidence"] = result["evidence"][:args.top]
    result["displayed_evidence_terms"] = args.top
    result["total_evidence_terms_in_model"] = 74
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
