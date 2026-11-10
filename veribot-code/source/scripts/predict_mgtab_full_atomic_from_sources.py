"""Return one MGTAB frozen prediction with source-level evidence terms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("account_id")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--variant", choices=("full_atomic", "ab_group_additive"),
                        default="ab_group_additive")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.top <= 150:
        raise ValueError("--top must lie between 1 and 150")
    root = Path(__file__).resolve().parents[2]
    result = MGTABFullAtomicRuntime(root, args.seed,
                                    variant=args.variant).predict(args.account_id)
    result["evidence"] = result["evidence"][:args.top]
    result["displayed_evidence_terms"] = args.top
    result["total_evidence_terms_in_model"] = 150
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
