"""Build source-addressable MGTAB neighbor profile facts without labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_neighbor_facts import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.data_dir, args.output)
    print(json.dumps({"account_count": result["account_count"],
                      "fact_count": len(result["feature_names"]),
                      "facts_sha256": result["facts_sha256"],
                      "relation_summary": result["relation_summary"]}, indent=2))


if __name__ == "__main__":
    main()
