"""Build label-blind source-addressable token incidence facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.twibot20_token_facts import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocabulary-size", type=int, default=128)
    args = parser.parse_args()
    manifest = build(args.zip, args.base_facts, args.output,
                     vocabulary_size=args.vocabulary_size)
    print(json.dumps({"accounts": manifest["account_count"],
                      "tokens": len(manifest["vocabulary"]),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
