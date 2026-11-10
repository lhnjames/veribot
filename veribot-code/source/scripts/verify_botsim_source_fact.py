"""Issue one BotSim reply graph fact certificate from original post/comment records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.botsim_source_certificate import certificate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=Path, required=True)
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--facts-dir", type=Path, required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--fact-name", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = certificate(args.users, args.activity, args.facts_dir,
                         args.account_id, args.fact_name)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
