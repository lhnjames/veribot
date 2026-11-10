"""Issue one MGTAB typed-edge structural fact certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_structural_certificate import certificate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--account-index", type=int, required=True)
    parser.add_argument("--fact-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = certificate(args.data_dir, args.facts, args.account_index,
                         args.fact_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
