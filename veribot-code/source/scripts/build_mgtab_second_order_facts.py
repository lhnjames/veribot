"""Build label-free neighbor degree evidence from published MGTAB relations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_second_order_facts import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.data_dir, args.output)
    print(json.dumps({"output": str(args.output),
                      "accounts": result["account_count"],
                      "facts": len(result["feature_names"]),
                      "source_level": result["source_level"]}, indent=2))


if __name__ == "__main__":
    main()
