"""Build source-addressable lexical facts from the official TwiBot-20 ZIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.twibot20_lexical_facts import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.zip, args.output, args.base_facts)
    print(json.dumps({"output": str(args.output),
                      "accounts": manifest["account_count"],
                      "facts": len(manifest["fact_names"]),
                      "source_archive_sha256": manifest["source_archive_sha256"]},
                     indent=2))


if __name__ == "__main__":
    main()
