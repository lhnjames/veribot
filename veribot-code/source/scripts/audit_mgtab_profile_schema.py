"""Check MGTAB's first 20 published coordinates against paper Table A5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_profile_schema import audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mapping_status": result["mapping_status"],
                      "accounts": result["accounts"],
                      "coordinates_checked": len(result["coordinates"]),
                      "features_sha256": result["features_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
