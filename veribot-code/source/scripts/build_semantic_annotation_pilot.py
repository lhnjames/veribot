"""Build blind semantic-evidence pilot packets from TwiBot-20 and BotSim records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.semantic_annotation import botsim_packets, save, twibot_packets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twibot20-zip", type=Path, required=True)
    parser.add_argument("--botsim-activity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    twibot, twibot_source = twibot_packets(args.twibot20_zip)
    botsim, botsim_source = botsim_packets(args.botsim_activity)
    manifest = save(twibot + botsim, twibot_source + botsim_source,
                    args.output_dir, [args.twibot20_zip, args.botsim_activity])
    print(json.dumps({"output_dir": str(args.output_dir), "counts": manifest["counts"],
                      "blind_packet_sha256": manifest["blind_packet_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
