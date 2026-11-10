"""Export one source-recomputed Fox8 evidence certificate from a frozen v2 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.evidence_v2 import AtomicClaim, audit_candidate
from verifiable_bot.fox8_adapter import load_fox8_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--kind", choices=("profile", "tweet"), default="profile")
    parser.add_argument("--field", default="followers_count")
    parser.add_argument("--tweet-ordinal", type=int, default=0)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    adapter, sources = load_fox8_adapter(args.raw, args.features, args.checkpoint,
                                          args.account_id, knn_k=args.knn_k)
    if args.kind == "profile":
        source_id = f"fox8:{args.account_id}:profile:{args.field}"
        if source_id not in sources:
            parser.error(f"raw profile has no field {args.field!r}")
        controls = ()
        control_protocol = "none"
    else:
        tweets = sorted((sid for sid, record in sources.items() if record.kind == "tweet"),
                        key=lambda sid: int(sid.rsplit(":", 1)[1]))
        if not 0 <= args.tweet_ordinal < len(tweets):
            parser.error("tweet ordinal outside this account's records")
        source_id = tweets[args.tweet_ordinal]
        controls = tuple(sid for sid in tweets if sid != source_id)[:1]
        control_protocol = "first other tweet; unmatched diagnostic"
    record = sources[source_id]
    if args.field not in record.payload:
        parser.error(f"field {args.field!r} is absent from {source_id}")
    claim = AtomicClaim((source_id,), "field", record.payload[args.field], args.field)
    report = audit_candidate(args.account_id, claim, sources, adapter, controls,
                             control_protocol=control_protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": report["status"], "reason": report.get("reason"),
                      "delta_margin": report.get("delta_margin"),
                      "prediction_flipped": report.get("prediction_flipped"),
                      "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
