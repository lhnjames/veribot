"""Fit and freeze the v4 bucket thresholds. TRAIN SPLIT ONLY.

    python scripts/compute_bucket_thresholds_v4.py \
        --train data/twibot20_complete/train.json \
        --out   outputs/v4/render_v4/bucket_thresholds_v4.json

The fitter itself refuses any file not named train.json (thresholds_v4.fit_thresholds), so there is
no flag here that could be used to point it at dev/test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from verifiable_bot.v4.thresholds_v4 import fit_thresholds, save_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/twibot20_complete/train.json")
    parser.add_argument("--out", default="outputs/v4/render_v4/bucket_thresholds_v4.json")
    args = parser.parse_args()

    thresholds = fit_thresholds(args.train)
    path = save_thresholds(thresholds, args.out)

    print(f"fitted on : {thresholds['source_file']}")
    print(f"sha256    : {thresholds['source_sha256']}")
    print(f"accounts  : {thresholds['n_accounts']} (with tweets: {thresholds['n_accounts_with_tweets']})")
    print(f"convention: {thresholds['interval_convention']}")
    print(f"written   : {Path(path).resolve()}\n")
    header = f"{'feature':22s} {'idx':>3s} {'distinct':>8s}  cuts (p10,p30,p70,p90)"
    print(header)
    print("-" * len(header))
    for name, spec in thresholds["features"].items():
        cuts = ", ".join(f"{c:.4g}" for c in spec["cuts"])
        flag = "" if spec["n_distinct_cuts"] == 4 else "  <- tied cuts, some slots unreachable"
        print(f"{name:22s} {spec['index']:3d} {spec['n_distinct_cuts']:8d}  [{cuts}]{flag}")


if __name__ == "__main__":
    main()
