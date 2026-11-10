"""Read-only CPU diagnostic: canonical-prompt token distribution at FIXED tweet caps, over all three
splits. Purpose is to show WHY the adaptive fitter is needed — a fixed cap of 30 leaves a long tail
over budget, which is exactly the truncation risk the adaptive builder removes.

This uses the same prompt_v4 canonical formatter as the builder and E1/E2 (no bespoke wrapper), so
the counts here match the real prompt. The authoritative, per-account, in-budget counts live in the
render manifest produced by build_render_v4.py; this script is diagnostic only.

    python scripts/probe_prompt_token_lengths.py --caps 10 20 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from verifiable_bot.v4.prompt_v4 import PROMPT_BUDGET, count_tokens, default_model_path
from verifiable_bot.v4.render_v4 import build_units_v4, render_units
from verifiable_bot.v4.thresholds_v4 import load_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--data-dir", default="data/twibot20_complete")
    parser.add_argument("--thresholds", default="outputs/v4/render_v4/bucket_thresholds_v4.json")
    parser.add_argument("--caps", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--budget", type=int, default=PROMPT_BUDGET)
    parser.add_argument("--out", default="outputs/v4/render_v4/prompt_token_report.json")
    args = parser.parse_args()

    model_path = args.model or default_model_path()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    thresholds = load_thresholds(args.thresholds)
    records = {s: json.loads((Path(args.data_dir) / f"{s}.json").read_text(encoding="utf-8"))
               for s in ("train", "dev", "test")}

    report = {"model": model_path, "budget": args.budget, "note":
              "FIXED-cap diagnostic via canonical prompt_v4 formatter; the builder uses adaptive "
              "per-account fitting so the shipped shards have zero over-budget accounts.", "caps": {}}
    print(f"canonical formatter, budget {args.budget}\n")
    for cap in args.caps:
        report["caps"][str(cap)] = {}
        print(f"=== fixed max_tweets = {cap} (no adaptation) ===")
        for split, recs in records.items():
            lengths = [count_tokens(render_units(build_units_v4(r, thresholds, max_tweets=cap)), tok)
                       for r in recs]
            arr = np.asarray(lengths)
            stats = {"n": int(arr.size), "p50": int(np.percentile(arr, 50)),
                     "p95": int(np.percentile(arr, 95)), "p99": int(np.percentile(arr, 99)),
                     "max": int(arr.max()), "over_budget": int((arr > args.budget).sum()),
                     "over_budget_frac": round(float((arr > args.budget).mean()), 4)}
            report["caps"][str(cap)][split] = stats
            print(f"  {split:5s} p50={stats['p50']:5d} p95={stats['p95']:5d} p99={stats['p99']:5d} "
                  f"max={stats['max']:5d}  >budget: {stats['over_budget']}/{stats['n']} "
                  f"({stats['over_budget_frac']*100:.2f}%)")
        print()

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {args.out}")
    print("注：固定 cap 有超预算长尾，故 builder 采用自适应逐账号降 tweet，shard 内零超限。")


if __name__ == "__main__":
    main()
