"""Render train/dev/test into v4 jsonl shards using the frozen train-fitted thresholds and the
canonical prompt formatter, adaptively fitting every account under the token budget.

    python scripts/build_render_v4.py --split all

Thresholds are loaded read-only; this script never fits anything, so running it on dev/test cannot
move a boundary. Every account is fitted through prompt_v4.fit_account_prompt (candidate cap 30,
evenly-sampled tweets, dropped one at a time until the whole canonical prompt is <= budget). After
building, a HARD assertion fails the run if any account exceeds the budget — there is no silent
truncation anywhere in the pipeline.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from verifiable_bot.v4.prompt_v4 import (
    CANDIDATE_MAX_TWEETS,
    OUTPUT_BUDGET,
    PROMPT_BUDGET,
    PROMPT_VERSION,
    canonical_prompt_sha256,
    default_model_path,
    fit_account_prompt,
    tokenizer_fingerprint,
)
from verifiable_bot.v4.render_v4 import record_payload
from verifiable_bot.v4.thresholds_v4 import load_thresholds, sha256_of

SPLITS = ("train", "dev", "test")


def build(split, data_dir, out_dir, thresholds, tokenizer, budget, start_tweets):
    records = json.loads((data_dir / f"{split}.json").read_text(encoding="utf-8"))
    out_path = out_dir / f"{split}.jsonl"
    tmp = out_path.with_suffix(".jsonl.tmp")
    stats = Counter()
    prefix_counts = Counter()
    token_counts: list[int] = []
    effective_caps: list[int] = []
    tweet_units: list[int] = []
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            units, tokens, effective_cap = fit_account_prompt(
                record, thresholds, tokenizer, budget=budget, start_tweets=start_tweets
            )
            payload = record_payload(
                record, units, prompt_token_count=tokens,
                candidate_max_tweets=start_tweets, effective_tweet_cap=effective_cap,
            )
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            token_counts.append(tokens)
            effective_caps.append(effective_cap)
            tweet_units.append(payload["n_tweet_units"])
            stats["accounts"] += 1
            stats[f"label_{payload['label']}"] += 1
            if payload["n_tweet_units"] == 0:
                stats["no_tweet_units"] += 1
            if not any(u["unit_id"].startswith("ACTIVITY_") for u in payload["units"]):
                stats["no_activity_units"] += 1
            if not any(u["unit_id"].startswith("RELATION_") for u in payload["units"]):
                stats["no_relation_units"] += 1
            for unit in payload["units"]:
                prefix_counts[unit["unit_id"].split("_")[0]] += 1
                bad = [ch for ch in unit["text"] if ch.isdigit()]
                if bad:  # hard-fail, do not merely count: numeric glyphs must never ship
                    raise ValueError(
                        f"{split}/{payload['account_id']} unit {unit['unit_id']} has numeric "
                        f"glyphs {bad}: {unit['text']!r}")
    tmp.replace(out_path)  # atomic

    over = int(sum(1 for t in token_counts if t > budget))
    assert over == 0, f"{split}: {over} accounts exceed the {budget}-token budget — must be zero"
    tok = np.asarray(token_counts)
    cap = np.asarray(effective_caps)
    tw = np.asarray(tweet_units)
    # adapted_for_budget: the cap was actually lowered by the fitter (effective < candidate).
    # natural_timeline_shorter_than_cap: cap stayed at candidate but the account simply has fewer
    # tweets than the cap — NOT an adaptation, and previously miscounted by comparing n_tweets<30.
    adapted = int((cap < start_tweets).sum())
    natural_short = int(((cap >= start_tweets) & (tw < start_tweets)).sum())
    return {
        "path": str(out_path),
        "stats": dict(stats),
        "unit_prefixes": dict(prefix_counts),
        "prompt_tokens": {
            "min": int(tok.min()), "p50": int(np.percentile(tok, 50)),
            "p95": int(np.percentile(tok, 95)), "max": int(tok.max()), "over_budget": over,
        },
        "effective_tweet_cap_dist": {
            "min": int(cap.min()), "p50": int(np.percentile(cap, 50)),
            "p95": int(np.percentile(cap, 95)), "max": int(cap.max()),
        },
        "n_tweet_units_dist": {
            "min": int(tw.min()), "p50": int(np.percentile(tw, 50)),
            "p95": int(np.percentile(tw, 95)), "max": int(tw.max()),
        },
        "adapted_for_budget": adapted,
        "natural_timeline_shorter_than_cap": natural_short,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/twibot20_complete")
    parser.add_argument("--out-dir", default="outputs/v4/render_v4")
    parser.add_argument("--thresholds", default="outputs/v4/render_v4/bucket_thresholds_v4.json")
    parser.add_argument("--split", default="all", choices=(*SPLITS, "all"))
    parser.add_argument("--budget", type=int, default=PROMPT_BUDGET)
    parser.add_argument("--start-tweets", type=int, default=CANDIDATE_MAX_TWEETS)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args.thresholds)
    model_path = args.model or default_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    manifest = {
        "prompt_version": PROMPT_VERSION,
        "canonical_prompt_sha256": canonical_prompt_sha256(tokenizer),
        "thresholds_file": str(Path(args.thresholds).resolve()),
        "thresholds_sha256": sha256_of(args.thresholds),
        "thresholds_source": thresholds["source_file"],
        "prompt_budget": args.budget,
        "output_budget": OUTPUT_BUDGET,
        "candidate_max_tweets": args.start_tweets,
        "tokenizer": tokenizer_fingerprint(tokenizer, model_path),
        "splits": {},
    }
    for split in (SPLITS if args.split == "all" else (args.split,)):
        result = build(split, data_dir, out_dir, thresholds, tokenizer, args.budget, args.start_tweets)
        manifest["splits"][split] = result
        print(f"{split}: {result['stats']}")
        print(f"   prompt tokens: {result['prompt_tokens']}")
        print(f"   n_tweet_units: {result['n_tweet_units_dist']}")

    (out_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nmanifest -> {out_dir / 'render_manifest.json'}")
    print("HARD ASSERTION PASSED: every account is within the token budget on every split.")


if __name__ == "__main__":
    main()
