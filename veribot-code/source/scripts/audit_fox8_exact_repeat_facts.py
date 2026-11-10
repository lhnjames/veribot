"""Independently reconstruct every Fox8 exact-repeat fact and tweet locator."""

from __future__ import annotations

from collections import defaultdict
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.facts.with_suffix(".json").read_text(encoding="utf-8"))
    trace_path = args.facts.with_name("exact_repeat_sources.jsonl.gz")
    if (manifest["source_sha256"] != sha256(args.raw)
            or manifest["features_csv_sha256"] != sha256(args.features_csv)
            or manifest["facts_sha256"] != sha256(args.facts)
            or manifest["trace_sha256"] != sha256(trace_path)):
        raise ValueError("repeat fact input or output hash differs")
    with args.features_csv.open("r", encoding="utf-8", newline="") as source:
        ids = [row["user_id"] for row in csv.DictReader(source)]
    with np.load(args.facts, allow_pickle=False) as cached:
        if cached["ids"].astype(str).tolist() != ids or \
                cached["names"].astype(str).tolist() != [manifest["fact_name"]]:
            raise ValueError("repeat fact account order or name differs")
        values = cached["values"][:, 0].astype(float)
        supports = cached["supports"][:, 0].astype(int)
    lookup = {uid: index for index, uid in enumerate(ids)}
    seen = set()
    repeated_accounts = 0
    repeated_tweets = 0
    with gzip.open(args.raw, "rt", encoding="utf-8") as source, \
            gzip.open(trace_path, "rt", encoding="utf-8") as trace:
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            uid = str(record["user_id"])
            if uid in seen:
                continue
            seen.add(uid)
            saved = json.loads(next(trace))
            if saved["account_id"] != uid or saved["raw_line"] != line_number:
                raise ValueError("repeat source locator order differs")
            groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
            for index, tweet in enumerate(record.get("user_tweets") or []):
                value = " ".join(str(tweet.get("text") or "").casefold().split())
                if value:
                    tweet_id = str(tweet.get("id_str") or tweet.get("id") or index)
                    groups[value].append((index, tweet_id))
            denominator = sum(len(group) for group in groups.values())
            repeated = [(text, group) for text, group in sorted(groups.items())
                        if len(group) >= 2]
            numerator = sum(len(group) for _, group in repeated)
            expected = numerator / denominator if denominator else float("nan")
            if (saved["nonempty_sampled_tweets"] != denominator
                    or saved["repeated_tweet_count"] != numerator
                    or supports[lookup[uid]] != denominator
                    or not np.allclose(values[lookup[uid]], expected, equal_nan=True,
                                       rtol=0, atol=1e-15)
                    or len(saved["repeated_groups"]) != len(repeated)):
                raise ValueError("repeat fact arithmetic differs from raw tweets")
            for item, (text, group) in zip(saved["repeated_groups"], repeated):
                if (item["normalized_text_sha256"] !=
                        hashlib.sha256(text.encode("utf-8")).hexdigest()
                        or item["tweet_indices"] != [index for index, _ in group]
                        or item["tweet_ids"] != [tweet_id for _, tweet_id in group]):
                    raise ValueError("repeat tweet locator differs")
            repeated_accounts += int(numerator > 0)
            repeated_tweets += numerator
        if next(trace, None) is not None:
            raise ValueError("repeat trace has extra accounts")
    if len(seen) != len(ids) or seen != set(ids):
        raise ValueError("repeat audit population differs")
    result = {
        "dataset": "fox8", "fact_name": manifest["fact_name"],
        "accounts": len(ids), "repeated_accounts": repeated_accounts,
        "repeated_tweets": repeated_tweets,
        "facts_sha256": manifest["facts_sha256"],
        "trace_sha256": manifest["trace_sha256"],
        "source_sha256": manifest["source_sha256"],
        "status": "all_accounts_values_and_tweet_locators_recomputed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
