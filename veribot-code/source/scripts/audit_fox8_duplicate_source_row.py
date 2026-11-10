"""Check the duplicate raw Fox8 ID against the frozen first-row input policy."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.fox8_adapter import FEATURE_FIELDS, profile_features, tweet_features


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
OUT = ROOT / "research/certificates/fox8_duplicate_source_row_audit.json"


def features(row: dict) -> np.ndarray:
    tweets = row["user_tweets"]
    user = tweets[0]["user"]
    values = {**profile_features(user), **tweet_features(tweets)}
    return np.asarray([values[name] for name in FEATURE_FIELDS], dtype=np.float32)


def main() -> None:
    pattern = re.compile(rb'^\{"user_id"\s*:\s*"?(\d+)"?')
    first: dict[str, tuple[int, bytes]] = {}
    duplicate = []
    with gzip.open(RAW, "rb") as stream:
        for line_number, line in enumerate(stream, 1):
            match = pattern.match(line)
            if match is None:
                raise ValueError("raw Fox8 row lacks account ID")
            uid = match.group(1).decode("ascii")
            if uid in first:
                duplicate.append((uid, first[uid][0], first[uid][1],
                                  line_number, line))
            else:
                first[uid] = (line_number, line)
    if len(duplicate) != 1:
        raise ValueError("expected one duplicate Fox8 source row")
    uid, first_line, first_bytes, duplicate_line, duplicate_bytes = duplicate[0]
    first_row = json.loads(first_bytes)
    later_row = json.loads(duplicate_bytes)
    first_features, later_features = features(first_row), features(later_row)
    changed = [name for name, a, b in zip(
        FEATURE_FIELDS, first_features, later_features, strict=True) if a != b]
    test_membership = {}
    for seed in (0, 1, 2):
        path = ROOT / ("output/v3_fox8_neighbor_atomic_evidence_capacity/"
                       f"seed{seed}/test_predictions.json")
        ids = {str(row["account_id"]) for row in json.loads(
            path.read_text(encoding="utf-8"))["predictions"]}
        test_membership[str(seed)] = uid in ids
    cited_as_neighbor = {}
    for seed in (0, 1, 2):
        path = ROOT / ("output/v3_best_verified_evidence_full_test/"
                       f"fox8_seed{seed}.jsonl.gz")
        references = 0
        citing_accounts = set()
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                card = json.loads(line)
                for term in card["evidence_terms"]:
                    query = term["source_query"]
                    if query is None:
                        continue
                    count = sum(f"#line={first_line},user_id={uid}" in source
                                for source in query["source_ids"])
                    references += count
                    if count:
                        citing_accounts.add(card["account_id"])
        cited_as_neighbor[str(seed)] = {
            "source_references": references,
            "distinct_test_accounts": len(citing_accounts),
        }
    OUT.write_text(json.dumps({
        "status": "first_duplicate_source_row_identified",
        "account_id": uid,
        "first_line": first_line,
        "later_line": duplicate_line,
        "first_row_sha256": hashlib.sha256(first_bytes).hexdigest(),
        "later_row_sha256": hashlib.sha256(duplicate_bytes).hexdigest(),
        "same_raw_bytes": first_bytes == duplicate_bytes,
        "same_31_model_account_features": bool(np.array_equal(
            first_features, later_features)),
        "changed_account_feature_names": changed,
        "first_sampled_tweet_count": len(first_row["user_tweets"]),
        "later_sampled_tweet_count": len(later_row["user_tweets"]),
        "in_fixed_test_by_seed": test_membership,
        "cited_by_test_cards": cited_as_neighbor,
        "source_selection_rule": "first raw row, matching Fox8 adapter and frozen preprocessing",
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
