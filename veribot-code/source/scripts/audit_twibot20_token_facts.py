"""Independently rescan official tweets and audit every token-incidence cell."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile

import ijson
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(archive: Path, facts: Path) -> dict:
    manifest = json.loads(facts.with_suffix(".json").read_text(encoding="utf-8"))
    if (sha256(archive) != manifest["source_archive_sha256"]
            or sha256(facts) != manifest["facts_sha256"]):
        raise ValueError("official archive or fact cache differs")
    with np.load(facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str).tolist()
        parts = cache["split"].astype(str).tolist()
        positions = cache["positions"].astype(int).tolist()
        names = cache["names"].astype(str).tolist()
        actual = cache["values"].astype(float)
    vocabulary = manifest["vocabulary"]
    if names != [f"tweet_token_incidence:{token}" for token in vocabulary]:
        raise ValueError("fact vocabulary differs")
    column = {token: j for j, token in enumerate(vocabulary)}
    stop = set(ENGLISH_STOP_WORDS) | {"http", "https", "www", "com",
                                      "url", "user", "ret", "amp"}
    pattern = re.compile(r"(?<![a-z])[a-z]{3,}(?![a-z])")
    url = re.compile(r"https?://\S+", re.I)
    mention = re.compile(r"(?<!\w)@\w+")
    expected = np.full_like(actual, np.nan)
    document_frequency: Counter[str] = Counter()
    row_index = 0
    with zipfile.ZipFile(archive) as zipped:
        for part in ("train", "dev", "test"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    if (row_index >= len(ids) or ids[row_index] != str(row["ID"])
                            or parts[row_index] != part
                            or positions[row_index] != position):
                        raise ValueError("official row identity differs")
                    tweets = row.get("tweet") or []
                    counts = Counter()
                    account_tokens = set()
                    for tweet in tweets:
                        cleaned = mention.sub(" ", url.sub(" ", str(tweet or "").casefold()))
                        tokens = {token for token in pattern.findall(cleaned)
                                  if token not in stop}
                        account_tokens.update(tokens)
                        for token in tokens:
                            if token in column:
                                counts[column[token]] += 1
                    if part == "train":
                        document_frequency.update(account_tokens)
                    if tweets:
                        expected[row_index] = 0.0
                        for j, count in counts.items():
                            expected[row_index, j] = count / len(tweets)
                    row_index += 1
    if row_index != len(ids):
        raise ValueError("official account count differs")
    ranked = [token for token, _ in sorted(
        document_frequency.items(), key=lambda item: (-item[1], item[0]))
        [:len(vocabulary)]]
    if ranked != vocabulary:
        raise ValueError("label-blind train-only vocabulary differs")
    missing_match = bool(np.array_equal(np.isnan(expected), np.isnan(actual)))
    maximum = float(np.max(np.abs(np.nan_to_num(expected) -
                                  np.nan_to_num(actual))))
    if not missing_match or maximum > 1e-7:
        raise ValueError("token incidence facts differ from raw tweets")
    return {"dataset": "twibot20",
            "status": "verified_all_token_fact_cells_and_train_only_vocabulary",
            "account_count": len(ids), "token_count": len(vocabulary),
            "fact_cells_including_missing": int(actual.size),
            "nonmissing_fact_cells": int(np.isfinite(actual).sum()),
            "missingness_match": missing_match,
            "max_absolute_numeric_difference": maximum,
            "train_only_vocabulary_match": True,
            "label_fields_read": False,
            "source_archive_sha256": manifest["source_archive_sha256"],
            "fact_cache_sha256": manifest["facts_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.zip, args.facts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
