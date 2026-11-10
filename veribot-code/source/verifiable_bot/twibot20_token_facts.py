"""Label-blind, source-addressable tweet-token incidence facts for TwiBot-20."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile

import ijson
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


URL = re.compile(r"https?://\S+", re.IGNORECASE)
MENTION = re.compile(r"(?<!\w)@\w+")
TOKEN = re.compile(r"(?<![a-z])[a-z]{3,}(?![a-z])")
EXCLUDED = set(ENGLISH_STOP_WORDS) | {"http", "https", "www", "com",
                                      "url", "user", "ret", "amp"}


def tokenize(text: str) -> set[str]:
    cleaned = MENTION.sub(" ", URL.sub(" ", str(text or "").casefold()))
    return {value for value in TOKEN.findall(cleaned)
            if value not in EXCLUDED}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(archive: Path, base_facts: Path, output: Path,
          *, vocabulary_size: int = 128) -> dict:
    if vocabulary_size < 1:
        raise ValueError("vocabulary size must be positive")
    ids, parts, positions = [], [], []
    account_tweets: list[list[set[str]]] = []
    document_frequency: Counter[str] = Counter()
    with zipfile.ZipFile(archive) as zipped:
        for part in ("train", "dev", "test"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    ids.append(str(row["ID"]))
                    parts.append(part)
                    positions.append(position)
                    tweets = [tokenize(text) for text in row.get("tweet") or []]
                    account_tweets.append(tweets)
                    if part == "train":
                        document_frequency.update(set().union(*tweets))
    with np.load(base_facts, allow_pickle=False) as base:
        if (ids != base["ids"].astype(str).tolist()
                or parts != base["split"].astype(str).tolist()
                or positions != base["positions"].astype(int).tolist()):
            raise ValueError("token source population differs from official facts")
    vocabulary = [token for token, _ in sorted(
        document_frequency.items(), key=lambda item: (-item[1], item[0]))
        [:vocabulary_size]]
    if len(vocabulary) != vocabulary_size:
        raise ValueError("not enough training vocabulary")
    column = {token: j for j, token in enumerate(vocabulary)}
    values = np.full((len(ids), len(vocabulary)), np.nan, dtype=np.float32)
    for i, tweets in enumerate(account_tweets):
        if not tweets:
            continue
        values[i] = 0.0
        counts: Counter[int] = Counter()
        for tokens in tweets:
            for token in tokens:
                j = column.get(token)
                if j is not None:
                    counts[j] += 1
        for j, count in counts.items():
            values[i, j] = count / len(tweets)
    if values.shape != (11826, vocabulary_size):
        raise ValueError("token fact matrix differs from official population")
    names = [f"tweet_token_incidence:{token}" for token in vocabulary]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.asarray(ids), split=np.asarray(parts),
                        positions=np.asarray(positions),
                        names=np.asarray(names), values=values)
    manifest = {
        "dataset": "twibot20", "account_count": len(ids),
        "vocabulary_selection": "most account-document-frequent tokens in official train; no labels",
        "vocabulary": vocabulary,
        "fact_names": names,
        "formula": "sampled tweets containing token / number of sampled tweets; missing when no tweets",
        "token_rule": "casefold; remove URLs and @mentions; ASCII alphabetic tokens of length >=3; remove sklearn English stopwords and listed artifacts",
        "source_locator": "official ZIP member + account array ordinal + tweet index + exact token",
        "source_archive_sha256": sha256(archive),
        "base_facts_sha256": sha256(base_facts),
        "facts_sha256": sha256(output),
        "label_fields_read": False,
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
