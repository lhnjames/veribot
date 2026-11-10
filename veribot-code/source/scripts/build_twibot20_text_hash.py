"""Hash source-addressable TwiBot-20 profile and tweet text on the server.

The vectorizer is fixed and unsupervised. Support rows receive text features
too, so full-graph inference uses the same source population as training.
Every feature can be recalculated from its archive member and row ordinal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import ijson
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_text(row: dict) -> str:
    profile = row.get("profile") or {}
    return " ".join(str(profile.get(key) or "") for key in
                    ("name", "screen_name", "description", "location"))


def tweet_text(row: dict) -> str:
    return " ".join(str(item or "") for item in (row.get("tweet") or []))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.width <= 0 or args.batch_size <= 0:
        raise ValueError("positive width and batch size required")
    with np.load(args.bundle, allow_pickle=False) as bundle:
        ids = bundle["ids"].astype(str)
        splits = bundle["split"].astype(str)
        members = bundle["members"].astype(str)
        ordinals = bundle["ordinals"].astype(np.int64)
    n = len(ids)
    matrix = np.zeros((n, args.width * 2), dtype=np.float32)
    profile_hasher = HashingVectorizer(
        n_features=args.width, alternate_sign=False, norm="l2",
        ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
    tweet_hasher = HashingVectorizer(
        n_features=args.width, alternate_sign=False, norm="l2",
        ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
    cursor = 0
    with zipfile.ZipFile(args.zip) as archive:
        for split in ("train", "dev", "test", "support"):
            member = f"Twibot-20/{split}.json"
            with archive.open(member) as stream:
                iterator = enumerate(ijson.items(stream, "item"))
                while True:
                    block = []
                    for _ in range(args.batch_size):
                        try:
                            block.append(next(iterator))
                        except StopIteration:
                            break
                    if not block:
                        break
                    end = cursor + len(block)
                    if end > n:
                        raise ValueError("more text rows than graph accounts")
                    for offset, (ordinal, row) in enumerate(block):
                        pos = cursor + offset
                        if (str(row["ID"]) != ids[pos] or splits[pos] != split
                                or members[pos] != member or ordinals[pos] != ordinal):
                            raise ValueError(f"raw text provenance differs at {pos}")
                    profile = profile_hasher.transform(
                        profile_text(row) for _, row in block)
                    tweet = tweet_hasher.transform(
                        tweet_text(row) for _, row in block)
                    matrix[cursor:end, :args.width] = profile.toarray().astype(np.float32)
                    matrix[cursor:end, args.width:] = tweet.toarray().astype(np.float32)
                    cursor = end
            print(json.dumps({"split": split, "rows_done": cursor}), flush=True)
    if cursor != n or not np.all(np.isfinite(matrix)):
        raise ValueError("text rows or values differ from graph population")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, values=matrix, ids=ids)
    manifest = {
        "source_archive_sha256": sha256(args.zip),
        "native_graph_bundle_sha256": sha256(args.bundle),
        "nodes": n, "profile_width": args.width, "tweet_width": args.width,
        "feature_family": "fixed_word_unigram_bigram_hash_l2",
        "raw_locator": "archive member + row ordinal + account ID + profile field or tweet ordinal",
        "label_access": "text extraction reads no label",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": manifest}), flush=True)


if __name__ == "__main__":
    main()
