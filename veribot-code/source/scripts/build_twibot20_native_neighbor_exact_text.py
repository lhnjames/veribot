"""Stream official ZIP once to build literal text-sharing facts on native edges."""

from __future__ import annotations

from array import array
import gzip
import hashlib
import json
import re
import struct
import sys
import unicodedata
import zipfile
from pathlib import Path

import ijson
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256


PARTS = ("train", "dev", "test", "support")
NAMES = (
    "native_neighbor_exact_shared_text_fraction",
    "native_neighbor_exact_shared_text_neighbor_log1p",
    "native_neighbor_exact_shared_text_distinct_log1p",
)


def fingerprint(text: object) -> tuple[int, int]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if len(normalized) < 30 or sum(char.isalpha() for char in normalized) < 15:
        return 0, 0
    return struct.unpack("<QQ", hashlib.blake2b(
        normalized.encode("utf-8"), digest_size=16).digest())


def pairs(values: array):
    for index in range(0, len(values), 2):
        pair = values[index], values[index + 1]
        if pair != (0, 0):
            yield index // 2, pair


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw_manifest = json.loads((root / "data/raw/twibot20/manifest.json")
                              .read_text(encoding="utf-8"))
    archive = Path(raw_manifest["archive_path"])
    archive_hash = sha256(archive)
    if archive_hash != raw_manifest["archive_sha256"]:
        raise ValueError("TwiBot-20 official ZIP differs")
    owner_rows: list[tuple[str, str, int, set[str]]] = []
    hashed: dict[str, array] = {}
    locator: dict[str, tuple[str, int]] = {}
    with zipfile.ZipFile(archive) as zipped:
        for part in PARTS[:3]:
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    uid = str(row["ID"])
                    if uid in hashed:
                        raise ValueError("duplicate labeled account ID")
                    neighbor = row.get("neighbor") or {}
                    linked = set(map(str, neighbor.get("following") or ()))
                    linked.update(map(str, neighbor.get("follower") or ()))
                    linked.discard(uid)
                    owner_rows.append((uid, part, position, linked))
                    tweet_hashes = array("Q")
                    for tweet in row.get("tweet") or ():
                        tweet_hashes.extend(fingerprint(tweet))
                    hashed[uid] = tweet_hashes
                    locator[uid] = part, position
            print(json.dumps({"part": part, "owner_rows": len(owner_rows)}), flush=True)
        targets = set().union(*(row[3] for row in owner_rows)) - set(hashed)
        found_support = 0
        with zipped.open("Twibot-20/support.json") as stream:
            for position, row in enumerate(ijson.items(stream, "item")):
                uid = str(row["ID"])
                if uid not in targets:
                    continue
                if uid in hashed:
                    raise ValueError("duplicate support account ID")
                tweet_hashes = array("Q")
                for tweet in row.get("tweet") or ():
                    tweet_hashes.extend(fingerprint(tweet))
                hashed[uid] = tweet_hashes
                locator[uid] = "support", position
                found_support += 1
                if found_support % 25000 == 0:
                    print(json.dumps({"support_accounts_hashed": found_support}), flush=True)
    values = np.empty((len(owner_rows), len(NAMES)), dtype=np.float64)
    examples = []
    positive_accounts = 0
    for index, (uid, part, position, linked) in enumerate(owner_rows):
        owner_pairs = list(pairs(hashed[uid]))
        own_set = {digest for _, digest in owner_pairs}
        shared: set[tuple[int, int]] = set()
        neighbor_hits = 0
        examples_this_owner = 0
        for neighbor_id in sorted(linked):
            other = hashed.get(neighbor_id)
            if other is None:
                continue
            other_pairs = list(pairs(other))
            overlap = own_set & {digest for _, digest in other_pairs}
            if not overlap:
                continue
            neighbor_hits += 1
            shared.update(overlap)
            if examples_this_owner < 5:
                digest = min(overlap)
                owner_tweet = next(j for j, value in owner_pairs if value == digest)
                neighbor_tweet = next(j for j, value in other_pairs if value == digest)
                neighbor_part, neighbor_position = locator[neighbor_id]
                examples.append({
                    "account_id": uid, "owner_member": part,
                    "owner_position": position, "owner_tweet_index": owner_tweet,
                    "neighbor_id": neighbor_id,
                    "neighbor_member": neighbor_part,
                    "neighbor_position": neighbor_position,
                    "neighbor_tweet_index": neighbor_tweet,
                    "normalized_text_blake2b_128": struct.pack("<QQ", *digest).hex(),
                })
                examples_this_owner += 1
        if shared:
            positive_accounts += 1
        values[index] = (
            (sum(digest in shared for _, digest in owner_pairs) / len(owner_pairs)
             if owner_pairs else float("nan")),
            np.log1p(neighbor_hits), np.log1p(len(shared)),
        )
        if (index + 1) % 2000 == 0:
            print(json.dumps({"owners_computed": index + 1,
                              "owners_with_shared_text": positive_accounts}), flush=True)
    destination = root / "data/processed/twibot20_v3/native_neighbor_exact_text.npz"
    np.savez_compressed(destination,
                        ids=np.asarray([row[0] for row in owner_rows]),
                        split=np.asarray([row[1] for row in owner_rows]),
                        positions=np.asarray([row[2] for row in owner_rows]),
                        names=np.asarray(NAMES), values=values)
    examples_path = destination.with_name("native_neighbor_exact_text_examples.jsonl.gz")
    with gzip.open(examples_path, "wt", encoding="utf-8") as stream:
        for row in examples:
            stream.write(json.dumps(row) + "\n")
    summary = {
        "dataset": "twibot20",
        "scope": "native_follow_neighbor_exact_normalized_tweet_text",
        "source_archive_sha256": archive_hash,
        "owner_accounts": len(owner_rows),
        "target_neighbor_ids": len(targets),
        "matched_support_neighbors": found_support,
        "owners_with_shared_text": positive_accounts,
        "source_fact_names": NAMES,
        "normalization": "Unicode NFKC, casefold, whitespace collapse, length >=30, alphabetic >=15",
        "duplicate_owner_tweets_counted_in_fraction_denominator": True,
        "classification_labels_used": False,
        "fact_cache_sha256": sha256(destination),
        "example_locators_sha256": sha256(examples_path),
        "example_locator_count": len(examples),
    }
    destination.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
