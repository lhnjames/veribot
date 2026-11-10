"""Independently check every saved match against both raw tweets and native links."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

import ijson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256


def normalize(value: object) -> str:
    # Independent spelling of the same explicit normalization contract.
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    archive_manifest = json.loads((root / "data/raw/twibot20/manifest.json")
                                  .read_text(encoding="utf-8"))
    archive = Path(archive_manifest["archive_path"])
    cache = root / "data/processed/twibot20_v3/native_neighbor_exact_text.npz"
    manifest = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
    examples_path = cache.with_name("native_neighbor_exact_text_examples.jsonl.gz")
    if (manifest["source_archive_sha256"] != sha256(archive)
            or manifest["example_locators_sha256"] != sha256(examples_path)):
        raise ValueError("TwiBot source archive or match locator cache differs")
    with gzip.open(examples_path, "rt", encoding="utf-8") as stream:
        examples = [json.loads(line) for line in stream]
    wanted: dict[str, dict[int, dict]] = {part: {} for part in
                                         ("train", "dev", "test", "support")}
    for row in examples:
        for role in ("owner", "neighbor"):
            part, position, uid, tweet_index = (
                row[f"{role}_member"], int(row[f"{role}_position"]),
                row["account_id" if role == "owner" else "neighbor_id"],
                int(row[f"{role}_tweet_index"]))
            record = wanted[part].setdefault(position, {"account_id": uid,
                                                         "tweet_indices": set(),
                                                         "is_owner": False})
            if record["account_id"] != uid:
                raise ValueError("locator position maps to multiple account IDs")
            record["tweet_indices"].add(tweet_index)
            record["is_owner"] |= role == "owner"
    observed: dict[tuple[str, int, int], str] = {}
    links: dict[str, set[str]] = {}
    with zipfile.ZipFile(archive) as zipped:
        for part, positions in wanted.items():
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, raw in enumerate(ijson.items(stream, "item")):
                    request = positions.get(position)
                    if request is None:
                        continue
                    uid = str(raw["ID"])
                    if uid != request["account_id"]:
                        raise ValueError("saved account locator differs from official ZIP")
                    tweets = raw.get("tweet") or ()
                    for tweet_index in request["tweet_indices"]:
                        if not 0 <= tweet_index < len(tweets):
                            raise ValueError("saved tweet index is absent")
                        observed[(part, position, tweet_index)] = normalize(tweets[tweet_index])
                    if request["is_owner"]:
                        neighbor = raw.get("neighbor") or {}
                        linked = set(map(str, neighbor.get("following") or ()))
                        linked.update(map(str, neighbor.get("follower") or ()))
                        links[uid] = linked
    retweets = urls = 0
    for row in examples:
        owner = observed[(row["owner_member"], row["owner_position"],
                          row["owner_tweet_index"])]
        neighbor = observed[(row["neighbor_member"], row["neighbor_position"],
                             row["neighbor_tweet_index"])]
        if (owner != neighbor or row["neighbor_id"] not in links[row["account_id"]]
                or len(owner) < 30 or sum(char.isalpha() for char in owner) < 15
                or hashlib.blake2b(owner.encode("utf-8"), digest_size=16).hexdigest()
                != row["normalized_text_blake2b_128"]):
            raise ValueError("saved native-neighbor text match differs from raw sources")
        retweets += owner.startswith("rt @")
        urls += "http://" in owner or "https://" in owner
    certificate = {
        "scope": "all_saved_native_neighbor_text_example_pairs_raw_source_check",
        "source_archive_sha256": manifest["source_archive_sha256"],
        "fact_cache_sha256": manifest["fact_cache_sha256"],
        "example_locators_sha256": manifest["example_locators_sha256"],
        "exact_raw_text_and_native_edge_pairs_verified": len(examples),
        "retweet_prefixed_pairs": retweets,
        "url_containing_pairs": urls,
        "classification_labels_used": False,
        "status": "all_saved_pairs_equal_at_both_raw_tweet_positions_and_native_edge",
    }
    path = root / "research/certificates/twibot20_native_neighbor_exact_text_examples.json"
    path.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(certificate), flush=True)


if __name__ == "__main__":
    main()
