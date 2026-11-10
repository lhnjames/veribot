"""Label-free pilot of repeated tweet text across released follow neighbors."""

from __future__ import annotations

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


PARTS = ("train", "dev", "test", "support")


def normalized(text: object) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", value).strip()


def fingerprint(text: object) -> bytes | None:
    value = normalized(text)
    if len(value) < 30 or sum(character.isalpha() for character in value) < 15:
        return None
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "data/raw/twibot20/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = Path(manifest["archive_path"])
    if sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("TwiBot-20 official archive differs from source audit")
    candidates = []
    with zipfile.ZipFile(archive) as zipped:
        for part in ("train", "dev"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    uid = str(row["ID"])
                    neighbor = row.get("neighbor") or {}
                    linked = set(map(str, neighbor.get("following") or ()))
                    linked.update(map(str, neighbor.get("follower") or ()))
                    linked.discard(uid)
                    order = hashlib.sha256(uid.encode("utf-8")).digest()
                    candidates.append((order, uid, part, position, linked))
        selected = sorted(candidates)[:500]
        owner_ids = {row[1] for row in selected}
        targets = owner_ids | set().union(*(row[4] for row in selected))
        observed: dict[str, set[bytes]] = {}
        tweet_counts: dict[str, int] = {}
        for part in PARTS:
            matched = 0
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for row in ijson.items(stream, "item"):
                    uid = str(row["ID"])
                    if uid not in targets:
                        continue
                    if uid in observed:
                        raise ValueError("duplicate account ID in official ZIP")
                    tweets = row.get("tweet") or ()
                    digests = {digest for text in tweets
                               if (digest := fingerprint(text)) is not None}
                    observed[uid] = digests
                    tweet_counts[uid] = len(tweets)
                    matched += 1
            print(json.dumps({"part": part, "targets_found": matched}), flush=True)
    rows = []
    for _, uid, part, position, neighbor_ids in selected:
        own = observed.get(uid, set())
        links_with_text = 0
        links_with_match = 0
        matched_hashes: set[bytes] = set()
        for linked in neighbor_ids:
            other = observed.get(linked)
            if not other:
                continue
            links_with_text += 1
            shared = own & other
            if shared:
                links_with_match += 1
                matched_hashes.update(shared)
        rows.append({"account_id": uid, "part": part, "source_position": position,
                     "declared_neighbor_count": len(neighbor_ids),
                     "neighbors_with_eligible_text": links_with_text,
                     "distinct_eligible_own_texts": len(own),
                     "neighbors_sharing_exact_normalized_text_hash": links_with_match,
                     "distinct_shared_normalized_text_hashes": len(matched_hashes)})
    result = {
        "scope": "label_free_native_follow_neighbor_exact_text_feasibility_pilot",
        "source_archive_sha256": manifest["archive_sha256"],
        "selection": "500 train/dev owner IDs with lexicographically smallest SHA-256 digest",
        "normalization": "Unicode NFKC, casefold, whitespace collapse, length >=30, alphabetic characters >=15",
        "classification_labels_used": False,
        "test_account_labels_used": False,
        "owner_count": len(rows),
        "target_accounts": len(targets),
        "matched_source_accounts": len(observed),
        "owners_with_observed_neighbor_text": sum(row["neighbors_with_eligible_text"] > 0
                                                  for row in rows),
        "owners_with_shared_text_hash": sum(row["neighbors_sharing_exact_normalized_text_hash"] > 0
                                            for row in rows),
        "total_neighbor_links_with_text": sum(row["neighbors_with_eligible_text"]
                                               for row in rows),
        "total_neighbor_links_with_shared_text_hash": sum(
            row["neighbors_sharing_exact_normalized_text_hash"] for row in rows),
        "rows": rows,
    }
    path = root / "output/v3_twibot20_native_neighbor_exact_text/pilot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}),
          flush=True)


if __name__ == "__main__":
    main()
