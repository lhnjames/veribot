"""Aggregate source-local neighbor word incidence over official native follow lists."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import ijson
import numpy as np

from train_v3_twibot20 import sha256


DIRECTIONS = ("following", "follower")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    base_path = root / "data/processed/twibot20_v3/facts.npz"
    incidence_path = root / "data/processed/twibot20_v3/neighbor_active_unigram_incidence.npz"
    incidence_manifest = json.loads(incidence_path.with_suffix(".json")
                                    .read_text(encoding="utf-8"))
    raw_manifest = json.loads((root / "data/raw/twibot20/manifest.json")
                              .read_text(encoding="utf-8"))
    if (incidence_manifest["fact_cache_sha256"] != sha256(incidence_path)
            or incidence_manifest["source_archive_sha256"] !=
            raw_manifest["archive_sha256"]):
        raise ValueError("neighbor word incidence differs from official ZIP")
    with np.load(base_path, allow_pickle=False) as base:
        owner_ids = base["ids"].astype(str)
        parts = base["split"].astype(str)
        positions = base["positions"].astype(int)
    with np.load(incidence_path, allow_pickle=False) as incidence_cache:
        source_ids = incidence_cache["ids"].astype(str)
        source_tweet_count = incidence_cache["tweet_counts"].astype(int)
        active = incidence_cache["vocabulary"].astype(str)
        incidence = incidence_cache["incidence"].astype(np.uint8)
    if (len(active) != 109 or incidence.shape != (len(source_ids), 109)
            or len(set(source_ids)) != len(source_ids)):
        raise ValueError("source word incidence population differs")
    lookup = {account_id: index for index, account_id in enumerate(source_ids)}
    names = [f"native_{direction}_neighbor_{word}_tweet_presence_share"
             for direction in DIRECTIONS for word in active]
    values = np.full((len(owner_ids), len(names)), np.nan, dtype=np.float32)
    supports = np.zeros((len(owner_ids), len(DIRECTIONS)), dtype=np.int16)
    found = {direction: 0 for direction in DIRECTIONS}
    with zipfile.ZipFile(raw_manifest["archive_path"]) as zipped:
        account = 0
        for part in ("train", "dev", "test"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    if (account >= len(owner_ids) or owner_ids[account] !=
                            str(row["ID"]) or parts[account] != part
                            or positions[account] != position):
                        raise ValueError("official owner account order differs")
                    neighbor = row.get("neighbor") or {}
                    for direction_index, direction in enumerate(DIRECTIONS):
                        neighbor_ids = set(map(str, neighbor.get(direction) or ()))
                        rows = [lookup[account_id] for account_id in neighbor_ids
                                if account_id in lookup and
                                source_tweet_count[lookup[account_id]] > 0]
                        if rows:
                            supports[account, direction_index] = len(rows)
                            start = direction_index * len(active)
                            values[account, start:start + len(active)] = (
                                incidence[rows].mean(axis=0)).astype(np.float32)
                            found[direction] += 1
                    account += 1
            print(part, account, flush=True)
    if account != len(owner_ids):
        raise ValueError("official labeled account population differs")
    destination = root / "data/processed/twibot20_v3/neighbor_active_unigram_facts.npz"
    np.savez_compressed(destination, ids=owner_ids, split=parts, positions=positions,
                        names=np.asarray(names), values=values, supports=supports)
    manifest = {
        "dataset": "twibot20", "scope": "native_follow_neighbor_raw_tweet_word_share",
        "source_archive_sha256": raw_manifest["archive_sha256"],
        "base_facts_sha256": sha256(base_path),
        "neighbor_incidence_sha256": sha256(incidence_path),
        "fact_cache_sha256": sha256(destination),
        "owner_accounts": len(owner_ids), "fact_count": len(names),
        "active_unigram_count": len(active),
        "accounts_with_source_tweet_neighbors_by_direction": found,
        "fact_operation": "fraction of distinct published native-follow neighbors with at least one sampled source tweet containing a frozen active unigram; missing when none",
        "support_definition": "number of distinct published neighbors with at least one sampled tweet",
        "classification_label_field_used": False,
        "test_labels_used_for_feature_definition": False,
    }
    destination.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(found), flush=True)


if __name__ == "__main__":
    main()
