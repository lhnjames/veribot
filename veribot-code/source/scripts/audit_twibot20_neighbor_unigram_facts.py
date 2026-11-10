"""Independently replay every labeled owner's native-neighbor word-share fact."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import ijson
import numpy as np

from train_v3_twibot20 import sha256


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw_manifest = json.loads((root / "data/raw/twibot20/manifest.json")
                              .read_text(encoding="utf-8"))
    incidence_path = root / "data/processed/twibot20_v3/neighbor_active_unigram_incidence.npz"
    facts_path = root / "data/processed/twibot20_v3/neighbor_active_unigram_facts.npz"
    facts_manifest = json.loads(facts_path.with_suffix(".json")
                                .read_text(encoding="utf-8"))
    if (facts_manifest["source_archive_sha256"] != raw_manifest["archive_sha256"]
            or facts_manifest["neighbor_incidence_sha256"] != sha256(incidence_path)
            or facts_manifest["fact_cache_sha256"] != sha256(facts_path)):
        raise ValueError("neighbor fact source binding differs")
    with np.load(incidence_path, allow_pickle=False) as source:
        source_ids = source["ids"].astype(str)
        source_counts = source["tweet_counts"].astype(int)
        incidence = source["incidence"].astype(np.uint8)
        words = source["vocabulary"].astype(str)
    with np.load(facts_path, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        parts = cache["split"].astype(str)
        positions = cache["positions"].astype(int)
        names = cache["names"].astype(str).tolist()
        values = cache["values"].astype(np.float32)
        supports = cache["supports"].astype(int)
    if (values.shape != (11826, 218) or supports.shape != (11826, 2)
            or names != [f"native_{direction}_neighbor_{word}_tweet_presence_share"
                         for direction in ("following", "follower") for word in words]):
        raise ValueError("neighbor fact population or vocabulary differs")
    lookup = {account_id: index for index, account_id in enumerate(source_ids)}
    maximum_difference = 0.0
    accounts = 0
    with zipfile.ZipFile(raw_manifest["archive_path"]) as zipped:
        for part in ("train", "dev", "test"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    if (ids[accounts] != str(row["ID"]) or parts[accounts] != part
                            or positions[accounts] != position):
                        raise ValueError("official owner source position differs")
                    for direction_index, direction in enumerate(("following", "follower")):
                        neighbor_ids = set(map(str, (row.get("neighbor") or {})
                                               .get(direction) or ()))
                        selected = sorted(lookup[neighbor_id]
                                          for neighbor_id in neighbor_ids
                                          if neighbor_id in lookup and
                                          source_counts[lookup[neighbor_id]] > 0)
                        start = direction_index * 109
                        observed = values[accounts, start:start + 109]
                        if supports[accounts, direction_index] != len(selected):
                            raise ValueError("neighbor source support differs")
                        if selected:
                            expected = incidence[selected].sum(axis=0,
                                                                 dtype=np.int32) / len(selected)
                            if np.isnan(observed).any():
                                raise ValueError("available neighbor fact is missing")
                            difference = float(np.max(np.abs(
                                observed - expected.astype(np.float32))))
                            maximum_difference = max(maximum_difference, difference)
                            if difference > 1e-7:
                                raise ValueError("neighbor word share differs")
                        elif not np.isnan(observed).all():
                            raise ValueError("unavailable neighbor fact has a value")
                    accounts += 1
    if accounts != len(ids):
        raise ValueError("official owner account count differs")
    certificate = {
        "dataset": "twibot20",
        "status": "verified_all_labeled_owner_neighbor_word_share_cells",
        "accounts_checked": accounts,
        "fact_cells_checked": accounts * len(names),
        "maximum_value_difference": maximum_difference,
        "source_archive_sha256": raw_manifest["archive_sha256"],
        "incidence_cache_sha256": sha256(incidence_path),
        "fact_cache_sha256": sha256(facts_path),
        "fact_manifest_sha256": sha256(facts_path.with_suffix(".json")),
        "operation": "independent per-owner unique native-follow IDs, matched official tweet-bearing neighbors, summed token indicators divided by support",
    }
    destination = root / "research/certificates/twibot20_neighbor_unigram_fact_full_aggregate_audit.json"
    destination.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(certificate["fact_cells_checked"], maximum_difference, flush=True)


if __name__ == "__main__":
    main()
