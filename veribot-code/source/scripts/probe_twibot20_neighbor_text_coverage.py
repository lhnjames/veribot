"""Count official native-follow neighbors reachable in labeled tweet records."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import ijson
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "data/raw/twibot20/manifest.json")
                          .read_text(encoding="utf-8"))
    archive_path = Path(manifest["archive_path"])
    with np.load(root / "data/processed/twibot20_v3/facts.npz",
                 allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        parts = cache["split"].astype(str)
    labeled = set(ids)
    train_dev = set(ids[np.isin(parts, ["train", "dev"])])
    targets = set()
    edges = {direction: 0 for direction in ("following", "follower")}
    available = {direction: 0 for direction in edges}
    owners = {direction: 0 for direction in edges}
    with zipfile.ZipFile(archive_path) as zipped:
        for part in ("train", "dev"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for row in ijson.items(stream, "item"):
                    neighbor = row.get("neighbor") or {}
                    for direction in edges:
                        unique = set(map(str, neighbor.get(direction) or ()))
                        if unique:
                            owners[direction] += 1
                        edges[direction] += len(unique)
                        available[direction] += len(unique & train_dev)
                        targets.update(unique)
    result = {
        "scope": "official_train_dev_neighbor_text_source_coverage",
        "archive_sha256_from_manifest": manifest["archive_sha256"],
        "owner_count": len(train_dev),
        "unique_neighbor_ids": len(targets),
        "unique_neighbor_ids_in_all_labeled_splits": len(targets & labeled),
        "unique_neighbor_ids_in_train_dev": len(targets & train_dev),
        "unique_neighbor_ids_outside_all_labeled_splits": len(targets - labeled),
        "unique_neighbor_rows_by_direction": edges,
        "rows_to_train_dev_neighbors_by_direction": available,
        "accounts_with_nonempty_neighbor_list_by_direction": owners,
    }
    path = root / "research/twibot20_neighbor_text_coverage_2026-11-03.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
