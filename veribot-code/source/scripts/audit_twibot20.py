"""Audit the official TwiBot-20 labelled splits before HGT retraining."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "data/raw/twibot20"


def main() -> None:
    report: dict = {"splits": {}}
    id_sets: dict[str, set[str]] = {}
    for name in ("train", "dev", "test"):
        path = ROOT / f"{name}.json"
        with path.open(encoding="utf-8") as source:
            rows = json.load(source)
        ids = [str(row["ID"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate IDs within {name}")
        id_sets[name] = set(ids)
        labels = Counter(str(row["label"]) for row in rows)
        report["splits"][name] = {"accounts": len(rows), "labels": dict(labels)}
        if name == "train":
            edges: set[tuple[str, str]] = set()
            isolated = 0
            for row in rows:
                uid = str(row["ID"])
                neighbor = row.get("neighbor")
                if neighbor is None:
                    isolated += 1
                    continue
                edges.update((uid, str(dst)) for dst in neighbor.get("following") or []
                             if str(dst) in id_sets[name])
                edges.update((str(src), uid) for src in neighbor.get("follower") or []
                             if str(src) in id_sets[name])
            report["splits"][name]["follows_edges_deduplicated"] = len(edges)
            report["splits"][name]["null_neighbor_accounts"] = isolated
        print(name, report["splits"][name], flush=True)
    overlaps = {f"{a}_{b}": len(id_sets[a] & id_sets[b]) for a, b in
                (("train", "dev"), ("train", "test"), ("dev", "test"))}
    report["cross_split_id_overlap"] = overlaps
    if any(overlaps.values()):
        raise ValueError(f"IDs overlap across official splits: {overlaps}")
    with (ROOT / "data_audit.json").open("w", encoding="utf-8") as dest:
        json.dump(report, dest, indent=2)
    print("Cross-split overlap:", overlaps, flush=True)


if __name__ == "__main__":
    main()
