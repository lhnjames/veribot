"""Independent full-population audit of TwiBot-20's 22 cached facts.

The verifier streams only the official labeled ZIP members. It does not use
the extractor's account_facts function or any Bot/Human labels.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import zipfile

import ijson
import numpy as np

from .twibot20_facts import FACT_NAMES
from .twibot20_source_certificate import sha256


PARTS = ("train", "dev", "test")
DIRECT_NAMES = FACT_NAMES[:-3]


def _numeric(value: object) -> float:
    return float(str(value).strip()) if value is not None and str(value).strip() else float("nan")


def _boolean(value: object) -> float:
    if value is None or not str(value).strip():
        return float("nan")
    normalized = str(value).strip().lower()
    if normalized not in ("true", "false", "1", "0"):
        raise ValueError(f"unknown published Boolean {value!r}")
    return float(normalized in ("true", "1"))


def direct_facts(row: dict) -> np.ndarray:
    """Requery field and tweet operations without calling the training extractor."""
    profile = row.get("profile") or {}
    tweets = [str(text or "") for text in (row.get("tweet") or [])]
    count = len(tweets)
    def fraction(hits: int) -> float:
        return hits / count if count else float("nan")
    neighbor = row.get("neighbor")
    values = [
        *(_numeric(profile.get(name)) for name in (
            "followers_count", "friends_count", "statuses_count",
            "listed_count", "favourites_count")),
        *(_boolean(profile.get(name)) for name in (
            "verified", "protected", "default_profile", "default_profile_image")),
        float(count),
        fraction(sum(text.lstrip().startswith("RT @") for text in tweets)),
        fraction(sum("http://" in text or "https://" in text for text in tweets)),
        fraction(sum(re.search(r"(?<!\w)#\w", text) is not None for text in tweets)),
        fraction(sum(re.search(r"(?<!\w)@\w", text) is not None for text in tweets)),
        float(sum(map(len, tweets)) / count) if count else float("nan"),
        fraction(len(set(tweets))),
        float(len(str(profile.get("description") or ""))),
        float(len(neighbor.get("following") or [])) if neighbor is not None else float("nan"),
        float(len(neighbor.get("follower") or [])) if neighbor is not None else float("nan"),
    ]
    if len(values) != len(DIRECT_NAMES):
        raise AssertionError("direct fact vocabulary and source operations differ")
    return np.asarray(values, dtype=float)


def declared_edges(row: dict) -> set[tuple[str, str]]:
    uid = str(row["ID"])
    neighbor = row.get("neighbor") or {}
    outgoing = {(uid, str(other)) for other in neighbor.get("following") or []
                if str(other) != uid}
    incoming = {(str(other), uid) for other in neighbor.get("follower") or []
                if str(other) != uid}
    return outgoing | incoming


def audit(archive: Path, facts_path: Path) -> dict:
    source_manifest = json.loads(facts_path.with_suffix(".json").read_text(
        encoding="utf-8"))
    archive_hash = sha256(archive)
    if archive_hash != source_manifest["archive_sha256"]:
        raise ValueError("official ZIP differs from fact-cache manifest")
    with np.load(facts_path, allow_pickle=False) as cached:
        ids = cached["ids"].astype(str).tolist()
        splits = cached["split"].astype(str).tolist()
        positions = cached["positions"].astype(int).tolist()
        names = cached["names"].astype(str).tolist()
        stored = cached["values"].copy()
    if names != list(FACT_NAMES) or stored.shape != (len(ids), len(names)):
        raise ValueError("cached population or feature vocabulary differs")

    raw_rows = []
    edges: set[tuple[str, str]] = set()
    index = 0
    split_counts = Counter()
    with zipfile.ZipFile(archive) as zipped:
        for part in PARTS:
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for ordinal, row in enumerate(ijson.items(stream, "item")):
                    if index >= len(ids) or str(row["ID"]) != ids[index] or \
                            splits[index] != part or positions[index] != ordinal:
                        raise ValueError("ZIP account or split locator differs from fact cache")
                    raw_rows.append(direct_facts(row))
                    edges.update(declared_edges(row))
                    split_counts[part] += 1
                    index += 1
    if index != len(ids) or dict(split_counts) != source_manifest["split_counts"]:
        raise ValueError("official labeled split population differs")
    if len(edges) != source_manifest["directed_follow_edges"]:
        raise ValueError("deduplicated published follow edge count differs")

    outgoing = Counter(source for source, _ in edges)
    incoming = Counter(target for _, target in edges)
    reciprocal = Counter(source for source, target in edges
                         if (target, source) in edges)
    graph = np.asarray([[outgoing[uid], incoming[uid], reciprocal[uid]]
                        for uid in ids], dtype=float)
    reconstructed = np.column_stack((np.asarray(raw_rows, dtype=float), graph))
    equal = np.isclose(reconstructed, stored, rtol=0, atol=1e-10,
                       equal_nan=True)
    mismatch = {name: int(np.count_nonzero(~equal[:, j]))
                for j, name in enumerate(names)}
    if any(mismatch.values()):
        raise ValueError(f"source fact table differs from ZIP: {mismatch}")
    return {
        "dataset": "twibot20",
        "status": "verified_all_labeled_account_fact_cells_against_official_zip",
        "provenance_level": "official_profile_tweet_and_sampled_follow_declarations",
        "source_archive_sha256": archive_hash,
        "fact_cache_sha256": sha256(facts_path),
        "labeled_accounts": len(ids), "split_counts": dict(split_counts),
        "distinct_directed_follow_edges": len(edges),
        "fact_count": len(names), "verified_fact_cells": int(equal.size),
        "missing_by_fact": {name: int(np.count_nonzero(np.isnan(stored[:, j])))
                            for j, name in enumerate(names)},
        "mismatches_by_fact": mismatch,
        "support_member_scanned": False,
        "label_fields_read_for_fact_recomputation": False,
    }
