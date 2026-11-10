"""Independently recompute all follow-neighbor profile cells from the raw ZIP."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import zipfile

import ijson
import numpy as np

from .twibot20_neighbor_profile_facts import FIELDS, FACT_NAMES, sha256


def _field(profile: dict, name: str) -> float:
    raw = profile.get(name)
    if raw is None or str(raw).strip() == "":
        return float("nan")
    if name in ("verified", "default_profile_image"):
        item = str(raw).strip().lower()
        if item not in ("true", "false", "1", "0"):
            raise ValueError("invalid source Boolean")
        return float(item in ("true", "1"))
    number = float(raw)
    if number < 0 or not np.isfinite(number):
        raise ValueError("invalid source count")
    return float(np.log1p(number))


def audit(archive: Path, base_facts: Path, neighbor_facts: Path) -> dict:
    manifest = json.loads(neighbor_facts.with_suffix(".json").read_text(
        encoding="utf-8"))
    if (manifest["source_archive_sha256"] != sha256(archive)
            or manifest["base_facts_sha256"] != sha256(base_facts)
            or manifest["facts_sha256"] != sha256(neighbor_facts)):
        raise ValueError("source or cache hash differs")
    with np.load(neighbor_facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str).tolist()
        names = cache["names"].astype(str).tolist()
        actual = cache["values"].astype(float)
        actual_support = cache["support"].astype(int)
    with np.load(base_facts, allow_pickle=False) as base:
        if (ids != base["ids"].astype(str).tolist()
                or names != list(FACT_NAMES)):
            raise ValueError("source population or vocabulary differs")
    attributes = {}
    out_neighbors: dict[str, set[str]] = defaultdict(set)
    in_neighbors: dict[str, set[str]] = defaultdict(set)
    with zipfile.ZipFile(archive) as zipped:
        for part in ("train", "dev", "test", "support"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for row in ijson.items(stream, "item"):
                    uid = str(row["ID"])
                    if uid in attributes:
                        raise ValueError("duplicate source account")
                    attributes[uid] = tuple(_field(row.get("profile") or {}, field)
                                            for field in FIELDS)
                    if part == "support":
                        continue
                    neighbor = row.get("neighbor") or {}
                    for other in neighbor.get("following") or []:
                        target = str(other)
                        if target != uid:
                            out_neighbors[uid].add(target)
                            in_neighbors[target].add(uid)
                    for other in neighbor.get("follower") or []:
                        source = str(other)
                        if source != uid:
                            out_neighbors[source].add(uid)
                            in_neighbors[uid].add(source)
    if any(other not in attributes for neighbors in
           list(out_neighbors.values()) + list(in_neighbors.values())
           for other in neighbors):
        raise ValueError("edge endpoint missing in raw ZIP")
    expected = np.full_like(actual, np.nan)
    support = np.zeros_like(actual_support)
    for i, uid in enumerate(ids):
        for direction, relation in enumerate((out_neighbors, in_neighbors)):
            neighbors = relation.get(uid, set())
            for j in range(len(FIELDS)):
                observed = [attributes[other][j] for other in neighbors
                            if np.isfinite(attributes[other][j])]
                column = direction * len(FIELDS) + j
                support[i, column] = len(observed)
                if observed:
                    expected[i, column] = sum(observed) / len(observed)
    same_missing = np.array_equal(np.isnan(expected), np.isnan(actual))
    difference = np.abs(np.nan_to_num(expected) - np.nan_to_num(actual))
    max_diff = float(np.max(difference))
    support_match = bool(np.array_equal(support, actual_support))
    if not same_missing or max_diff > 1e-10 or not support_match:
        raise ValueError("source recomputation differs from cached facts")
    return {
        "dataset": "twibot20",
        "status": "verified_all_follow_neighbor_profile_fact_cells_against_official_zip",
        "provenance_level": "official_follow_declarations_and_raw_neighbor_profiles",
        "source_archive_sha256": manifest["source_archive_sha256"],
        "fact_cache_sha256": manifest["facts_sha256"],
        "account_count": len(ids), "fact_count": len(names),
        "fact_cells_including_missing": int(actual.size),
        "nonmissing_fact_cells": int(np.isfinite(actual).sum()),
        "max_absolute_numeric_difference": max_diff,
        "support_counts_match": support_match,
        "label_fields_read": False,
    }
