"""Recompute selected follow-neighbor profile facts from official raw records."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import zipfile

import ijson
import numpy as np

from .twibot20_neighbor_profile_audit import _field
from .twibot20_neighbor_profile_facts import FACT_NAMES, FIELDS, sha256


def certificates(archive: Path, facts_path: Path, account_id: str,
                 fact_names: list[str]) -> list[dict]:
    if any(name not in FACT_NAMES for name in fact_names):
        raise ValueError("requested fact is outside follow-neighbor vocabulary")
    manifest = json.loads(facts_path.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest["source_archive_sha256"] != sha256(archive) or \
            manifest["facts_sha256"] != sha256(facts_path):
        raise ValueError("source ZIP or fact cache differs")
    with np.load(facts_path, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        matches = np.flatnonzero(ids == str(account_id))
        if len(matches) != 1:
            raise ValueError("account absent or duplicated")
        index = int(matches[0])
        names = cache["names"].astype(str).tolist()
        values = cache["values"][index].astype(float)
        support = cache["support"][index].astype(int)
    if names != list(FACT_NAMES):
        raise ValueError("fact cache vocabulary differs")
    declarations: dict[tuple[str, str], list[str]] = defaultdict(list)
    with zipfile.ZipFile(archive) as zipped:
        for part in ("train", "dev", "test"):
            member = f"Twibot-20/{part}.json"
            with zipped.open(member) as stream:
                for ordinal, row in enumerate(ijson.items(stream, "item")):
                    uid = str(row["ID"])
                    neighbor = row.get("neighbor") or {}
                    for field, reversed_edge in (("following", False),
                                                 ("follower", True)):
                        for position, other in enumerate(neighbor.get(field) or []):
                            edge = ((str(other), uid) if reversed_edge
                                    else (uid, str(other)))
                            if str(other) != uid and str(account_id) in edge:
                                declarations[edge].append(
                                    f"{member}[{ordinal}].neighbor.{field}[{position}]")
        following = {b for a, b in declarations if a == str(account_id)}
        follower = {a for a, b in declarations if b == str(account_id)}
        needed = following | follower
        raw_profiles: dict[str, tuple[dict, str]] = {}
        for part in ("train", "dev", "test", "support"):
            if len(raw_profiles) == len(needed):
                break
            member = f"Twibot-20/{part}.json"
            with zipped.open(member) as stream:
                for ordinal, row in enumerate(ijson.items(stream, "item")):
                    uid = str(row["ID"])
                    if uid in needed:
                        raw_profiles[uid] = (
                            row.get("profile") or {}, f"{member}[{ordinal}].profile")
        if set(raw_profiles) != needed:
            raise ValueError("neighbor profile source missing")
    output = []
    for name in fact_names:
        column = names.index(name)
        direction, field = name.split("_neighbor_mean_", 1)
        neighbors = following if direction == "following" else follower
        source_items = []
        for other in sorted(neighbors):
            profile, locator = raw_profiles[other]
            value = _field(profile, field)
            if not np.isfinite(value):
                continue
            edge = ((str(account_id), other) if direction == "following"
                    else (other, str(account_id)))
            source_items.append({
                "neighbor_id": other,
                "edge_declarations": sorted(declarations[edge]),
                "profile_query": f"{locator}.{field}",
                "transformed_value": value,
            })
        computed = (float(np.mean([item["transformed_value"]
                                   for item in source_items]))
                    if source_items else float("nan"))
        cached = float(values[column])
        matched = ((np.isnan(computed) and np.isnan(cached)) or
                   (np.isfinite(computed) and np.isfinite(cached) and
                    abs(computed - cached) <= 1e-12))
        if len(source_items) != int(support[column]) or not matched:
            raise ValueError(f"source value or support count differs: {name}")
        output.append({
            "dataset": "twibot20", "account_id": str(account_id),
            "fact_name": name,
            "source_value": computed if np.isfinite(computed) else None,
            "cached_value": cached if np.isfinite(cached) else None,
            "support": len(source_items),
            "source_query": source_items,
            "source_archive_sha256": manifest["source_archive_sha256"],
            "fact_cache_sha256": manifest["facts_sha256"],
            "status": "verified",
        })
    return output
