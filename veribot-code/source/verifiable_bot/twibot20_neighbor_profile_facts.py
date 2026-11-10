"""Label-blind follow-neighbor profile facts from the official TwiBot-20 ZIP."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

import ijson
import numpy as np


FIELDS = ("followers_count", "friends_count", "statuses_count",
          "verified", "default_profile_image")
FACT_NAMES = tuple(f"{direction}_neighbor_mean_{field}"
                   for direction in ("following", "follower") for field in FIELDS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile_values(profile: dict | None) -> np.ndarray:
    row = profile or {}
    values = []
    for name in FIELDS:
        value = row.get(name)
        if value is None or str(value).strip() == "":
            values.append(float("nan"))
        elif name in ("verified", "default_profile_image"):
            normalized = str(value).strip().lower()
            if normalized not in ("true", "false", "1", "0"):
                raise ValueError(f"invalid Boolean profile value: {value!r}")
            values.append(float(normalized in ("true", "1")))
        else:
            number = float(str(value).strip())
            if not np.isfinite(number) or number < 0:
                raise ValueError(f"invalid nonnegative profile value: {value!r}")
            values.append(float(np.log1p(number)))
    return np.asarray(values, dtype=np.float64)


def source_rows(archive: Path):
    with zipfile.ZipFile(archive) as zipped:
        for part in ("train", "dev", "test", "support"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for ordinal, row in enumerate(ijson.items(stream, "item")):
                    yield part, ordinal, row


def build(archive: Path, base_facts: Path, output: Path) -> dict:
    with np.load(base_facts, allow_pickle=False) as base:
        ids = base["ids"].astype(str)
        splits = base["split"].astype(str)
        positions = base["positions"].astype(int)
    labeled = []
    profiles = {}
    edges = set()
    for part, ordinal, row in source_rows(archive):
        uid = str(row["ID"])
        if uid in profiles:
            raise ValueError("duplicate official account ID")
        profiles[uid] = profile_values(row.get("profile"))
        if part == "support":
            continue
        labeled.append((uid, part, ordinal))
        neighbor = row.get("neighbor") or {}
        edges.update((uid, str(other)) for other in neighbor.get("following") or []
                     if str(other) != uid)
        edges.update((str(other), uid) for other in neighbor.get("follower") or []
                     if str(other) != uid)
    if (labeled != list(zip(ids, splits, positions)) or len(labeled) != 11826
            or any(a not in profiles or b not in profiles for a, b in edges)):
        raise ValueError("official account order or graph endpoint differs")
    following: dict[str, set[str]] = defaultdict(set)
    followers: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        following[a].add(b)
        followers[b].add(a)
    values = np.full((len(ids), len(FACT_NAMES)), np.nan, dtype=np.float64)
    support = np.zeros_like(values, dtype=np.int32)
    for i, uid in enumerate(ids):
        for direction, neighbors in enumerate((following[uid], followers[uid])):
            if not neighbors:
                continue
            matrix = np.stack([profiles[other] for other in sorted(neighbors)])
            for j in range(len(FIELDS)):
                finite = matrix[np.isfinite(matrix[:, j]), j]
                if len(finite):
                    col = direction * len(FIELDS) + j
                    support[i, col] = len(finite)
                    values[i, col] = float(np.mean(finite))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=ids, split=splits, positions=positions,
                        names=np.asarray(FACT_NAMES), values=values, support=support)
    manifest = {
        "dataset": "twibot20", "account_count": len(ids),
        "fact_names": list(FACT_NAMES),
        "provenance_level": "official_follow_declarations_and_raw_neighbor_profiles",
        "source_archive_sha256": sha256(archive),
        "base_facts_sha256": sha256(base_facts),
        "facts_sha256": sha256(output),
        "source_locator": "ZIP member + account ID + following/follower neighbor declaration + neighbor profile field",
        "formula": "deduplicated directed follow endpoints; mean log1p count or Boolean profile value among neighbors with a valid field",
        "support_accounts_used_as_unlabeled_profile_sources": True,
        "support_neighbor_declarations_used": False,
        "label_fields_read": False,
        "directed_edges": len(edges),
        "nonmissing_fact_cells": int(np.isfinite(values).sum()),
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
