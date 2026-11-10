"""Verify materialized Fox8 profile counts against their raw source records."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RawStatusesSnapshot:
    values: np.ndarray
    ambiguous_account_ids: tuple[str, ...]


def read_first_raw_statuses(path: Path, account_ids: list[str]) -> RawStatusesSnapshot:
    """Read the first source snapshot and flag accounts with conflicting later ones."""
    index = {str(uid): i for i, uid in enumerate(account_ids)}
    if len(index) != len(account_ids):
        raise ValueError("account IDs must be unique")
    values = np.full(len(index), np.nan, dtype=float)
    ambiguous: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            uid = str(record["user_id"])
            if uid not in index:
                continue
            tweets = record.get("user_tweets") or []
            profile = (tweets[0].get("user") or {}) if tweets else {}
            status = float(profile.get("statuses_count", 0) or 0)
            if not np.isfinite(status) or status < 0:
                raise ValueError(f"invalid raw statuses_count for {uid}")
            position = index[uid]
            if np.isfinite(values[position]):
                if status != values[position]:
                    ambiguous.add(uid)
            else:
                values[position] = status
    if not np.isfinite(values).all():
        raise ValueError("raw source does not cover the account population")
    return RawStatusesSnapshot(values, tuple(sorted(ambiguous)))


def detect_statuses_discrepancies(
        account_ids: list[str], materialized: np.ndarray,
        snapshot: RawStatusesSnapshot) -> list[str]:
    """Return mismatched, unambiguous source IDs without reading class labels."""
    values = np.asarray(materialized, dtype=float)
    if values.shape != snapshot.values.shape or len(account_ids) != len(values):
        raise ValueError("materialized and raw populations differ")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("materialized statuses_count has invalid entries")
    ambiguous = set(snapshot.ambiguous_account_ids)
    return [str(uid) for uid, actual, source in zip(
        account_ids, values, snapshot.values)
        if str(uid) not in ambiguous and actual != source]
