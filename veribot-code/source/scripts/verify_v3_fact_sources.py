"""Recompute every Fox8/BotSim published fact row from its raw release.

This checks the full population, not just selected explanation examples. The
report records per-field mismatches and file fingerprints. Passing this gate
establishes numeric source consistency for the current fact vocabulary; it does
not establish that the fact has external semantic validity or is a model cause.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.evidence_v2 import SourceRecord
from verifiable_bot.fox8_adapter import (
    FEATURE_FIELDS as FOX8_FIELDS, profile_features, tweet_features,
)
from verifiable_bot.botsim_adapter import (
    CSV_KEYS, FEATURE_FIELDS as BOTSIM_FIELDS, account_features,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(expected: np.ndarray, actual: np.ndarray, names: list[str], uid: str,
            mismatches: list[dict], counts: Counter) -> None:
    matched = np.isclose(expected, actual, rtol=1e-5, atol=1e-5, equal_nan=False)
    for j in np.flatnonzero(~matched):
        counts[names[j]] += 1
        if len(mismatches) < 100:
            mismatches.append({"account_id": uid, "field": names[j],
                               "expected": float(expected[j]), "raw_recomputed": float(actual[j])})


def fox8(raw: Path, features: Path) -> dict:
    frame = pd.read_csv(features)
    if list(frame.columns) != ["user_id", *FOX8_FIELDS]:
        raise ValueError("Fox8 feature fields differ from audited formulas")
    ids = frame["user_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("processed Fox8 account IDs are not unique")
    index = {uid: i for i, uid in enumerate(ids)}
    expected = frame[list(FOX8_FIELDS)].to_numpy(dtype=np.float32)
    found = set()
    mismatches: list[dict] = []
    counts: Counter = Counter()
    duplicated = 0
    opener = gzip.open if raw.suffix == ".gz" else open
    with opener(raw, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid in found:
                duplicated += 1
                continue
            found.add(uid)
            if uid not in index:
                continue
            tweets = row.get("user_tweets") or []
            profile = tweets[0]["user"] if tweets else {}
            values = {**profile_features(profile), **tweet_features(tweets)}
            actual = np.asarray([float(values[name]) for name in FOX8_FIELDS], dtype=np.float32)
            compare(expected[index[uid]], actual, list(FOX8_FIELDS), uid, mismatches, counts)
    return {"dataset": "fox8", "accounts_in_features": len(ids),
            "raw_unique_accounts": len(found), "raw_duplicate_rows": duplicated,
            "missing_feature_accounts": sorted(set(ids) - found),
            "mismatch_by_field": dict(counts), "first_mismatches": mismatches,
            "all_rows_match": found >= set(ids) and not counts,
            "files_sha256": {str(path.resolve()): sha256(path) for path in (raw, features)}}


def botsim(users_csv: Path, activity_json: Path, features_csv: Path) -> dict:
    users = pd.read_csv(users_csv)
    frame = pd.read_csv(features_csv)
    if list(frame.columns) != list(BOTSIM_FIELDS) or len(users) != len(frame):
        raise ValueError("BotSim feature rows or names differ from audited formulas")
    ids = users["user_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("BotSim account IDs are not unique")
    with activity_json.open("r", encoding="utf-8") as stream:
        activity = json.load(stream)
    expected = frame[list(BOTSIM_FIELDS)].to_numpy(dtype=np.float32)
    mismatches: list[dict] = []
    counts: Counter = Counter()
    for i, uid in enumerate(ids):
        records: dict[str, SourceRecord] = {}
        row = users.iloc[i]
        for key in CSV_KEYS:
            if pd.isna(row[key]):
                continue
            sid = f"botsim:{uid}:user:{key}"
            records[sid] = SourceRecord(sid, uid, "user_field", {key: row[key]},
                                        f"{users_csv.resolve()}#row={i + 2};{key}")
        for kind in ("posts", "comment_1", "comment_2"):
            for ordinal, value in enumerate(activity.get(uid, {}).get(kind, [])):
                sid = f"botsim:{uid}:{kind}:{ordinal}"
                records[sid] = SourceRecord(sid, uid, "post" if kind == "posts" else kind,
                                            value, f"{activity_json.resolve()}#{uid}/{kind}/{ordinal}")
        compare(expected[i], account_features(records), list(BOTSIM_FIELDS),
                uid, mismatches, counts)
    return {"dataset": "botsim", "accounts_in_features": len(ids),
            "activity_accounts": len(activity),
            "missing_activity_accounts": sorted(set(ids) - set(activity)),
            "mismatch_by_field": dict(counts), "first_mismatches": mismatches,
            "all_rows_match": set(ids) <= set(activity) and not counts,
            "excluded_label_proxy": "character_setting",
            "files_sha256": {str(path.resolve()): sha256(path)
                             for path in (users_csv, activity_json, features_csv)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fox8", "botsim"), required=True)
    parser.add_argument("--raw", type=Path, help="Fox8 NDJSON or NDJSON.gz")
    parser.add_argument("--users", type=Path, help="BotSim Users.csv")
    parser.add_argument("--activity", type=Path, help="BotSim user_post_comment.json")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset == "fox8":
        if args.raw is None:
            parser.error("Fox8 needs --raw")
        report = fox8(args.raw, args.features)
    else:
        if args.users is None or args.activity is None:
            parser.error("BotSim needs --users and --activity")
        report = botsim(args.users, args.activity, args.features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "files_sha256" and key != "first_mismatches"}, indent=2))
    if not report["all_rows_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
