"""Full raw-record audit for the Fox8 native-neighbor statuses count concept."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.fox8_adapter import FEATURE_FIELDS


FACT = "native_interaction_neighbor_mean_statuses_count"


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw_path = root / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    features_path = Path("D:/Code/SAHG/SAHG/code/data/fox8_23/features.csv")
    event_path = root / "data/processed/fox8_v3/interaction_event_index.json"
    fact_path = root / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
    frame = pd.read_csv(features_path)
    ids = frame["user_id"].astype(str).tolist()
    if len(ids) != 2279 or len(set(ids)) != 2279:
        raise ValueError("Fox8 account population differs")
    index = {uid: i for i, uid in enumerate(ids)}
    raw_values = np.full(len(ids), np.nan, dtype=np.float64)
    raw_lines = np.zeros(len(ids), dtype=np.int64)
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            uid = str(row["user_id"])
            i = index.get(uid)
            if i is None or raw_lines[i] != 0:
                continue
            tweets = row["user_tweets"]
            profile = tweets[0]["user"]
            raw_values[i] = float(profile.get("statuses_count", 0) or 0)
            raw_lines[i] = line_number
    if np.any(raw_lines == 0) or not np.isfinite(raw_values).all():
        raise ValueError("raw statuses_count missing for a published account")
    published = frame["statuses_count"].to_numpy(dtype=np.float64)
    if not np.array_equal(raw_values, published):
        raise ValueError("raw profile statuses_count differs from features.csv")
    neighbors = [set() for _ in ids]
    events = json.loads(event_path.read_text(encoding="utf-8"))["events"]
    for event in events:
        u = index.get(str(event["source_id"]))
        v = index.get(str(event["target_id"]))
        if u is not None and v is not None and u != v:
            neighbors[u].add(v)
            neighbors[v].add(u)
    with np.load(fact_path, allow_pickle=False) as saved:
        if saved["ids"].astype(str).tolist() != ids:
            raise ValueError("cached Fox8 account order differs")
        names = saved["names"].astype(str).tolist()
        fact_column = names.index(FACT)
        cached_values = saved["values"][:, fact_column].astype(np.float64)
        cached_support = saved["supports"][:, fact_column].astype(np.int64)
    output_dir = root / "output/v3_fox8_targeted_neighbor_20_concepts"
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "native_neighbor_volume_raw_sources.jsonl.gz"
    maximum_difference = 0.0
    maximum_tolerance_ratio = 0.0
    edge_links = 0
    with gzip.open(trace_path, "wt", encoding="utf-8") as trace:
        for i, uid in enumerate(ids):
            selected = sorted(neighbors[i])
            count = len(selected)
            edge_links += count
            if cached_support[i] != count:
                raise ValueError(f"native neighbor support differs for {uid}")
            if selected:
                mean = float(np.mean(raw_values[selected]))
                difference = abs(float(cached_values[i]) - mean)
                tolerance = 1e-4 + 3.0 * np.finfo(np.float32).eps * abs(mean)
                maximum_difference = max(maximum_difference, difference)
                maximum_tolerance_ratio = max(maximum_tolerance_ratio,
                                              difference / tolerance)
                if difference > tolerance:
                    raise ValueError(f"raw neighbor mean differs for {uid}")
            else:
                mean = None
                if not np.isnan(cached_values[i]):
                    raise ValueError(f"empty native neighborhood is not missing for {uid}")
            trace.write(json.dumps({
                "account_id": uid,
                "neighbor_count": count,
                "raw_neighbor_values": [
                    {"neighbor_account_id": ids[j],
                     "raw_gzip_line": int(raw_lines[j]),
                     "user_statuses_count": float(raw_values[j])}
                    for j in selected],
                "recomputed_mean": mean,
                "stored_float32_mean": (
                    float(cached_values[i]) if np.isfinite(cached_values[i]) else None),
            }) + "\n")
    result = {
        "scope": "fox8_all_accounts_native_neighbor_statuses_count_raw_source",
        "account_count": len(ids),
        "fact_name": FACT,
        "raw_profile_field": "user.statuses_count",
        "native_relation_edge_source": "reply_or_retweet_event_index",
        "symmetrized_nonself_neighbor_links": edge_links,
        "accounts_with_native_neighbors": int(sum(bool(row) for row in neighbors)),
        "raw_source_sha256": sha256_file(raw_path),
        "published_feature_sha256": sha256_file(features_path),
        "event_index_sha256": sha256_file(event_path),
        "fact_cache_sha256": sha256_file(fact_path),
        "trace_path": str(trace_path.relative_to(root)),
        "trace_sha256": sha256_file(trace_path),
        "max_raw_to_cached_float32_difference": float(maximum_difference),
        "max_float32_tolerance_ratio": float(maximum_tolerance_ratio),
        "status": "all_target_accounts_and_neighbor_raw_profiles_verified",
    }
    certificate = root / "research/certificates/fox8_neighbor_volume_full_raw_source.json"
    certificate.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "account_count", "symmetrized_nonself_neighbor_links",
        "max_raw_to_cached_float32_difference",
        "max_float32_tolerance_ratio")}), flush=True)


if __name__ == "__main__":
    main()
