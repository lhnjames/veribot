"""Raw-record spot audit of Fox8 native-neighbor named profile means."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.fox8_adapter import (FEATURE_FIELDS, index_sources,
                                         profile_features, tweet_features)


def raw_feature_matrix64(sources, account_ids):
    profiles = {}
    tweets = defaultdict(list)
    for item in sources.values():
        if item.kind == "profile_field":
            profiles.setdefault(item.account_id, {}).update(item.payload)
        elif item.kind == "tweet":
            ordinal = int(item.source_id.rsplit(":", 1)[1])
            tweets[item.account_id].append((ordinal, dict(item.payload)))
    rows = []
    for uid in account_ids:
        ordered = [row for _, row in sorted(tweets[uid])]
        features = {**profile_features(profiles.get(uid, {})),
                    **tweet_features(ordered)}
        rows.append([float(features[name]) for name in FEATURE_FIELDS])
    return np.asarray(rows, dtype=np.float64)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw_path = root / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    features_path = Path("D:/Code/SAHG/SAHG/code/data/fox8_23/features.csv")
    event_path = root / "data/processed/fox8_v3/interaction_event_index.json"
    artifact = root / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
    manifest = json.loads(artifact.with_suffix(".json").read_text())
    if (manifest["facts_sha256"] != sha256_file(artifact)
            or manifest["input_sha256"]["features.csv"] != sha256_file(features_path)
            or manifest["input_sha256"]["interaction_event_index.json"] != sha256_file(event_path)):
        raise ValueError("Fox8 neighbor-profile source hashes differ")
    features = pd.read_csv(features_path)
    ids = features["user_id"].astype(str).tolist()
    index = {uid: pos for pos, uid in enumerate(ids)}
    with np.load(artifact, allow_pickle=False) as saved:
        values = saved["values"]
        supports = saved["supports"]
        if saved["ids"].astype(str).tolist() != ids:
            raise ValueError("Fox8 neighbor-profile account order differs")
    event = json.loads(event_path.read_text())
    graph = [set() for _ in ids]
    for record in event["events"]:
        u = index.get(str(record["source_id"]))
        v = index.get(str(record["target_id"]))
        if u is not None and v is not None and u != v:
            graph[u].add(v)
            graph[v].add(u)
    nonempty = np.asarray([u for u, g in enumerate(graph) if g], dtype=int)
    rng = np.random.default_rng(20261103)
    sample = set(map(int, rng.choice(nonempty, size=50, replace=False)))
    sample.update(np.argsort([len(g) for g in graph])[-10:].tolist())
    selected = sorted(sample)
    neighbors = sorted(set(v for u in selected for v in graph[u]))
    neighbor_ids = [ids[v] for v in neighbors]
    sources = index_sources(raw_path, neighbor_ids)
    raw = raw_feature_matrix64(sources, neighbor_ids)
    feature_table = features[list(FEATURE_FIELDS)].to_numpy(dtype=float)
    feature_difference = float(np.max(np.abs(raw - feature_table[neighbors])))
    if feature_difference > 1e-4:
        row, col = np.unravel_index(np.argmax(np.abs(raw - feature_table[neighbors])),
                                  raw.shape)
        raise ValueError(
            f"Published Fox8 named features differ from raw records: "
            f"account={neighbor_ids[int(row)]} field={FEATURE_FIELDS[int(col)]} "
            f"raw={raw[row, col]} table={feature_table[neighbors[row], col]} "
            f"max_difference={feature_difference}")
    raw_by_index = {v: raw[pos] for pos, v in enumerate(neighbors)}
    maximum_mean_difference = 0.0
    maximum_mean_tolerance_ratio = 0.0
    checked = 0
    for u in selected:
        rows = np.asarray([raw_by_index[v] for v in sorted(graph[u])])
        count = np.isfinite(rows).sum(axis=0)
        mean = np.divide(np.nansum(rows, axis=0), count,
                         out=np.full(len(FEATURE_FIELDS), np.nan), where=count > 0)
        if not np.array_equal(supports[u], count):
            raise ValueError("Neighbor profile source support differs")
        difference = np.abs(values[u] - mean)
        tolerance = 1e-4 + 3.0 * np.finfo(np.float32).eps * np.abs(mean)
        ratio = np.divide(difference, tolerance,
                          out=np.zeros_like(difference), where=tolerance > 0)
        maximum_mean_tolerance_ratio = max(maximum_mean_tolerance_ratio,
                                           float(np.nanmax(ratio)))
        maximum_mean_difference = max(maximum_mean_difference,
                                      float(np.nanmax(difference)))
        checked += len(FEATURE_FIELDS)
    if maximum_mean_tolerance_ratio > 1.0:
        raise ValueError(
            f"Neighbor means differ beyond float32 rounding tolerance: "
            f"maximum_absolute={maximum_mean_difference} "
            f"maximum_tolerance_ratio={maximum_mean_tolerance_ratio}")
    result = {
        "dataset": "fox8",
        "scope": "raw_profile_and_tweet_recalculation_of_sampled_native_neighbor_means",
        "fact_artifact_sha256": sha256_file(artifact),
        "event_index_sha256": sha256_file(event_path),
        "raw_source_sha256": sha256_file(raw_path),
        "sample_seed": 20261103,
        "sample_target_account_count": len(selected),
        "sample_neighbor_source_account_count": len(neighbors),
        "source_fact_cells_checked": checked,
        "maximum_raw_to_published_feature_difference": feature_difference,
        "maximum_raw_to_neighbor_mean_difference": maximum_mean_difference,
        "maximum_neighbor_mean_float32_tolerance_ratio":
            maximum_mean_tolerance_ratio,
        "status": "sampled_raw_source_neighbor_means_verified",
    }
    path = root / "research/certificates/fox8_native_neighbor_profile_raw_sample_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(len(selected), len(neighbors), checked,
          feature_difference, maximum_mean_difference)


if __name__ == "__main__":
    main()
