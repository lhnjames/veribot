"""Label-free native interaction neighbor means for named Fox8 fields."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fox8_adapter import FEATURE_FIELDS
from .mgtab_source_certificate import sha256


def extract(features: pd.DataFrame, events: list[dict]
            ) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    if list(features.columns) != ["user_id", *FEATURE_FIELDS]:
        raise ValueError("Fox8 named feature table differs")
    ids = features["user_id"].astype(str).tolist()
    if len(ids) != 2279 or len(set(ids)) != len(ids):
        raise ValueError("Fox8 unique account population differs")
    index = {uid: i for i, uid in enumerate(ids)}
    neighbors = [set() for _ in ids]
    observed = set()
    for item in events:
        u = index.get(str(item["source_id"]))
        v = index.get(str(item["target_id"]))
        if u is None or v is None or u == v:
            continue
        observed.add((u, v))
        neighbors[u].add(v)
        neighbors[v].add(u)
    raw = features[list(FEATURE_FIELDS)].to_numpy(dtype=float)
    values = np.full((len(ids), len(FEATURE_FIELDS)), np.nan, dtype=np.float32)
    support = np.zeros_like(values, dtype=np.int32)
    for u, group in enumerate(neighbors):
        if not group:
            continue
        chosen = raw[np.fromiter(sorted(group), dtype=int)]
        available = np.isfinite(chosen)
        count = available.sum(axis=0)
        mean = np.divide(np.nansum(chosen, axis=0), count,
                         out=np.full(len(FEATURE_FIELDS), np.nan),
                         where=count > 0)
        values[u] = mean.astype(np.float32)
        support[u] = count.astype(np.int32)
    names = [f"native_interaction_neighbor_mean_{name}" for name in FEATURE_FIELDS]
    summary = {
        "observed_unique_directed_pairs": len(observed),
        "symmetric_directed_pairs": len(observed | {(v, u) for u, v in observed}),
        "accounts_with_native_neighbors": sum(bool(g) for g in neighbors),
        "maximum_native_neighbor_count": max(map(len, neighbors)),
    }
    return values, support, names, summary


def build(root: Path) -> dict:
    features_path = Path("D:/Code/SAHG/SAHG/code/data/fox8_23/features.csv")
    event_path = root / "data/processed/fox8_v3/interaction_event_index.json"
    event_manifest_path = root / "data/processed/fox8_v3/native_interaction_facts.json"
    event_manifest = json.loads(event_manifest_path.read_text(encoding="utf-8"))
    if event_manifest["event_index_sha256"] != sha256(event_path):
        raise ValueError("Fox8 native event index differs from source manifest")
    features = pd.read_csv(features_path)
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    values, supports, names, summary = extract(features, payload["events"])
    if values.shape != (2279, 31) or supports.shape != values.shape:
        raise ValueError("Native neighbor profile fact count differs")
    path = root / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
    np.savez_compressed(path, ids=np.asarray(features["user_id"].astype(str).tolist(),
                                             dtype=str),
                        names=np.asarray(names), values=values, supports=supports)
    manifest = {
        "dataset": "fox8", "account_count": len(features),
        "fact_count": len(names), "feature_names": names,
        "source_level": "raw_profile_and_sampled_tweets_via_published_feature_table_plus_native_interaction_event_targets",
        "source_operation": "mean of each named raw account feature over unique nonself native reply-or-retweet neighbors, symmetrized",
        "support_definition": "number of observed neighbor feature values; missing when no observed neighbors",
        "label_access": "none",
        "input_sha256": {"features.csv": sha256(features_path),
                         "interaction_event_index.json": sha256(event_path)},
        "native_event_manifest_sha256": sha256(event_manifest_path),
        "facts_sha256": sha256(path),
        "summary": summary,
    }
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n",
                                        encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = build(Path(__file__).resolve().parents[2])
    print(result["summary"], result["facts_sha256"])
