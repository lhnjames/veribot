"""Fit and version model-visible qualitative profile buckets on train-only accounts.

The deterministic project holdout is excluded from fitting.  Counts are log-transformed before
quantiles.  Zero-inflated fields use an explicit zero bucket plus positive-value quantiles rather
than pretending identical zero values occupy several distinct percentile bands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


FIELDS = ("followers_count", "following_count", "tweet_count", "listed_count")
LABELS = ("very low", "low", "moderate", "high", "very high")


def internal_holdout(uid: str) -> bool:
    return int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) % 20 == 0


def number(value) -> float:
    try:
        return max(float(str(value).strip() or 0), 0.0)
    except (TypeError, ValueError):
        return 0.0


def bucket(values: np.ndarray, spec: dict) -> np.ndarray:
    transformed = np.log1p(np.maximum(values, 0))
    if spec["strategy"] == "zero_aware_log_quantile":
        result = np.zeros(len(values), dtype=np.int8)
        positive = values > 0
        result[positive] = np.searchsorted(spec["cutpoints_log1p"], transformed[positive], side="right") + 1
        return result
    return np.searchsorted(spec["cutpoints_log1p"], transformed, side="right").astype(np.int8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--twibot20", type=Path, required=True)
    parser.add_argument("--twibot22", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows20 = json.loads((args.twibot20 / "train.json").read_text(encoding="utf-8"))
    values = {field: [] for field in FIELDS}
    ids = []
    sources = []
    for row in rows20:
        uid = str(row.get("ID", "")).strip()
        if internal_holdout(uid):
            continue
        profile = row.get("profile") or {}
        mapping = {
            "followers_count": profile.get("followers_count"),
            "following_count": profile.get("friends_count"),
            "tweet_count": profile.get("statuses_count"),
            "listed_count": profile.get("listed_count"),
        }
        for field in FIELDS:
            values[field].append(number(mapping[field]))
        ids.append(uid)
        sources.append("twibot20")

    raw22 = np.load(args.twibot22 / "num_properties_raw.npy", mmap_mode="r")
    uid_to_idx = torch.load(args.twibot22 / "uid_to_idx.pt", map_location="cpu", weights_only=False)
    idx_to_uid = [""] * len(uid_to_idx)
    for uid, index in uid_to_idx.items():
        idx_to_uid[int(index)] = str(uid)
    train_indices = np.fromiter(
        (i for i, uid in enumerate(idx_to_uid) if not internal_holdout(uid)), dtype=np.int64
    )
    for column, field in enumerate(FIELDS):
        values[field].extend(np.maximum(np.asarray(raw22[train_indices, column]), 0).astype(float).tolist())
    ids.extend(idx_to_uid[i] for i in train_indices)
    sources.extend("twibot22" for _ in train_indices)

    config = {
        "version": "veriroute_lite_v3_buckets_1",
        "fit_scope": "official training accounts excluding deterministic project holdout",
        "fit_sources": ["twibot20", "twibot22"],
        "n_fit_accounts": len(ids),
        "labels": list(LABELS),
        "fields": {},
        "observations": {
            "ratio": {
                "minimum_ratio": 2.0, "strong_ratio": 5.0, "extreme_ratio": 20.0,
                "labels": ["mild imbalance", "strong imbalance", "extreme imbalance"],
            },
            "burst": {
                "window_minutes_hidden": 15, "minimum_posts": 5,
                "moderate_posts": 10, "high_posts": 20,
                "labels": ["low-level burst", "moderate burst", "high-intensity burst"],
            },
            "graph_direction": {
                "strong_share": 0.75,
                "labels": ["mostly incoming", "mixed-direction", "mostly outgoing"],
            },
        },
    }
    arrays = {field: np.asarray(values[field], dtype=np.float64) for field in FIELDS}
    for field, array in arrays.items():
        zero_fraction = float(np.mean(array == 0))
        transformed = np.log1p(array)
        standard = np.quantile(transformed, [0.10, 0.30, 0.70, 0.90])
        if zero_fraction >= 0.10 or len(np.unique(standard)) < 4:
            positives = transformed[array > 0]
            if positives.size == 0:
                cuts = np.array([0.0, 0.0, 0.0])
            else:
                cuts = np.quantile(positives, [0.25, 0.65, 0.90])
            strategy = "zero_aware_log_quantile"
        else:
            cuts = standard
            strategy = "log_quantile"
        spec = {
            "strategy": strategy,
            "cutpoints_log1p": [float(v) for v in cuts],
            "cutpoints_raw": [float(np.expm1(v)) for v in cuts],
            "zero_fraction": zero_fraction,
        }
        assignments = bucket(array, spec)
        counts = np.bincount(assignments, minlength=5)
        spec["bucket_counts"] = {LABELS[i]: int(counts[i]) for i in range(5)}
        spec["bucket_fractions"] = {LABELS[i]: float(counts[i] / len(array)) for i in range(5)}
        # Store real accounts closest to each boundary for the required manual boundary audit.
        boundary_examples = []
        for cut in cuts:
            order = np.argsort(np.abs(transformed - cut))[:4]
            boundary_examples.append([
                {"source": sources[int(i)], "account_id": ids[int(i)], "raw_value": float(array[int(i)])}
                for i in order
            ])
        spec["boundary_examples"] = boundary_examples
        config["fields"][field] = spec

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
