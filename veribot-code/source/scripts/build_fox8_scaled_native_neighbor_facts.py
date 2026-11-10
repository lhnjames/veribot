"""Rebuild named Fox8 neighbor means after train-only scaling and clipping.

The operation matches the native-edge HypBot graph input, while preserving
source field names, train-only scaler parameters, graph provenance and degree.
No bot/human label is used except by the released stratified split algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_hypbot_native_graph_paired import native_symmetric_graph
from gate0_shallow_tree import sha256_file
from train_v2_detector import load_data
from verifiable_bot.fox8_adapter import FEATURE_FIELDS


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    previous = root / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
    raw_audit_path = root / "research/certificates/fox8_native_neighbor_profile_raw_sample_audit.json"
    raw_audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
    if raw_audit["fact_artifact_sha256"] != sha256_file(previous):
        raise ValueError("Raw-record neighbor source audit differs")
    with np.load(previous, allow_pickle=False) as cache:
        old_ids = cache["ids"].astype(str).tolist()
        raw_neighbor_mean = cache["values"].astype(float)
        support = cache["supports"]
    args = argparse.Namespace(
        dataset="fox8", data_dir=Path("D:/Code/SAHG/SAHG/code/data/fox8_23"),
        account_file=None, raw_source=[root / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"],
        knn_k=10)
    runs = []
    for seed in (0, 1, 2):
        x, _, split, ids, _, _, hashes, scaler = load_data(args, seed)
        if ids != old_ids or x.shape[1] != len(FEATURE_FIELDS):
            raise ValueError("Fox8 named source feature order differs")
        edge, graph = native_symmetric_graph(root, "fox8", ids)
        from hypbot.data import mean_adjacency
        adjacency = mean_adjacency(edge, len(ids))
        aggregate = torch.sparse.mm(adjacency, x).numpy()
        degree = np.bincount(edge[1].numpy(), minlength=len(ids))
        if not np.array_equal(support, degree[:, None] * np.ones_like(support)):
            raise ValueError("Native neighbor support differs from graph degree")
        available = degree > 0
        raw_mean_then_clip = np.clip(
            (raw_neighbor_mean[available] - scaler.mean_) / scaler.scale_,
            -3.0, 3.0)
        gap = np.abs(aggregate[available] - raw_mean_then_clip)
        rng = np.random.default_rng(20261103 + seed)
        sample = rng.choice(np.flatnonzero(available), size=100, replace=False)
        graph_neighbors = [set() for _ in ids]
        for u, v in edge.T.numpy():
            graph_neighbors[int(v)].add(int(u))
        max_independent_error = 0.0
        for u in sample:
            expected = x[np.asarray(sorted(graph_neighbors[int(u)]))].numpy().mean(axis=0)
            max_independent_error = max(
                max_independent_error,
                float(np.max(np.abs(expected - aggregate[u]))))
        if max_independent_error > 1e-6:
            raise ValueError("Named neighbor aggregate differs from source graph")
        values = aggregate.copy()
        values[~available] = np.nan
        path = root / f"data/processed/fox8_v3/scaled_native_neighbor_seed{seed}.npz"
        np.savez_compressed(path, ids=np.asarray(ids, dtype=str),
                            names=np.asarray([f"native_neighbor_scaled_mean_{name}"
                                              for name in FEATURE_FIELDS]),
                            values=values.astype(np.float32),
                            supports=support.astype(np.int32),
                            scaler_mean=scaler.mean_.astype(np.float64),
                            scaler_scale=scaler.scale_.astype(np.float64),
                            train_indices=np.asarray(split["train"], dtype=np.int64))
        run = {
            "seed": seed,
            "source_feature_sha256": hashes["feature_sha256"],
            "split_sha256": hashes["split_sha256"],
            "native_graph": graph,
            "raw_neighbor_mean_sha256": sha256_file(previous),
            "raw_neighbor_audit_sha256": sha256_file(raw_audit_path),
            "scaler_fingerprint": array_hash(scaler.mean_, scaler.scale_),
            "accounts_with_native_neighbors": int(np.count_nonzero(available)),
            "mean_absolute_gap_from_mean_raw_then_clip": float(np.mean(gap)),
            "fraction_of_active_account_feature_cells_gap_over_0_25":
                float(np.mean(gap > 0.25)),
            "max_gap_from_mean_raw_then_clip": float(np.max(gap)),
            "independent_account_sample_count": len(sample),
            "max_independent_graph_aggregate_difference": max_independent_error,
            "artifact_sha256": sha256_file(path),
            "artifact_path": str(path),
        }
        runs.append(run)
        print(json.dumps({"seed": seed,
                          "mean_gap": run["mean_absolute_gap_from_mean_raw_then_clip"],
                          "fraction_gap_over_0_25": run[
                              "fraction_of_active_account_feature_cells_gap_over_0_25"],
                          "max_independent_error": max_independent_error}), flush=True)
    report = {
        "dataset": "fox8",
        "scope": "train_only_scaled_clipped_named_native_neighbor_aggregate",
        "test_labels_loaded_by_full_dataset_adapter": True,
        "test_labels_used_for_facts_or_scaler": False,
        "test_predictions_or_metrics_computed": False,
        "fact_operation": "mean over source-audited symmetric native neighbors of each named source feature after train-only StandardScaler and clip(-3,3)",
        "runs": runs,
    }
    path = root / "research/certificates/fox8_scaled_native_neighbor_aggregate_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.path.insert(0, str(Path("D:/Code/SAHG/SAHG/HypBot")))
    torch.set_num_threads(4)
    main()
