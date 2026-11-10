"""Build selected-model source facts without opening account class labels.

This loader reads released feature records and source-audited graph caches.
It checks their hashes against frozen training manifests. Independent raw
record audits establish how the released feature records were derived.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .botsim_adapter import FEATURE_FIELDS as BOTSIM_FIELDS
from .fox8_adapter import FEATURE_FIELDS as FOX8_FIELDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_file(path: Path, expected: dict[str, str]) -> None:
    key = str(path.resolve())
    if key not in expected or _sha256(path) != expected[key]:
        raise ValueError(f"source file differs from frozen model: {path}")


def _relation_degrees(edge_index: np.ndarray, edge_type: np.ndarray,
                      n_accounts: int) -> tuple[np.ndarray, list[str]]:
    relation_names = ("followers", "friends", "mention", "reply",
                      "quoted", "url", "hashtag")
    if (edge_index.shape[0] != 2 or edge_index.shape[1] != len(edge_type)
            or not np.array_equal(np.unique(edge_type), np.arange(7))):
        raise ValueError("MGTAB typed edge tensor differs")
    columns = []
    names = []
    for kind, relation in enumerate(relation_names):
        selected = edge_type == kind
        for endpoint, direction in ((0, "out"), (1, "in")):
            count = np.bincount(edge_index[endpoint, selected], minlength=n_accounts)
            columns.append(np.log1p(count).astype(np.float32))
            names.append(f"published_{relation}_{direction}_degree_log1p")
    return np.column_stack(columns), names


def _cache(path: Path, ids: list[str], input_names: list[str],
           expected_files: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    manifest_path = path.with_suffix(".json") if path.with_suffix(".json").exists() \
        else path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _check_file(path, expected_files)
    if str(manifest_path.resolve()) in expected_files:
        _check_file(manifest_path, expected_files)
    if manifest["facts_sha256"] != _sha256(path):
        raise ValueError("source fact cache differs from its provenance manifest")
    with np.load(path, allow_pickle=False) as saved:
        saved_ids = saved["ids"].astype(str).tolist()
        names = saved["names"].astype(str).tolist()
        values = saved["values"].copy()
    expected_names = manifest.get("feature_names", manifest.get("fact_names", names))
    if (saved_ids != ids or names != list(expected_names)
            or values.shape != (len(ids), len(names))
            or set(names) & set(input_names)):
        raise ValueError("cached source fact population or vocabulary differs")
    return values, names


def _fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_source_population(dataset: str, data_dir: Path, full_run_dir: Path,
                           *, account_file: Path | None = None,
                           native_facts: Path | None = None,
                           structural_facts: Path | None = None,
                           neighbor_facts: Path | None = None,
                           raw_source: Path | None = None,
                           inference_metadata: Path | None = None
                           ) -> tuple[np.ndarray, list[str], list[str]]:
    """Return fact values, fact names and IDs; no label file is opened."""
    metadata_path = (inference_metadata if inference_metadata is not None
                     else full_run_dir / "report.json")
    report = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = report["manifest"]
    if manifest["dataset"] != dataset:
        raise ValueError("frozen source model dataset differs")
    expected = manifest["files_sha256"]
    if dataset == "botsim":
        if account_file is None or native_facts is None:
            raise ValueError("BotSim needs user rows and native reply facts")
        features_path = data_dir / "features.csv"
        for path in (features_path, account_file):
            _check_file(path, expected)
        frame = pd.read_csv(features_path)
        users = pd.read_csv(account_file)
        if list(frame.columns) != list(BOTSIM_FIELDS):
            raise ValueError("BotSim feature vocabulary differs")
        ids = users["user_id"].astype(str).tolist()
        x = frame[list(BOTSIM_FIELDS)].to_numpy(dtype=np.float32)
        names = list(BOTSIM_FIELDS)
        graph, graph_names = _cache(native_facts, ids, names, expected)
        x = np.column_stack((x, graph))
        names += graph_names
    elif dataset == "fox8":
        if native_facts is None or raw_source is None:
            raise ValueError("Fox8 needs native interaction facts and its raw archive")
        features_path = data_dir / "features.csv"
        _check_file(features_path, expected)
        frame = pd.read_csv(features_path)
        if list(frame.columns) != ["user_id", *FOX8_FIELDS]:
            raise ValueError("Fox8 feature vocabulary differs")
        ids = frame["user_id"].astype(str).tolist()
        x = frame[list(FOX8_FIELDS)].to_numpy(dtype=np.float32)
        names = list(FOX8_FIELDS)
        graph, graph_names = _cache(native_facts, ids, names, {
            str(native_facts.resolve()): manifest["native_facts_sha256"]})
        native_manifest = json.loads(native_facts.with_suffix(".json").read_text(
            encoding="utf-8"))
        if (native_manifest["raw_source_sha256"] != _sha256(raw_source)
                or native_manifest["features_csv_sha256"] != _sha256(features_path)):
            raise ValueError("Fox8 native graph and released records differ")
        x = np.column_stack((x, graph))
        names += graph_names
    elif dataset == "mgtab":
        if structural_facts is None or neighbor_facts is None:
            raise ValueError("MGTAB needs audited structural and neighbor caches")
        paths = [data_dir / name for name in ("features.pt", "edge_index.pt", "edge_type.pt")]
        for path in paths:
            _check_file(path, expected)
        published = torch.load(paths[0], map_location="cpu", weights_only=True).numpy()
        edge_index = torch.load(paths[1], map_location="cpu", weights_only=True).numpy()
        edge_type = torch.load(paths[2], map_location="cpu", weights_only=True).numpy()
        ids = [str(i) for i in range(len(published))]
        graph, graph_names = _relation_degrees(edge_index, edge_type, len(ids))
        x = np.column_stack((published[:, :20], graph))
        names = [f"published_profile_coordinate_{i:02d}" for i in range(20)] + graph_names
        for cache_path in (structural_facts, neighbor_facts):
            added, added_names = _cache(cache_path, ids, names, expected)
            x = np.column_stack((x, added))
            names += added_names
        projection_path = full_run_dir / "projection.npz"
        if _sha256(projection_path) != manifest["projection_artifact_sha256"]:
            raise ValueError("MGTAB train-only projection artifact differs")
        with np.load(projection_path, allow_pickle=False) as projection:
            scaler_mean = projection["scaler_mean"]
            scaler_scale = projection["scaler_scale"]
            pca_mean = projection["pca_mean"]
            components = projection["pca_components"]
            added_names = projection["feature_names"].astype(str).tolist()
        if (_fingerprint(scaler_mean, scaler_scale, pca_mean, components) !=
                manifest["projection_fingerprint"] or len(added_names) != 64):
            raise ValueError("MGTAB projection parameters differ")
        normalized = published[:, 20:].astype(np.float64).copy()
        normalized -= scaler_mean
        normalized /= scaler_scale
        projected = normalized @ components.T
        projected -= pca_mean.reshape(1, -1) @ components.T
        x = np.column_stack((x, projected))
        names += added_names
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    expected_names = (report["source_fact_names"]
                      if inference_metadata is not None else
                      report["reasoner"]["directions"])
    if (len(ids) != len(set(ids)) or x.shape != (len(ids), len(names))
            or tuple(names) != tuple(expected_names)):
        raise ValueError("label-free source matrix differs from frozen reference")
    return x, names, ids
