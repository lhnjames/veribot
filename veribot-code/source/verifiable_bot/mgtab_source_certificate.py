"""Published-tensor source certificates for MGTAB profile and relation facts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .mgtab_profile_schema import PAPER_TABLE, PAPER_URL, PROFILE_SCHEMA


RELATIONS = ("followers", "friends", "mention", "reply", "quoted", "url", "hashtag")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relation_source(edge_index: np.ndarray, edge_type: np.ndarray,
                    account_index: int, relation_index: int,
                    direction: str) -> dict:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or \
            edge_index.shape[1] != len(edge_type):
        raise ValueError("edge_index and edge_type dimensions differ")
    if direction not in ("out", "in") or not 0 <= relation_index < len(RELATIONS):
        raise ValueError("unknown relation or direction")
    endpoint = 0 if direction == "out" else 1
    indices = np.flatnonzero((edge_type == relation_index) &
                             (edge_index[endpoint] == account_index)).astype(np.int64)
    # The published edge tensor may contain repeated endpoints; the model counts rows.
    source_bytes = json.dumps(indices.tolist(), separators=(",", ":")).encode()
    return {"count": len(indices), "value": float(np.float32(np.log1p(len(indices)))),
            "edge_row_indices_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "edge_row_indices_first_10": indices[:10].tolist(),
            "edge_row_indices_last_10": indices[-10:].tolist(),
            "query": f"edge_type == {relation_index} and edge_index[{endpoint}] == {account_index}",
            "operation": "log1p(number of matching published edge rows), float32"}


def certificate(data_dir: Path, account_index: int, fact_name: str) -> dict:
    features_path = data_dir / "features.pt"
    features = torch.load(features_path, map_location="cpu", weights_only=True).numpy()
    if not 0 <= account_index < len(features):
        raise ValueError("account index outside published feature tensor")
    files = {"features.pt": sha256(features_path)}
    paper_mapping = None
    if fact_name.startswith("published_profile_coordinate_"):
        coordinate = int(fact_name.removeprefix("published_profile_coordinate_"))
        if not 0 <= coordinate < 20:
            raise ValueError("profile coordinate outside published first 20")
        value = float(features[account_index, coordinate])
        paper_name, paper_type = PROFILE_SCHEMA[coordinate]
        paper_mapping = {
            "feature_name": paper_name, "feature_type": paper_type,
            "source": PAPER_TABLE, "paper_url": PAPER_URL,
            "status": "paper_dimension_and_published_tensor_type_consistent",
            "raw_profile_value_recomputed": False,
        }
        source = {"query": f"features.pt[{account_index}, {coordinate}]",
                  "operation": "read published normalized profile coordinate"}
        level = "published_feature_tensor"
    else:
        parsed = [
            (index, direction)
            for index, relation in enumerate(RELATIONS)
            for direction in ("out", "in")
            if fact_name == f"published_{relation}_{direction}_degree_log1p"
        ]
        if len(parsed) != 1:
            raise ValueError("fact name is absent from MGTAB published-fact vocabulary")
        edge_path = data_dir / "edge_index.pt"
        type_path = data_dir / "edge_type.pt"
        edge = torch.load(edge_path, map_location="cpu", weights_only=True).numpy()
        relation_type = torch.load(type_path, map_location="cpu", weights_only=True).numpy()
        if len(np.unique(relation_type)) != len(RELATIONS):
            raise ValueError("published relation type count differs")
        files.update({"edge_index.pt": sha256(edge_path),
                      "edge_type.pt": sha256(type_path)})
        relation_index, direction = parsed[0]
        source = relation_source(edge, relation_type, account_index, relation_index, direction)
        value = source["value"]
        endpoint = 0 if direction == "out" else 1
        model_fact_value = float(np.log1p(np.bincount(
            edge[endpoint, relation_type == relation_index],
            minlength=len(features))[account_index]).astype(np.float32))
        if value != model_fact_value:
            raise ValueError("source query differs from MGTAB model fact calculation")
        level = "published_typed_edge_tensor"
    return {"dataset": "mgtab", "account_index": account_index,
            "fact_name": fact_name, "value": value,
            "provenance_level": level, "files_sha256": files,
            "source_query": source, "paper_mapping": paper_mapping,
            "status": "verified_against_published_tensor"}
