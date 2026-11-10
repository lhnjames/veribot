"""MGTAB profile dimensions as described by the source paper's Table A5.

The paper gives names and one-based dimensions. The release contains only a
normalized tensor, so these aliases do not certify original profile values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


PAPER_URL = "https://arxiv.org/pdf/2301.01123"
PAPER_TABLE = "Table A5, one-based dimensions 1-20"
PROFILE_SCHEMA = (
    ("profile_use_background_image", "boolean"),
    ("default_profile", "boolean"),
    ("verified", "boolean"),
    ("followers_count", "normalized_numerical"),
    ("default_profile_image", "boolean"),
    ("listed_count", "normalized_numerical"),
    ("statuses_count", "normalized_numerical"),
    ("friends_count", "normalized_numerical"),
    ("geo_enabled", "boolean"),
    ("favourites_count", "normalized_numerical"),
    ("created_at", "normalized_numerical"),
    ("screen_name_length", "normalized_numerical"),
    ("name_length", "normalized_numerical"),
    ("description_length", "normalized_numerical"),
    ("followers_friends_ratio", "normalized_numerical"),
    ("default_profile_background_color", "boolean"),
    ("default_profile_sidebar_fill_color", "boolean"),
    ("default_profile_sidebar_border_color", "boolean"),
    ("has_url", "boolean"),
    ("profile_background_image_url", "boolean"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(features_path: Path) -> dict:
    features = torch.load(features_path, map_location="cpu", weights_only=True).numpy()
    if features.ndim != 2 or features.shape[1] < len(PROFILE_SCHEMA):
        raise ValueError("published MGTAB tensor lacks 20 profile dimensions")
    rows = []
    for index, (name, expected_type) in enumerate(PROFILE_SCHEMA):
        values = np.asarray(features[:, index], dtype=float)
        unique = np.unique(values)
        in_unit_interval = bool(np.isfinite(values).all() and
                                np.min(values) >= 0 and np.max(values) <= 1)
        if expected_type == "boolean":
            type_consistent = in_unit_interval and set(unique.tolist()) <= {0.0, 1.0}
        else:
            type_consistent = in_unit_interval and len(unique) > 2
        if not type_consistent:
            raise ValueError(f"dimension {index+1} contradicts paper Table A5 type")
        rows.append({"zero_based_index": index, "paper_dimension": index+1,
                     "paper_feature_name": name, "paper_type": expected_type,
                     "distinct_values": len(unique),
                     "minimum": float(unique[0]), "maximum": float(unique[-1]),
                     "type_consistent": True})
    return {
        "dataset": "mgtab", "paper_url": PAPER_URL, "paper_table": PAPER_TABLE,
        "provenance_level": "paper_dimension_name_plus_published_normalized_tensor",
        "raw_profile_values_available": False,
        "mapping_status": "paper_order_consistent_with_tensor_type_not_raw_field_recomputed",
        "features_sha256": sha256(features_path),
        "accounts": int(len(features)), "coordinates": rows,
    }
