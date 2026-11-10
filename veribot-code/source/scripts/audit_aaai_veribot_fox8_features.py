"""Audit every Fox8 model feature against first-occurrence raw account records."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

from verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS
from verifiable_bot.aaai_fox8_raw_features import rebuild_feature_matrix

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
FEATURES = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
SPLIT = ROOT / "data/splits/fox8_23_seed0.json"
CHECKPOINT = ROOT / "output/aaai_veribot_fox8/seed0/model.pt"


def tensor_hash(value: torch.Tensor) -> str:
    digest = sha256()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/certificates/aaai_veribot_fox8_all_features_audit.json")
    args = parser.parse_args()
    frame = pd.read_csv(FEATURES)
    if list(frame.columns) != ["user_id", *FEATURE_FIELDS]:
        raise ValueError("feature table schema differs from model input")
    ids = frame["user_id"].astype(str).tolist()
    raw, locations, raw_rows = rebuild_feature_matrix(RAW, ids)
    supplied = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
    mismatch = ~np.isclose(raw, supplied, rtol=1e-6, atol=1e-5)
    if mismatch.any():
        row, column = np.argwhere(mismatch)[0]
        raise ValueError(f"raw fact differs: account={ids[row]} field={FEATURE_FIELDS[column]} "
                         f"raw={raw[row,column]} csv={supplied[row,column]}")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))["splits"]
    index = {uid: i for i, uid in enumerate(ids)}
    train_ids = list(dict.fromkeys(split["train"]["account_ids"]))
    train_rows = np.asarray([index[uid] for uid in train_ids])
    scaler = StandardScaler().fit(raw[train_rows])
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    contract = checkpoint["input_contract"]
    if not np.allclose(scaler.mean_, contract["scaler_mean"], rtol=0, atol=1e-8):
        raise ValueError("raw-derived training mean differs from checkpoint")
    if not np.allclose(scaler.scale_, contract["scaler_scale"], rtol=0, atol=1e-8):
        raise ValueError("raw-derived training scale differs from checkpoint")
    x = torch.from_numpy(np.clip(scaler.transform(raw), -3, 3).astype(np.float32))
    feature_sha = tensor_hash(x)
    if feature_sha != contract["feature_sha256"]:
        raise ValueError("raw-derived scaled input differs from frozen checkpoint")
    result = {
        "status": "verified_raw_to_all_named_features_and_frozen_input",
        "raw_source": str(RAW.resolve()),
        "raw_source_sha256": _file_sha256(RAW),
        "checkpoint_sha256": _file_sha256(CHECKPOINT),
        "account_count": len(ids),
        "released_record_count": raw_rows,
        "named_feature_count": len(FEATURE_FIELDS),
        "checked_cells": int(raw.size),
        "first_record_locations": locations,
        "max_raw_feature_absolute_difference": float(np.max(np.abs(raw - supplied))),
        "scaled_input_sha256": feature_sha,
        "label_fields_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "first_record_locations"}), flush=True)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
