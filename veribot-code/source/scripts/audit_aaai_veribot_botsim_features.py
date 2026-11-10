"""Audit all BotSim user facts and frozen inputs against raw CSV and JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/scripts"))

from train_aaai_veribot_fox8 import sha256_file, tensor_hash
from verifiable_bot.aaai_botsim_native_graph import read_memberships
from verifiable_bot.aaai_botsim_raw_features import FEATURE_FIELDS, rebuild_feature_matrix

USERS = ROOT / "data/raw/botsim_24/Users.csv"
ACTIVITY = ROOT / "data/raw/botsim_24/user_post_comment.json"
FEATURES = ROOT / "data/processed/aaai_veribot/botsim_24/features.csv"
SPLIT = ROOT / "data/splits/botsim_24_seed0.json"
CHECKPOINT = ROOT / "output/aaai_veribot_botsim/seed0/model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/certificates/aaai_veribot_botsim_all_features_audit.json")
    args = parser.parse_args()
    ids, _, memberships = read_memberships(USERS)
    raw = rebuild_feature_matrix(USERS, ACTIVITY, ids)
    frame = pd.read_csv(FEATURES)
    if list(frame.columns) != list(FEATURE_FIELDS):
        raise ValueError("model feature names and raw formulas differ")
    supplied = frame.to_numpy(dtype=np.float32)
    if raw.shape != supplied.shape or not np.array_equal(raw, supplied):
        mismatch = np.argwhere(~np.isclose(raw, supplied, rtol=1e-6, atol=1e-5))
        position = mismatch[0].tolist() if len(mismatch) else None
        raise ValueError(f"raw BotSim facts differ from training inputs at {position}")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))["splits"]
    train = np.asarray(split["train"]["indices"], dtype=np.int64)
    scaler = StandardScaler().fit(raw[train])
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    contract = saved["input_contract"]
    if not np.array_equal(scaler.mean_, np.asarray(contract["scaler_mean"])):
        raise ValueError("raw-derived train mean differs from checkpoint")
    if not np.array_equal(scaler.scale_, np.asarray(contract["scaler_scale"])):
        raise ValueError("raw-derived train scale differs from checkpoint")
    user_x = torch.from_numpy(np.clip(scaler.transform(raw), -3, 3).astype(np.float32))
    if tensor_hash(user_x) != contract["user_input_sha256"]:
        raise ValueError("raw-derived standardized input differs from model")
    output = {
        "status": "verified_raw_to_all_named_features_and_frozen_input",
        "account_count": len(ids),
        "feature_count": len(FEATURE_FIELDS),
        "checked_cells": int(raw.size),
        "max_raw_feature_absolute_difference": float(np.max(np.abs(raw - supplied))),
        "membership_count": len(memberships),
        "raw_users_sha256": sha256_file(USERS),
        "raw_activity_sha256": sha256_file(ACTIVITY),
        "feature_file_sha256": sha256_file(FEATURES),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "user_input_sha256": tensor_hash(user_x),
        "label_fields_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
