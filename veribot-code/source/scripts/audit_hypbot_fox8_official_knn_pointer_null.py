"""Audit whether raw native interaction target pointers enter Fox8 kNN HypBot.

The saved local HypBot kNN-configuration checkpoint is replayed. Every raw
reply/retweet target-user pointer supporting an internal native relation is
cleared in memory, and all 31 model feature coordinates are recomputed from
affected raw account rows. The edited features, derived kNN graph, and frozen
predictions must then be compared exactly with their original counterparts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/scripts"))
sys.path.insert(0, str(ROOT / "source"))
from audit_hypbot_fox8_raw_edge_response import (  # noqa: E402
    DATA, EVENT_INDEX, RAW, RAW_AUDIT, REPO,
)
from train_v2_detector import file_sha256, load_data  # noqa: E402
from verifiable_bot.fox8_adapter import (  # noqa: E402
    FEATURE_FIELDS, profile_features, tweet_features,
)

RUN = ROOT / "output/hypbot_fox8_paired/seed0"
OUT = ROOT / "research/certificates/hypbot_fox8_official_knn_pointer_null.json"


def raw_features(row: dict, tweets: list[dict]) -> np.ndarray:
    profile = profile_features((tweets[0].get("user") or {}) if tweets else {})
    merged = {**profile, **tweet_features(tweets)}
    return np.asarray([float(merged[name]) for name in FEATURE_FIELDS],
                      dtype=np.float32)


def main() -> None:
    sys.path.insert(0, str(REPO.resolve()))
    from hypbot.data import knn_graph
    from hypbot.model import HypBot
    from train import PRESETS

    torch.set_num_threads(4)
    saved = json.loads((RUN / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = saved["manifest"]
    checkpoint_path = RUN / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    feature_csv = DATA / "features.csv"
    args = argparse.Namespace(dataset="fox8", data_dir=DATA, account_file=None,
                              raw_source=[feature_csv], knn_k=10)
    x, labels, split, ids, edge, _, hashes, _ = load_data(args, 0)
    if (manifest["dataset"] != "fox8" or manifest["seed"] != 0
            or file_sha256(checkpoint_path) != manifest["model_sha256"]
            or any(checkpoint["hashes"][key] != hashes[key] for key in (
                "feature_sha256", "graph_sha256", "split_sha256"))
            or checkpoint["threshold"] != manifest["threshold"]
            or any(hashes[key] != manifest[key] for key in (
                "feature_sha256", "graph_sha256", "split_sha256"))
            or file_sha256(RAW) != json.loads(EVENT_INDEX.read_text(encoding="utf-8"))["raw_source_sha256"]):
        raise ValueError("Saved kNN model, features or raw source differ")
    audit = json.loads(RAW_AUDIT.read_text(encoding="utf-8"))
    if (audit["status"] != "verified_all_native_interaction_fact_cells_and_event_locators"
            or audit["raw_source_sha256"] != file_sha256(RAW)
            or audit["event_index_sha256"] != file_sha256(EVENT_INDEX)):
        raise ValueError("Full raw interaction audit does not match")
    frame = pd.read_csv(feature_csv)
    if list(frame.columns) != ["user_id", *FEATURE_FIELDS]:
        raise ValueError("Saved feature column contract differs")
    if frame["user_id"].astype(str).tolist() != ids:
        raise ValueError("Saved feature account order differs")
    raw_x = frame[list(FEATURE_FIELDS)].fillna(0.0).to_numpy(dtype=np.float32)
    edited_raw_x = raw_x.copy()
    by_line = defaultdict(list)
    event_index = json.loads(EVENT_INDEX.read_text(encoding="utf-8"))
    for event in event_index["events"]:
        for locator in event["locators"]:
            by_line[int(locator["line"])].append((event, locator))
    account_index = {uid: i for i, uid in enumerate(ids)}
    checked = 0
    touched = 0
    max_raw_csv_error = 0.0
    max_edited_feature_error = 0.0
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number not in by_line:
                continue
            row = json.loads(line)
            uid = str(row["user_id"])
            j = account_index.get(uid)
            if j is None:
                raise ValueError("Source event account absent from model features")
            tweets = list(row["user_tweets"])
            edited = list(tweets)
            before = raw_features(row, tweets)
            max_raw_csv_error = max(max_raw_csv_error,
                                    float(np.max(np.abs(before - raw_x[j]))))
            if not np.allclose(before, raw_x[j], rtol=1e-5, atol=1e-5):
                raise ValueError(f"Raw account {uid} does not rebuild saved 31 features")
            for event, locator in by_line[line_number]:
                if event["source_id"] != uid:
                    raise ValueError("Source event account differs")
                k = int(locator["tweet_index"])
                original = tweets[k]
                tid = str(original.get("id_str") or original.get("id") or k)
                if tid != str(locator["tweet_id"]):
                    raise ValueError("Raw tweet ID differs from source locator")
                modified = dict(edited[k])
                if event["relation"] == "reply":
                    target = (original.get("in_reply_to_user_id_str") or
                              original.get("in_reply_to_user_id"))
                    modified["in_reply_to_user_id_str"] = None
                    modified["in_reply_to_user_id"] = None
                elif event["relation"] == "retweet":
                    user = (original.get("retweeted_status") or {}).get("user") or {}
                    target = user.get("id_str") or user.get("id")
                    status_edit = dict(modified["retweeted_status"])
                    user_edit = dict(status_edit["user"])
                    user_edit["id_str"] = None
                    user_edit["id"] = None
                    status_edit["user"] = user_edit
                    modified["retweeted_status"] = status_edit
                else:
                    raise ValueError("Unknown raw relation")
                if str(target) != event["target_id"]:
                    raise ValueError("Raw target pointer differs")
                edited[k] = modified
                checked += 1
            after = raw_features(row, edited)
            edited_raw_x[j] = after
            max_edited_feature_error = max(max_edited_feature_error,
                                           float(np.max(np.abs(after - before))))
            touched += 1
    if checked != 5873 or touched != 1350 or not np.array_equal(edited_raw_x, raw_x):
        raise ValueError("Raw target-pointer edits changed model feature rows")
    scaler = StandardScaler().fit(raw_x[split["train"]])
    rebuilt_x = torch.from_numpy(np.clip(
        scaler.transform(edited_raw_x), -3, 3).astype(np.float32))
    if not torch.equal(rebuilt_x, x):
        raise ValueError("Edited raw source changed model feature tensor")
    official_edge = knn_graph(x, k=10)
    edited_edge = knn_graph(rebuilt_x, k=10)
    if not torch.equal(official_edge, edge) or not torch.equal(edited_edge, edge):
        raise ValueError("Edited raw source changed official kNN graph")
    model = HypBot(x.shape[1], K=checkpoint["num_sectors"], gated=False,
                   **PRESETS["fox8"]["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        original_prob = model.predict_prob(x, official_edge).cpu().numpy()
        edited_prob = model.predict_prob(rebuilt_x, edited_edge).cpu().numpy()
    test = np.asarray(split["test"], dtype=np.int64)
    if [ids[int(j)] for j in test] != [r["account_id"] for r in saved["predictions"]]:
        raise ValueError("Frozen kNN test population differs")
    if not np.array_equal(original_prob, edited_prob):
        raise ValueError("Edited pointer source changed kNN model scores")
    replay_error = float(np.max(np.abs(original_prob[test] -
                                    [r["probability"] for r in saved["predictions"]])))
    if replay_error > 1e-6:
        raise ValueError("Frozen kNN checkpoint does not replay saved predictions")
    report = {
        "scope": "Fox8 seed0 saved HypBot kNN-graph configuration",
        "source_operation": "clear all 5873 reply/retweet target-user ID pointers supporting native internal pairs, then rebuild all 31 node features and cosine kNN graph",
        "raw_source_sha256": file_sha256(RAW),
        "source_index_sha256": file_sha256(EVENT_INDEX),
        "full_raw_interaction_audit_sha256": file_sha256(RAW_AUDIT),
        "feature_csv_sha256": file_sha256(feature_csv),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "saved_predictions_sha256": file_sha256(RUN / "test_predictions.json"),
        "model_feature_sha256": hashes["feature_sha256"],
        "model_knn_graph_sha256": hashes["graph_sha256"],
        "raw_locator_count_verified": checked,
        "source_account_rows_with_pointer_edits": touched,
        "node_feature_columns": len(FEATURE_FIELDS),
        "max_raw_rebuild_vs_csv_feature_error": max_raw_csv_error,
        "max_edited_vs_original_raw_feature_error": max_edited_feature_error,
        "edited_scaled_features_identical": True,
        "edited_knn_graph_identical": True,
        "test_accounts": len(test),
        "max_saved_probability_replay_error": replay_error,
        "max_edited_vs_original_probability_difference": float(np.max(
            np.abs(edited_prob[test] - original_prob[test]))),
        "test_class_changes": int(np.count_nonzero(
            (original_prob[test] >= manifest["threshold"]) !=
            (edited_prob[test] >= manifest["threshold"]))),
        "mechanism": "native interaction target-user pointers are not read by the 31 feature coordinates or the feature-derived cosine kNN graph",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
