"""BotSim source adapter: field/post edits rebuild features and the kNN input."""

from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .evidence_v2 import BuiltInputs, RebuildingAdapter, SourceRecord, digest
from .v2_detector import EvidenceGatedDetector, knn_edges, relation_means


CSV_KEYS = ("submission_num", "comment_num", "comment_num_1", "comment_num_2",
            "description", "subreddit")
FEATURE_FIELDS = (
    "submission_num", "comment_num", "comment_num_1", "comment_num_2",
    "has_description", "description_len", "n_subreddits", "n_posts",
    "n_comments_received", "n_comments_made", "avg_post_score",
    "avg_upvote_ratio", "avg_post_len", "avg_comment_score",
    "avg_comment_len", "n_unique_subs", "is_active",
)


def _hash_chunks(*values: bytes) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(len(value).to_bytes(8, "little"))
        h.update(value)
    return h.hexdigest()


def _subreddits(value) -> list:
    if pd.isna(value) or not value:
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def account_features(records: dict[str, SourceRecord]) -> np.ndarray:
    fields = {}
    posts, comment_1, comment_2 = [], [], []
    for record in records.values():
        if record.kind == "user_field":
            fields.update(record.payload)
        elif record.kind == "post":
            posts.append(dict(record.payload))
        elif record.kind == "comment_1":
            comment_1.append(dict(record.payload))
        elif record.kind == "comment_2":
            comment_2.append(dict(record.payload))
    scores = [post.get("score", 0) or 0 for post in posts]
    comments_received = [post.get("num_comments", 0) or 0 for post in posts]
    upvotes = [post.get("upvote_ratio", 0.5) or 0.5 for post in posts]
    post_lengths = [len(post.get("posts", "") or "") for post in posts]
    comment_scores = [item.get("comment_score", 0) or 0 for item in comment_1]
    comment_lengths = [len(item.get("comment_body", "") or "") for item in comment_1]
    description = fields.get("description", "")
    if pd.isna(description):
        description = ""
    values = {
        "submission_num": fields.get("submission_num", 0) or 0,
        "comment_num": fields.get("comment_num", 0) or 0,
        "comment_num_1": fields.get("comment_num_1", 0) or 0,
        "comment_num_2": fields.get("comment_num_2", 0) or 0,
        "has_description": int(bool(description)),
        "description_len": len(description),
        "n_subreddits": len(_subreddits(fields.get("subreddit", ""))),
        "n_posts": len(posts),
        "n_comments_received": int(np.sum(comments_received)) if comments_received else 0,
        "n_comments_made": len(comment_1) + len(comment_2),
        "avg_post_score": float(np.mean(scores)) if scores else 0.0,
        "avg_upvote_ratio": float(np.mean(upvotes)) if upvotes else 0.5,
        "avg_post_len": float(np.mean(post_lengths)) if post_lengths else 0.0,
        "avg_comment_score": float(np.mean(comment_scores)) if comment_scores else 0.0,
        "avg_comment_len": float(np.mean(comment_lengths)) if comment_lengths else 0.0,
        "n_unique_subs": len(set(post.get("subreddit", "") for post in posts)),
        "is_active": int(bool(posts or comment_1 or comment_2)),
    }
    return np.array([float(values[key]) for key in FEATURE_FIELDS], dtype=np.float32)


def load_botsim_adapter(users_csv: Path, posts_json: Path, features_csv: Path,
                        checkpoint_path: Path, target_account_id: str,
                        *, knn_k: int = 10, device: str = "cpu") -> tuple[RebuildingAdapter, dict[str, SourceRecord]]:
    users = pd.read_csv(users_csv)
    ids = users["user_id"].astype(str).tolist()
    if target_account_id not in set(ids) or len(ids) != len(set(ids)):
        raise ValueError("BotSim account ID missing or duplicated")
    row_index = ids.index(target_account_id)
    frame = pd.read_csv(features_csv)
    if list(frame.columns) != list(FEATURE_FIELDS) or len(frame) != len(ids):
        raise ValueError("BotSim feature table does not align with source accounts")
    sources: dict[str, SourceRecord] = {}
    row = users.iloc[row_index]
    for field in CSV_KEYS:
        if field not in row or pd.isna(row[field]):
            continue
        sid = f"botsim:{target_account_id}:user:{field}"
        sources[sid] = SourceRecord(sid, target_account_id, "user_field", {field: row[field]},
                                    f"{users_csv.resolve()}#row={row_index + 2};{field}")
    with posts_json.open("r", encoding="utf-8") as stream:
        activity = json.load(stream)[target_account_id]
    for kind in ("posts", "comment_1", "comment_2"):
        for ordinal, item in enumerate(activity.get(kind, [])):
            item_id = str(item.get("submission_id", item.get("comment_id", ordinal)))
            sid = f"botsim:{target_account_id}:{kind}:{item_id}:{ordinal}"
            sources[sid] = SourceRecord(sid, target_account_id,
                                        "post" if kind == "posts" else kind,
                                        item, f"{posts_json.resolve()}#{target_account_id}/{kind}/{ordinal}")
    baseline_raw = frame.to_numpy(dtype=np.float32)
    if not np.allclose(account_features(sources), baseline_raw[row_index], rtol=1e-5, atol=1e-5):
        raise ValueError("BotSim raw record does not regenerate its feature row")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean"]
    scaler.scale_ = checkpoint["scaler_scale"]
    scaler.n_features_in_ = len(FEATURE_FIELDS)
    projection = checkpoint["projection"]
    model = EvidenceGatedDetector(checkpoint["feature_dim"], checkpoint["graph_dim"],
                                  checkpoint["num_relations"], checkpoint["hidden"],
                                  checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    def build(remaining: dict[str, SourceRecord]) -> BuiltInputs:
        raw = baseline_raw.copy()
        if len(remaining) != len(sources):
            raw[row_index] = account_features(remaining)
        x = torch.from_numpy(np.clip(scaler.transform(raw), -3, 3).astype(np.float32))
        edge, edge_type = knn_edges(x, knn_k)
        graph_x = x @ projection if projection is not None else x
        neighbor, degree = relation_means(graph_x, edge, edge_type, 1)
        x_hash = _hash_chunks(x.numpy().tobytes())
        graph_hash = _hash_chunks(edge.numpy().tobytes(), edge_type.numpy().tobytes())
        if len(remaining) == len(sources):
            if x_hash != checkpoint["hashes"]["feature_sha256"]:
                raise ValueError("BotSim features differ from detector training input")
            if graph_hash != checkpoint["hashes"]["graph_sha256"]:
                raise ValueError("BotSim kNN differs from detector training graph")
        dependencies = {sid: (f"features[{row_index}].{next(iter(record.payload)) if record.kind == 'user_field' else record.kind}",
                              "knn_graph[all_accounts]", "neighbor_means[all_accounts]")
                        for sid, record in remaining.items()}
        fingerprint = digest({"feature_hash": x_hash, "graph_hash": graph_hash,
                              "neighbor_hash": hashlib.sha256(neighbor.numpy().tobytes()).hexdigest()})
        return BuiltInputs((x.to(device), neighbor.to(device), degree.to(device)),
                           dependencies, fingerprint)

    def predict(uid: str, built) -> list[float]:
        if uid not in ids:
            raise KeyError(f"unknown BotSim account {uid}")
        x, neighbor, degree = built
        index = ids.index(uid)
        with torch.inference_mode():
            logit = float(model(x[index:index + 1], neighbor[index:index + 1],
                                degree[index:index + 1])[0])
        return [0.0, logit]

    adapter = RebuildingAdapter(
        model_name="VeriBot-v2-BotSim",
        model_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        preprocessor_sha256=digest({"features": FEATURE_FIELDS, "knn_k": knn_k,
                                    "training_feature_hash": checkpoint["hashes"]["feature_sha256"]}),
        sources=sources, build_inputs=build, predict=predict)
    return adapter, sources
