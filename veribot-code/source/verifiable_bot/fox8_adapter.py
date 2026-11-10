"""Fox8 raw-record adapter for source-consistent VeriBot v2 probes.

Feature formulas match the 31-column SAHG/HypBot Fox8 preprocessing. A tweet
edit rebuilds its aggregate fields, the full feature-derived kNN graph, and
all neighbor means before the frozen model is queried. kNN links are never
described as observed follower relations.
"""

from __future__ import annotations

from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .evidence_v2 import BuiltInputs, RebuildingAdapter, SourceRecord, digest
from .v2_detector import EvidenceGatedDetector, knn_edges, relation_means


PROFILE_FIELDS = (
    "followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count",
    "verified", "default_profile", "default_profile_image", "protected", "geo_enabled",
    "profile_use_background_image", "has_extended_profile", "contributors_enabled",
    "is_translator", "is_translation_enabled", "profile_background_tile", "description_len",
    "name_len", "screen_name_len", "has_location", "has_url", "has_description", "ff_ratio",
)
TWEET_FIELDS = (
    "n_tweets", "retweet_ratio", "reply_ratio", "avg_tweet_len", "avg_favorites",
    "avg_retweets", "avg_score", "unique_lang_ratio",
)
FEATURE_FIELDS = PROFILE_FIELDS + TWEET_FIELDS
RAW_PROFILE_KEYS = (
    "followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count",
    "verified", "default_profile", "default_profile_image", "protected", "geo_enabled",
    "profile_use_background_image", "has_extended_profile", "contributors_enabled",
    "is_translator", "is_translation_enabled", "profile_background_tile", "description",
    "name", "screen_name", "location", "url",
)


def profile_features(user: dict) -> dict[str, float]:
    numeric = ("followers_count", "friends_count", "statuses_count", "favourites_count", "listed_count")
    boolean = ("verified", "default_profile", "default_profile_image", "protected", "geo_enabled",
               "profile_use_background_image", "has_extended_profile", "contributors_enabled",
               "is_translator", "is_translation_enabled", "profile_background_tile")
    result = {key: user.get(key, 0) or 0 for key in numeric}
    result.update({key: int(bool(user.get(key, False))) for key in boolean})
    result.update({
        "description_len": len(user.get("description", "") or ""),
        "name_len": len(user.get("name", "") or ""),
        "screen_name_len": len(user.get("screen_name", "") or ""),
        "has_location": int(bool(user.get("location", ""))),
        "has_url": int(bool(user.get("url", ""))),
        "has_description": int(bool(user.get("description", ""))),
        "ff_ratio": (user.get("followers_count", 0) or 0)
                    / max((user.get("friends_count", 0) or 0) + 1, 1),
    })
    return result


def tweet_features(tweets: list[dict]) -> dict[str, float]:
    if not tweets:
        return {name: 0.0 for name in TWEET_FIELDS}
    n = len(tweets)
    retweets = sum(1 for tweet in tweets if (tweet.get("text", "") or "").startswith("RT "))
    replies = sum(1 for tweet in tweets if tweet.get("in_reply_to_status_id") is not None)
    texts = [tweet.get("text", "") or "" for tweet in tweets]
    favs = [tweet.get("favorite_count", 0) or 0 for tweet in tweets]
    rts = [tweet.get("retweet_count", 0) or 0 for tweet in tweets]
    langs = [tweet.get("lang", "und") for tweet in tweets]
    return {
        "n_tweets": n, "retweet_ratio": retweets / n, "reply_ratio": replies / n,
        "avg_tweet_len": float(np.mean([len(value) for value in texts])),
        "avg_favorites": float(np.mean(favs)), "avg_retweets": float(np.mean(rts)),
        "avg_score": float(np.mean([f + r for f, r in zip(favs, rts)])),
        "unique_lang_ratio": len(set(langs)) / n,
    }


def index_sources(raw_path: Path, account_ids: list[str]) -> dict[str, SourceRecord]:
    wanted = set(account_ids)
    found = set()
    sources: dict[str, SourceRecord] = {}
    opener = gzip.open if raw_path.suffix == ".gz" else open
    with opener(raw_path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid not in wanted or uid in found:
                continue  # SAHG/HypBot preprocessing keeps the first duplicate ID.
            found.add(uid)
            tweets = row["user_tweets"]
            profile = tweets[0]["user"]
            for key in RAW_PROFILE_KEYS:
                if key not in profile:
                    continue
                profile_id = f"fox8:{uid}:profile:{key}"
                sources[profile_id] = SourceRecord(profile_id, uid, "profile_field",
                                                   {key: profile[key]},
                                                   f"{raw_path.resolve()}#line={line_number};tweet=0;user.{key}")
            for ordinal, tweet in enumerate(tweets):
                tid = str(tweet.get("id_str", tweet.get("id", ordinal)))
                sid = f"fox8:{uid}:tweet:{tid}:{ordinal}"
                sources[sid] = SourceRecord(sid, uid, "tweet", tweet,
                                            f"{raw_path.resolve()}#line={line_number};tweet={ordinal}")
    if found != wanted:
        raise ValueError(f"raw Fox8 records missing {len(wanted - found)} feature accounts")
    return sources


def raw_feature_matrix(sources: dict[str, SourceRecord], account_ids: list[str]) -> np.ndarray:
    profiles: dict[str, dict] = {}
    tweets: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for record in sources.values():
        if record.kind == "profile_field":
            profiles.setdefault(record.account_id, {}).update(record.payload)
        elif record.kind == "tweet":
            ordinal = int(record.source_id.rsplit(":", 1)[1])
            tweets[record.account_id].append((ordinal, dict(record.payload)))
    rows = []
    for uid in account_ids:
        profile = profile_features(profiles.get(uid, {}))
        ordered_tweets = [tweet for _, tweet in sorted(tweets[uid])]
        feature = {**profile, **tweet_features(ordered_tweets)}
        rows.append([float(feature[name]) for name in FEATURE_FIELDS])
    return np.asarray(rows, dtype=np.float32)


def load_fox8_adapter(raw_path: Path, feature_csv: Path, checkpoint_path: Path,
                      target_account_id: str, *, knn_k: int = 10,
                      device: str = "cpu") -> tuple[RebuildingAdapter, dict[str, SourceRecord]]:
    frame = pd.read_csv(feature_csv)
    if list(frame.columns) != ["user_id", *FEATURE_FIELDS]:
        raise ValueError("Fox8 feature columns or their order differ from the recorded preprocessing")
    account_ids = frame["user_id"].astype(str).tolist()
    if target_account_id not in set(account_ids):
        raise KeyError(f"unknown Fox8 account {target_account_id}")
    sources = index_sources(raw_path, [target_account_id])
    baseline_raw = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
    target_index = account_ids.index(target_account_id)
    target_raw = raw_feature_matrix(sources, [target_account_id])[0]
    if not np.allclose(target_raw, baseline_raw[target_index], rtol=1e-5, atol=1e-5):
        raise ValueError("raw Fox8 account does not regenerate its supplied feature row")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mean = np.asarray(checkpoint["scaler_mean"])
    scale = np.asarray(checkpoint["scaler_scale"])
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.n_features_in_ = len(mean)
    expected_feature_hash = checkpoint["hashes"]["feature_sha256"]
    expected_graph_hash = checkpoint["hashes"]["graph_sha256"]
    projection = checkpoint["projection"]
    account_index = {uid: index for index, uid in enumerate(account_ids)}
    model = EvidenceGatedDetector(checkpoint["feature_dim"], checkpoint["graph_dim"],
                                  checkpoint["num_relations"], checkpoint["hidden"],
                                  checkpoint["dropout"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    def build(remaining: dict[str, SourceRecord]) -> BuiltInputs:
        raw = baseline_raw.copy()
        if len(remaining) != len(sources):
            raw[target_index] = raw_feature_matrix(remaining, [target_account_id])[0]
        x = torch.from_numpy(np.clip(scaler.transform(raw), -3, 3).astype(np.float32))
        edge, edge_type = knn_edges(x, knn_k)
        graph_x = x @ projection if projection is not None else x
        neighbor, degree = relation_means(graph_x, edge, edge_type, 1)
        x_hash = __import__("hashlib").sha256()
        x_hash.update(len(x.numpy().tobytes()).to_bytes(8, "little"))
        x_hash.update(x.numpy().tobytes())
        from_graph = __import__("hashlib").sha256()
        for value in (edge.numpy().tobytes(), edge_type.numpy().tobytes()):
            from_graph.update(len(value).to_bytes(8, "little"))
            from_graph.update(value)
        if len(remaining) == len(sources):
            if x_hash.hexdigest() != expected_feature_hash:
                raise ValueError("reconstructed features do not match detector training features")
            if from_graph.hexdigest() != expected_graph_hash:
                raise ValueError("reconstructed kNN graph does not match detector training graph")
        dependencies = {}
        for sid, record in remaining.items():
            row = target_index
            field_group = next(iter(record.payload)) if record.kind == "profile_field" else "tweet_aggregate"
            dependencies[sid] = (f"features[{row}].{field_group}",
                                 "knn_graph[all_accounts]", "neighbor_means[all_accounts]")
        fingerprint = digest({"feature_hash": x_hash.hexdigest(),
                              "graph_hash": from_graph.hexdigest(),
                              "neighbor_hash": __import__("hashlib").sha256(neighbor.numpy().tobytes()).hexdigest()})
        return BuiltInputs((x.to(device), neighbor.to(device), degree.to(device)),
                           dependencies, fingerprint)

    def predict(uid: str, built) -> list[float]:
        if uid not in account_index:
            raise KeyError(f"unknown Fox8 account {uid}")
        x, neighbor, degree = built
        idx = account_index[uid]
        with torch.inference_mode():
            logit = float(model(x[idx:idx + 1], neighbor[idx:idx + 1], degree[idx:idx + 1])[0])
        return [0.0, logit]

    adapter = RebuildingAdapter(
        model_name="VeriBot-v2-Fox8", model_sha256=__import__("hashlib").sha256(checkpoint_path.read_bytes()).hexdigest(),
        preprocessor_sha256=digest({"features": FEATURE_FIELDS, "knn_k": knn_k,
                                    "training_feature_hash": expected_feature_hash}),
        sources=sources, build_inputs=build, predict=predict)
    return adapter, sources
