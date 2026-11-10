from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .schema import AccountRecord, GraphEdge, PostRecord


def post_kind(text: str) -> str:
    text = text.lstrip()
    if text.startswith("RT @"):
        return "retweet"
    if text.startswith("@"):
        return "reply"
    return "original"


def select_posts(posts: list[str], limit: int) -> list[PostRecord]:
    """Deterministic diversity selection; never reads labels."""
    cleaned = [re.sub(r"\s+", " ", str(x)).strip() for x in posts]
    cleaned = [x for x in cleaned if x]
    buckets = {"original": [], "retweet": [], "reply": []}
    for i, text in enumerate(cleaned):
        buckets[post_kind(text)].append((i, text))
    chosen: list[tuple[int, str]] = []
    quota = max(1, limit // 3)
    for key in ("original", "retweet", "reply"):
        chosen.extend(buckets[key][:quota])
    selected_ids = {i for i, _ in chosen}
    chosen.extend((i, t) for i, t in enumerate(cleaned) if i not in selected_ids)
    chosen = chosen[:limit]
    return [PostRecord(post_id=f"p{i}", text=t, kind=post_kind(t)) for i, t in chosen]


def internal_split(uid: str) -> str:
    """Deterministic 5% holdout derived exclusively from the official training split."""
    return "validation" if int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) % 20 == 0 else "train"


def load_hgt_graph_sidecars(paths: list[Path]) -> dict[tuple[str, str], dict]:
    """Load v3 JSONL sidecars keyed by (source, account_id).

    Legacy JSON maps remain readable for audit/migration, but they do not contain an upstream
    verdict and therefore cannot satisfy a formal v3 export unless --allow-missing-hgt-verdict is
    explicitly requested.
    """
    out: dict[tuple[str, str], dict] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    key = (str(row["source"]), str(row["account_id"]))
                    if key in out:
                        raise ValueError(f"duplicate HGT sidecar entry: {key}")
                    out[key] = {
                        "facts": list(row.get("graph_facts") or []),
                        "verdict": row.get("upstream_verdict"),
                    }
        else:
            raw = json.loads(path.read_text(encoding="utf-8"))
            inferred_source = "twibot20" if "twibot20" in str(path).lower() else "twibot22"
            for account_id, fields in raw.items():
                texts = [fields[k] for k in sorted(fields) if str(k).upper().startswith("GRAPH_") and fields.get(k)]
                out[(inferred_source, str(account_id))] = {"facts": texts, "verdict": None}
    return out


def bucket_profile(profile: dict, config: dict) -> dict[str, str]:
    labels = config["labels"]
    result = {}
    for field, spec in config["fields"].items():
        value = max(as_number(profile.get(field, 0)), 0.0)
        transformed = float(np.log1p(value))
        cuts = spec["cutpoints_log1p"]
        if spec["strategy"] == "zero_aware_log_quantile":
            index = 0 if value == 0 else int(np.searchsorted(cuts, transformed, side="right")) + 1
        else:
            index = int(np.searchsorted(cuts, transformed, side="right"))
        result[field] = labels[min(index, len(labels) - 1)]
    return result


def content_key(text: str) -> str | None:
    normalized = re.sub(r"https?://\S+", "<URL>", re.sub(r"\s+", " ", text.lower())).strip()
    if post_kind(text) == "retweet" or len(re.sub(r"\W", "", normalized)) < 20:
        return None
    return hashlib.sha1(normalized.encode()).hexdigest()[:16]


def as_number(value) -> float:
    try: return float(str(value).strip() or 0)
    except (TypeError, ValueError): return 0.0


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def twibot20_records(
    root: Path, max_posts: int, hgt_sidecar: dict[tuple[str, str], dict], bucket_config: dict,
    allow_missing_verdict: bool = False,
) -> Iterable[AccountRecord]:
    for filename, split in (("train.json", "train"),):
        users = json.loads((root / filename).read_text(encoding="utf-8"))
        owners: dict[str, list[str]] = {}
        for user in users:
            uid = str(user.get("ID", "")).strip()
            for post in select_posts(user.get("tweet") or [], max_posts):
                key = content_key(post.text)
                if key and uid not in owners.setdefault(key, []) and len(owners[key]) < 6: owners[key].append(uid)
        for user in users:
            uid = str(user.get("ID", "")).strip()
            raw_profile = user.get("profile") or {}
            # Strict intersection with the fields available in the train-only TwiBot-22 export.
            profile = {"description": str(raw_profile.get("description", "")).strip(),
                       "followers_count": as_number(raw_profile.get("followers_count", 0)),
                       "following_count": as_number(raw_profile.get("friends_count", 0)),
                       "tweet_count": as_number(raw_profile.get("statuses_count", 0)),
                       "listed_count": as_number(raw_profile.get("listed_count", 0)),
                       "verified": as_bool(raw_profile.get("verified", False))}
            edges: list[GraphEdge] = []
            neighbor = user.get("neighbor") or {}
            for relation, values in neighbor.items():
                if not isinstance(values, list):
                    continue
                for j, target in enumerate(values[:24]):
                    edges.append(GraphEdge(edge_id=f"e{len(edges)}", source="self", target=f"n{j}", relation=str(relation)))
            label = "bot" if str(user.get("label", "")).lower() in {"1", "bot"} else "human"
            selected = select_posts(user.get("tweet") or [], max_posts)
            facts = ["A selected original post has the same normalized content as posts from other training accounts."
                     for p in selected if (k := content_key(p.text)) and len(owners.get(k, [])) > 1]
            hgt = hgt_sidecar.get(("twibot20", uid), {})
            if not allow_missing_verdict and hgt.get("verdict") not in {"bot", "human"}:
                raise ValueError(f"missing TwiBot-20 HGT verdict for {uid}")
            yield AccountRecord(account_id=uid, source="twibot20", split=internal_split(uid), profile=profile,
                                posts=selected, graph_edges=edges, graph_facts=facts,
                                hgt_neighbor_facts=hgt.get("facts", []), upstream_verdict=hgt.get("verdict"),
                                profile_buckets=bucket_profile(profile, bucket_config),
                                bucket_version=bucket_config["version"], label=label)


def twibot22_records(
    root: Path, max_posts: int, max_neighbors: int, hgt_sidecar: dict[tuple[str, str], dict],
    bucket_config: dict, allow_missing_verdict: bool = False,
) -> Iterable[AccountRecord]:
    with open(root / "user_tweets.pkl", "rb") as f: tweets = pickle.load(f)
    with open(root / "description.pkl", "rb") as f: descriptions = pickle.load(f)
    num = np.load(root / "num_properties_raw.npy", mmap_mode="r")
    verified = torch.load(root / "cat_properties_tensor.pt", map_location="cpu")[:, 0]
    edge_index = torch.load(root / "edge_index.pt", map_location="cpu").long()
    edge_type = torch.load(root / "edge_type.pt", map_location="cpu").long()
    uid_to_idx = torch.load(root / "uid_to_idx.pt", map_location="cpu")
    labels = torch.load(root / "label.pt", map_location="cpu")
    boundaries = json.loads((root / "boundaries.json").read_text())
    idx_to_uid = {int(i): str(uid) for uid, i in uid_to_idx.items()}
    owners: dict[str, list[int]] = {}
    for idx, raw_posts in tweets.items():
        for post in select_posts(raw_posts, max_posts):
            key = content_key(post.text)
            if key and int(idx) not in owners.setdefault(key, []) and len(owners[key]) < 6:
                owners[key].append(int(idx))
    adjacency: dict[int, list[tuple[int, str]]] = {}
    seen_edges: dict[int, set[tuple[int, str]]] = {}
    def add_neighbor(account: int, neighbor: int, relation: str) -> None:
        key = (neighbor, relation)
        if key in seen_edges.setdefault(account, set()) or len(adjacency.setdefault(account, [])) >= max_neighbors:
            return
        seen_edges[account].add(key)
        adjacency[account].append(key)
    for k in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, k]), int(edge_index[1, k])
        rel = int(edge_type[k])
        if rel == 0:  # stored src follows dst
            add_neighbor(src, dst, "following")
            add_neighbor(dst, src, "follower")
        else:         # stored row denotes that dst follows src
            add_neighbor(src, dst, "follower")
            add_neighbor(dst, src, "following")
    names = ["followers_count", "following_count", "tweet_count", "listed_count", "active_days"]
    for i in range(len(labels)):
        split = internal_split(idx_to_uid[i])
        profile = {name: float(num[i, j]) for j, name in enumerate(names[:4])}
        profile["verified"] = bool(verified[i])
        profile["description"] = descriptions[i]
        edges = [GraphEdge(edge_id=f"e{j}", source="self", target=f"n{j}", relation=rel)
                 for j, (_, rel) in enumerate(adjacency.get(i, []))]
        selected = select_posts(tweets.get(i, []), max_posts)
        facts = []
        facts.extend("A selected original post has the same normalized content as posts from other training accounts."
                     for p in selected if (k := content_key(p.text)) and len(owners.get(k, [])) > 1)
        uid = idx_to_uid[i]
        hgt = hgt_sidecar.get(("twibot22", uid), {})
        if not allow_missing_verdict and hgt.get("verdict") not in {"bot", "human"}:
            raise ValueError(f"missing TwiBot-22 HGT verdict for {uid}")
        yield AccountRecord(account_id=idx_to_uid[i], source="twibot22", split=split, profile=profile,
                            posts=selected, graph_edges=edges,
                            graph_facts=facts, hgt_neighbor_facts=hgt.get("facts", []),
                            upstream_verdict=hgt.get("verdict"),
                            profile_buckets=bucket_profile(profile, bucket_config),
                            bucket_version=bucket_config["version"],
                            label="bot" if int(labels[i]) else "human")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--twibot20", type=Path, required=True)
    ap.add_argument("--twibot22", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-posts", type=int, default=32)
    ap.add_argument("--max-neighbors", type=int, default=24)
    ap.add_argument("--hgt-sidecar", type=Path, action="append", required=True,
                    help="v3 model-visible HGT JSONL; pass once for each source")
    ap.add_argument("--bucket-thresholds", type=Path, required=True)
    ap.add_argument("--allow-missing-hgt-verdict", action="store_true",
                    help="migration/debug only; formal v3 exports must not use this")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hgt_sidecar = load_hgt_graph_sidecars(args.hgt_sidecar)
    bucket_config = json.loads(args.bucket_thresholds.read_text(encoding="utf-8"))
    writers = {s: open(args.output / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "validation", "test")}
    try:
        for source in (
            twibot20_records(args.twibot20, args.max_posts, hgt_sidecar, bucket_config,
                             args.allow_missing_hgt_verdict),
            twibot22_records(args.twibot22, args.max_posts, args.max_neighbors, hgt_sidecar,
                             bucket_config, args.allow_missing_hgt_verdict),
        ):
            for record in source:
                writers[record.split].write(record.model_dump_json() + "\n")
    finally:
        for f in writers.values(): f.close()


if __name__ == "__main__": main()
