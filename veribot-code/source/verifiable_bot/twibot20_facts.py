"""Source-indexed computable facts from the official TwiBot-20 ZIP.

Only the official train/dev/test accounts receive labels. Follow edges use
the original sampled neighbor IDs. Support accounts can supply both endpoint
identities and unlabeled edge declarations; no feature-derived kNN edges are
introduced. The released graph is a sampled-neighbor graph.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile

import ijson
import numpy as np


PROFILE_NUMERIC = ("followers_count", "friends_count", "statuses_count",
                   "listed_count", "favourites_count")
PROFILE_BOOLEAN = ("verified", "protected", "default_profile", "default_profile_image")
TWEET_FACTS = ("tweet_sample_count", "retweet_fraction", "url_fraction",
               "hashtag_fraction", "mention_fraction", "mean_tweet_length",
               "unique_tweet_fraction", "description_length")
GRAPH_FACTS = ("sampled_following_count", "sampled_follower_count",
               "graph_out_degree", "graph_in_degree", "graph_reciprocal_degree")
FACT_NAMES = PROFILE_NUMERIC + PROFILE_BOOLEAN + TWEET_FACTS + GRAPH_FACTS


def _number(value) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    return float(str(value).strip())


def _boolean(value) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    normalized = str(value).strip().lower()
    if normalized not in ("true", "false", "1", "0"):
        raise ValueError(f"unrecognized Boolean value {value!r}")
    return float(normalized in ("true", "1"))


def _fraction(count: int, total: int) -> float:
    return count / total if total else float("nan")


def account_facts(row: dict) -> tuple[np.ndarray, list[tuple[str, str]], str]:
    profile = row.get("profile") or {}
    tweets = [str(value or "") for value in (row.get("tweet") or [])]
    n = len(tweets)
    neighbor = row.get("neighbor")
    values = [_number(profile.get(field)) for field in PROFILE_NUMERIC]
    values.extend(_boolean(profile.get(field)) for field in PROFILE_BOOLEAN)
    values.extend((
        float(n),
        _fraction(sum(text.lstrip().startswith("RT @") for text in tweets), n),
        _fraction(sum("http://" in text or "https://" in text for text in tweets), n),
        _fraction(sum(bool(re.search(r"(?<!\w)#\w", text)) for text in tweets), n),
        _fraction(sum(bool(re.search(r"(?<!\w)@\w", text)) for text in tweets), n),
        float(np.mean([len(text) for text in tweets])) if n else float("nan"),
        _fraction(len(set(tweets)), n),
        float(len(str(profile.get("description") or ""))),
        float(len(neighbor.get("following") or [])) if neighbor is not None else float("nan"),
        float(len(neighbor.get("follower") or [])) if neighbor is not None else float("nan"),
    ))
    uid = str(row["ID"])
    edges = []
    if neighbor is not None:
        edges.extend((uid, str(target)) for target in neighbor.get("following") or []
                     if str(target) != uid)
        edges.extend((str(source), uid) for source in neighbor.get("follower") or []
                     if str(source) != uid)
    return np.asarray(values, dtype=float), edges, str(row.get("domain") or "unknown")


def build(archive: Path, *, include_support_neighbor_edges: bool = True) -> dict:
    path = archive.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    ids: list[str] = []
    split: list[str] = []
    labels: list[int] = []
    contexts: list[str] = []
    positions: list[int] = []
    facts: list[np.ndarray] = []
    edges: set[tuple[str, str]] = set()
    source_members = {name: f"Twibot-20/{name}.json"
                      for name in ("train", "dev", "test")}
    with zipfile.ZipFile(path) as zipped:
        for part, member in source_members.items():
            with zipped.open(member) as stream:
                for ordinal, row in enumerate(ijson.items(stream, "item")):
                    values, relation, context = account_facts(row)
                    ids.append(str(row["ID"]))
                    split.append(part)
                    labels.append(int(row["label"]))
                    contexts.append(context)
                    positions.append(ordinal)
                    facts.append(values)
                    edges.update(relation)
    if len(set(ids)) != len(ids) or len(ids) != 11826 or len(facts[0]) != len(FACT_NAMES) - 3:
        raise ValueError("TwiBot-20 population or feature vocabulary differs from audit")
    known = set(ids)
    support_member = "Twibot-20/support.json"
    support_ids: set[str] = set()
    support_edge_declarations = 0
    with zipfile.ZipFile(path) as zipped, zipped.open(support_member) as stream:
        if include_support_neighbor_edges:
            for row in ijson.items(stream, "item"):
                uid = str(row["ID"])
                if uid in support_ids:
                    raise ValueError("TwiBot-20 support ID repeats")
                support_ids.add(uid)
                neighbor = row.get("neighbor") or {}
                for other in neighbor.get("following") or []:
                    other = str(other)
                    if other != uid:
                        edges.add((uid, other))
                        support_edge_declarations += 1
                for other in neighbor.get("follower") or []:
                    other = str(other)
                    if other != uid:
                        edges.add((other, uid))
                        support_edge_declarations += 1
        else:
            support_ids = set(map(str, ijson.items(stream, "item.ID")))
    all_ids = known | support_ids
    if known & support_ids or any(a not in all_ids or b not in all_ids for a, b in edges):
        raise ValueError("TwiBot-20 edge endpoint is missing or overlaps official splits")
    indegree: Counter = Counter(b for _, b in edges)
    outdegree: Counter = Counter(a for a, _ in edges)
    reciprocal: Counter = Counter(a for a, b in edges if (b, a) in edges)
    graph_values = np.asarray([[float(outdegree[uid]), float(indegree[uid]),
                                float(reciprocal[uid])] for uid in ids])
    matrix = np.column_stack((np.asarray(facts, dtype=float), graph_values))
    if matrix.shape != (len(ids), len(FACT_NAMES)):
        raise ValueError("TwiBot-20 fact matrix dimension mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"values": matrix, "labels": np.asarray(labels, dtype=np.int8),
            "ids": np.asarray(ids), "split": np.asarray(split),
            "contexts": np.asarray(contexts), "positions": np.asarray(positions),
            "names": np.asarray(FACT_NAMES),
            "manifest": {
                "dataset": "twibot20", "archive": str(path),
                "archive_sha256": digest.hexdigest(),
                "split_counts": dict(Counter(split)),
                "support_ids": len(support_ids),
                "support_edge_declarations": support_edge_declarations,
                "directed_follow_edges": len(edges),
                "fact_names": list(FACT_NAMES),
                "source_locator": "archive member + array ordinal + account ID + profile/tweet/neighbor query",
                "graph_degree_query": (
                    "deduplicated union of all official sampled train/dev/test/support neighbor lists"
                    if include_support_neighbor_edges else
                    "deduplicated union of labeled train/dev/test sampled neighbor lists, with support IDs as endpoints"),
                "graph_scope": "officially provided sampled-neighbor declarations",
                "support_neighbor_edges_included": include_support_neighbor_edges,
                "label_access": "labels stored separately; fact extraction reads no label field",
            }}


def save(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: value for key, value in bundle.items()
                                 if key != "manifest"})
    path.with_suffix(".json").write_text(json.dumps(bundle["manifest"], indent=2) + "\n",
                                          encoding="utf-8")
