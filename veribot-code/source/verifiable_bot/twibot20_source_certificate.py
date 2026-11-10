"""Recompute a TwiBot-20 fact from its official ZIP and cached source index."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

import ijson
import numpy as np

from .twibot20_facts import (FACT_NAMES, GRAPH_FACTS, PROFILE_BOOLEAN,
                            PROFILE_NUMERIC, account_facts)


PARTS = ("train", "dev", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member(part: str) -> str:
    if part not in PARTS:
        raise ValueError("not an official labeled TwiBot-20 split")
    return f"Twibot-20/{part}.json"


def direct_query(part: str, ordinal: int, fact_name: str, row: dict) -> dict:
    prefix = f"{_member(part)}[{ordinal}]"
    if fact_name in PROFILE_NUMERIC or fact_name in PROFILE_BOOLEAN:
        return {"query": f"{prefix}.profile.{fact_name}",
                "operation": "parse numeric value" if fact_name in PROFILE_NUMERIC
                             else "parse Boolean value"}
    if fact_name == "description_length":
        return {"query": f"{prefix}.profile.description",
                "operation": "length of description string; missing counts as empty"}
    if fact_name in GRAPH_FACTS[:2]:
        field = "following" if fact_name == "sampled_following_count" else "follower"
        return {"query": f"{prefix}.neighbor.{field}[*]",
                "operation": "sampled list length; missing neighbor object is unavailable"}
    operations = {
        "tweet_sample_count": "number of sampled tweet strings",
        "retweet_fraction": "fraction whose left-stripped text starts with RT @",
        "url_fraction": "fraction containing http:// or https://",
        "hashtag_fraction": "fraction matching (?<!\\w)#\\w",
        "mention_fraction": "fraction matching (?<!\\w)@\\w",
        "mean_tweet_length": "mean length of sampled tweet strings",
        "unique_tweet_fraction": "unique sampled tweet strings divided by sample size",
    }
    return {"query": f"{prefix}.tweet[*]", "operation": operations[fact_name],
            "tweet_sample_size": len(row.get("tweet") or [])}


def row_at(zipped: zipfile.ZipFile, part: str, ordinal: int, uid: str) -> dict:
    with zipped.open(_member(part)) as stream:
        for index, row in enumerate(ijson.items(stream, "item")):
            if index == ordinal:
                if str(row["ID"]) != uid:
                    raise ValueError("source locator account ID differs from ZIP row")
                return row
    raise ValueError("source locator ordinal is outside ZIP member")


def graph_for_account(zipped: zipfile.ZipFile, uid: str) -> tuple[dict[str, float], dict]:
    """Scan original neighbor declarations; deduplicate directed edges by endpoints."""
    edge_declarations: dict[tuple[str, str], list[str]] = defaultdict(list)
    for part in PARTS:
        with zipped.open(_member(part)) as stream:
            for ordinal, row in enumerate(ijson.items(stream, "item")):
                source_id = str(row["ID"])
                neighbor = row.get("neighbor") or {}
                for field, reverse in (("following", False), ("follower", True)):
                    for position, neighbor_id in enumerate(neighbor.get(field) or []):
                        other = str(neighbor_id)
                        edge = (other, source_id) if reverse else (source_id, other)
                        if other == source_id or uid not in edge:
                            continue
                        locator = f"{_member(part)}[{ordinal}].neighbor.{field}[{position}]"
                        edge_declarations[edge].append(locator)
    out = {b for a, b in edge_declarations if a == uid}
    incoming = {a for a, b in edge_declarations if b == uid}
    reciprocal = out & incoming
    encoded = json.dumps(
        [[a, b, sorted(refs)] for (a, b), refs in sorted(edge_declarations.items())],
        separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    values = {"graph_out_degree": float(len(out)),
              "graph_in_degree": float(len(incoming)),
              "graph_reciprocal_degree": float(len(reciprocal))}
    source = {
        "query": "deduplicated directed edges involving account from all labeled neighbor arrays",
        "out_edges": len(out), "in_edges": len(incoming),
        "reciprocal_neighbors": len(reciprocal),
        "source_declarations": sum(map(len, edge_declarations.values())),
        "source_declarations_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_locator_format": "ZIP member + row ordinal + neighbor field + list index",
    }
    return values, source


def certificate(archive: Path, facts_path: Path, account_id: str,
                fact_name: str) -> dict:
    if fact_name not in FACT_NAMES:
        raise ValueError("fact is not in the fixed TwiBot-20 vocabulary")
    source_manifest = json.loads(facts_path.with_suffix(".json").read_text(encoding="utf-8"))
    if source_manifest["archive_sha256"] != sha256(archive):
        raise ValueError("ZIP differs from fact-cache source manifest")
    with np.load(facts_path, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        matches = np.flatnonzero(ids == str(account_id))
        if len(matches) != 1:
            raise ValueError("account ID absent or duplicated in fact cache")
        index = int(matches[0])
        names = cache["names"].astype(str).tolist()
        if names != list(FACT_NAMES):
            raise ValueError("fact cache vocabulary differs from verifier")
        part = str(cache["split"][index])
        ordinal = int(cache["positions"][index])
        cached_value = float(cache["values"][index, names.index(fact_name)])
    with zipfile.ZipFile(archive) as zipped:
        row = row_at(zipped, part, ordinal, str(account_id))
        if fact_name in GRAPH_FACTS[2:]:
            graph_values, graph_source = graph_for_account(zipped, str(account_id))
            source_value = graph_values[fact_name]
            query = graph_source
        else:
            direct, _, _ = account_facts(row)
            source_value = float(direct[names.index(fact_name)])
            query = direct_query(part, ordinal, fact_name, row)
    matched = ((np.isnan(source_value) and np.isnan(cached_value)) or
               (np.isfinite(source_value) and np.isfinite(cached_value) and
                abs(source_value - cached_value) <= 1e-12))
    return {
        "dataset": "twibot20", "account_id": str(account_id),
        "fact_name": fact_name, "split": part, "source_ordinal": ordinal,
        "provenance_level": "official_raw_zip_record_and_native_neighbor_relation",
        "source_archive_sha256": source_manifest["archive_sha256"],
        "fact_cache_sha256": sha256(facts_path),
        "source_value": source_value if np.isfinite(source_value) else None,
        "cached_value": cached_value if np.isfinite(cached_value) else None,
        "missing": bool(np.isnan(source_value)), "source_query": query,
        "status": "verified" if matched else "mismatch",
    }
