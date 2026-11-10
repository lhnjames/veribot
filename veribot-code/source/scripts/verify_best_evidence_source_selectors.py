"""Resolve every full-test evidence-card source selector to published records."""

from __future__ import annotations

import csv
import gzip
import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
CARDS = ROOT / "output/v3_best_verified_evidence_full_test"
CERTIFICATE = ROOT / "research/certificates/best_verified_evidence_full_test.json"
OUT = ROOT / "research/certificates/best_evidence_source_selector_resolution.json"


@lru_cache(maxsize=32)
def _is_file(path: Path) -> bool:
    return path.is_file()


def raw_ids() -> tuple[dict[str, set[str]], dict[str, int], int]:
    users = ROOT / "data/raw/botsim_24/Users.csv"
    activity = ROOT / "data/raw/botsim_24/user_post_comment.json"
    with users.open("r", encoding="utf-8-sig", newline="") as stream:
        bot_users = {str(row["user_id"]) for row in csv.DictReader(stream)}
    bot_activity = set(json.loads(activity.read_text(encoding="utf-8")))
    fox = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    pattern = re.compile(rb'^\{"user_id"\s*:\s*"?(\d+)"?')
    fox_users: set[str] = set()
    first_lines: dict[str, int] = {}
    duplicate_rows = 0
    with gzip.open(fox, "rb") as stream:
        for line_number, line in enumerate(stream, 1):
            match = pattern.match(line)
            if match is None:
                raise ValueError("Fox8 raw line lacks leading user ID")
            uid = match.group(1).decode("ascii")
            if uid in fox_users:
                duplicate_rows += 1
            else:
                first_lines[uid] = line_number
            fox_users.add(uid)
    if len(bot_users) != 2907 or bot_users != bot_activity or len(fox_users) != 2279:
        raise ValueError("published account populations differ")
    return {"botsim": bot_users, "fox8": fox_users}, first_lines, duplicate_rows


def main() -> None:
    export = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    populations, fox_first_lines, fox_duplicate_rows = raw_ids()
    tensor_dir = ROOT / "data/raw/mgtab_source.json"
    source = json.loads(tensor_dir.read_text(encoding="utf-8"))
    source_dir = Path(source["path"])
    features = torch.load(source_dir / "features.pt", map_location="cpu",
                          weights_only=True)
    edge = torch.load(source_dir / "edge_index.pt", map_location="cpu",
                      weights_only=True)
    kind = torch.load(source_dir / "edge_type.pt", map_location="cpu",
                      weights_only=True)
    if (tuple(features.shape) != (10199, 788) or edge.shape[0] != 2 or
            edge.shape[1] != len(kind)):
        raise ValueError("MGTAB published source tensor shapes differ")
    del features, edge, kind
    projection_paths = set()
    cache_paths = set()
    reference_counts = Counter()
    distinct_ids = {"botsim": set(), "fox8": set(), "mgtab": set()}
    cards_seen = 0
    terms_seen = 0
    for run in export["runs"]:
        dataset, seed = run["dataset"], int(run["seed"])
        path = CARDS / f"{dataset}_seed{seed}.jsonl.gz"
        if sha256_file(path) != run["cards_sha256"]:
            raise ValueError("source card hash differs")
        row_count = 0
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                card = json.loads(line)
                uid = str(card["account_id"])
                if dataset == "mgtab":
                    if not uid.isdecimal() or not 0 <= int(uid) < 10199:
                        raise ValueError("MGTAB account is not a published row")
                elif uid not in populations[dataset]:
                    raise ValueError("card account absent from raw records")
                distinct_ids[dataset].add(uid)
                for term in card["evidence_terms"]:
                    terms_seen += 1
                    query = term["source_query"]
                    if query is None:
                        if term["source_fact_observed"]:
                            raise ValueError("observed fact has no locator")
                        continue
                    ids = query["source_ids"]
                    if dataset == "mgtab" and any(
                            "#matching_edge_rows" in value for value in ids):
                        if not any("#incident_account=" + uid in value
                                   for value in ids):
                            raise ValueError("edge-type selector lacks account endpoint")
                    for value in ids:
                        name, separator, fragment = value.partition("#")
                        file = Path(name)
                        if not separator or not _is_file(file):
                            raise ValueError("selector lacks a source file or fragment")
                        if dataset == "botsim":
                            if fragment.startswith("user_id="):
                                if file.name != "Users.csv" or fragment[8:] != uid:
                                    raise ValueError("BotSim user selector differs")
                            elif fragment.startswith("account_id="):
                                if (file.name != "user_post_comment.json" or
                                        fragment[11:] != uid):
                                    raise ValueError("BotSim activity selector differs")
                            elif fragment == "all_user_names":
                                if file.name != "Users.csv":
                                    raise ValueError("BotSim name index differs")
                            elif fragment == "all_comments_and_parent_objects,select_account=" + uid:
                                if file.name != "user_post_comment.json":
                                    raise ValueError("BotSim reply activity differs")
                            elif fragment == "incident_account=" + uid:
                                if file.name != "reply_sources.jsonl.gz":
                                    raise ValueError("BotSim reply edge source differs")
                            else:
                                raise ValueError("unknown BotSim source selector")
                        elif dataset == "fox8":
                            if fragment.startswith("line="):
                                match = re.fullmatch(r"line=(\d+),user_id=(\d+)",
                                                     fragment)
                                if (file.name != "fox8_23_dataset.ndjson.gz" or
                                        match is None or
                                        fox_first_lines.get(match.group(2)) !=
                                        int(match.group(1))):
                                    raise ValueError("Fox8 first raw row differs")
                            elif fragment in {
                                    "all_interaction_events,select_account=" + uid,
                                    "incident_account=" + uid}:
                                expected = ("fox8_23_dataset.ndjson.gz"
                                    if fragment.startswith("all_interaction") else
                                    "interaction_event_index.json")
                                if file.name != expected:
                                    raise ValueError("Fox8 native event source differs")
                            else:
                                raise ValueError("unknown Fox8 source selector")
                        else:
                            if fragment.startswith("row="):
                                parts = fragment.split(",", 1)
                                if int(parts[0][4:]) != int(uid):
                                    raise ValueError("MGTAB row selector differs")
                                if len(parts) == 2:
                                    if parts[1] == "columns=20:788":
                                        if file.name != "features.pt":
                                            raise ValueError("content source differs")
                                    elif parts[1].startswith("column="):
                                        column = int(parts[1][7:])
                                        if (file.name != "features.pt" or
                                                column != int(term["name"].rsplit("_", 1)[1])):
                                            raise ValueError("profile source column differs")
                                    else:
                                        raise ValueError("unknown MGTAB row column")
                                else:
                                    if file.suffix != ".npz":
                                        raise ValueError("MGTAB cache row differs")
                                    cache_paths.add(file)
                            elif fragment == "incident_account=" + uid:
                                if file.name != "edge_index.pt":
                                    raise ValueError("MGTAB edge endpoint differs")
                            elif fragment == "matching_edge_rows":
                                if file.name != "edge_type.pt":
                                    raise ValueError("MGTAB edge type differs")
                            elif fragment.startswith("neighbor_rows,column="):
                                column = int(fragment.split("=", 1)[1])
                                suffixes = {1: "default_profile_mean",
                                            2: "verified_mean",
                                            13: "description_length_mean",
                                            14: "followers_friends_ratio_mean"}
                                if (file.name != "features.pt" or
                                        not term["name"].endswith(
                                            "_neighbor_" + suffixes.get(column, "?"))):
                                    raise ValueError("MGTAB neighbor profile column differs")
                            elif fragment == "train_only_projection":
                                if file.name != "projection.npz":
                                    raise ValueError("MGTAB projection source differs")
                                projection_paths.add(file)
                            else:
                                raise ValueError("unknown MGTAB source selector")
                        reference_counts[dataset, fragment.split("=", 1)[0]] += 1
                cards_seen += 1
                row_count += 1
        if row_count != run["accounts"]:
            raise ValueError("source card run count differs")
    for file in projection_paths:
        with np.load(file, allow_pickle=False) as data:
            if data["pca_components"].shape != (64, 768):
                raise ValueError("training projection source differs")
    for file in cache_paths:
        with np.load(file, allow_pickle=False) as data:
            if data["ids"].astype(str).tolist() != [str(i) for i in range(10199)]:
                raise ValueError("MGTAB derived fact cache row order differs")
    if cards_seen != export["total_accounts"] or terms_seen != export[
            "total_evidence_terms"]:
        raise ValueError("source selector audit population differs")
    OUT.write_text(json.dumps({
        "status": "all_evidence_source_selectors_resolve_to_published_records",
        "source_card_certificate_sha256": sha256_file(CERTIFICATE),
        "cards": cards_seen, "terms": terms_seen,
        "file_references": sum(reference_counts.values()),
        "distinct_card_accounts": {key: len(value)
                                   for key, value in distinct_ids.items()},
        "source_populations": {key: len(value)
                               for key, value in populations.items()},
        "fox8_duplicate_raw_account_rows": fox_duplicate_rows,
        "validated_projection_artifacts": len(projection_paths),
        "validated_derived_fact_caches": len(cache_paths),
        "reference_counts": {f"{dataset}:{selector}": count for
                             (dataset, selector), count in sorted(
                                 reference_counts.items())},
        "scope": "selector syntax, published account existence and tensor row/column bounds; not behavioral semantic truth",
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
