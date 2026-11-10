"""Exhaustively replay coherent tweet deletions for a seven-source GNN edge."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/scripts"))
from audit_hypbot_fox8_raw_edge_response import (  # noqa: E402
    DATA, EVENT_INDEX, RAW, REPO, RUN,
)
from audit_hypbot_fox8_relation_tweet_deletion import (  # noqa: E402
    graph_after_tweet_deletions, raw_features,
)
from export_hypbot_native_graph_paired import native_symmetric_graph  # noqa: E402
from train_v2_detector import file_sha256, load_data  # noqa: E402
from verifiable_bot.fox8_adapter import FEATURE_FIELDS  # noqa: E402

ACCOUNT = "415566147"
POINTER_CERT = ROOT / "research/certificates/hypbot_fox8_seven_pointer_minimality.json"
FULL_TWEET_CERT = ROOT / "research/certificates/hypbot_fox8_all_incident_tweet_deletion.json"
OUT = ROOT / "research/certificates/hypbot_fox8_seven_tweet_subset_deletion.json"


def main() -> None:
    sys.path.insert(0, str(REPO.resolve()))
    from hypbot.data import mean_adjacency
    from hypbot.model import HypBot
    from train import PRESETS

    torch.set_num_threads(4)
    pointer = json.loads(POINTER_CERT.read_text(encoding="utf-8"))
    full_tweet = json.loads(FULL_TWEET_CERT.read_text(encoding="utf-8"))
    event_index = json.loads(EVENT_INDEX.read_text(encoding="utf-8"))
    frozen = json.loads((RUN / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = frozen["manifest"]
    checkpoint_path = RUN / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    args = argparse.Namespace(dataset="fox8", data_dir=DATA, account_file=None,
                              raw_source=[RAW], knn_k=10)
    x, _, split, ids, _, _, hashes, scaler = load_data(args, 0)
    edge, graph_meta = native_symmetric_graph(ROOT, "fox8", ids)
    if (pointer["account_id"] != ACCOUNT
            or pointer["raw_source_sha256"] != file_sha256(RAW)
            or pointer["source_index_sha256"] != file_sha256(EVENT_INDEX)
            or pointer["graph_sha256"] != graph_meta["graph_sha256"]
            or pointer["model_sha256"] != file_sha256(checkpoint_path)
            or full_tweet["raw_source_sha256"] != file_sha256(RAW)
            or full_tweet["event_index_sha256"] != file_sha256(EVENT_INDEX)
            or full_tweet["model_sha256"] != file_sha256(checkpoint_path)
            or hashes["feature_sha256"] != manifest["node_feature_sha256"]
            or hashes["split_sha256"] != manifest["split_sha256"]
            or checkpoint["split_sha256"] != hashes["split_sha256"]
            or graph_meta != manifest["native_graph"]):
        raise ValueError("Frozen model or source contract differs")
    index = {uid: i for i, uid in enumerate(ids)}
    target = index[ACCOUNT]
    if target not in set(map(int, split["test"])):
        raise ValueError("Selected account is not in frozen test population")
    frame = pd.read_csv(DATA / "features.csv")
    if list(frame.columns) != ["user_id", *FEATURE_FIELDS]:
        raise ValueError("Feature columns differ")
    raw_x = frame[list(FEATURE_FIELDS)].fillna(0.0).to_numpy(dtype=np.float32)
    locators = pointer["raw_tweet_locators"]
    line_number = int(pointer["raw_account_line"])
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        raw_row = next(json.loads(line) for i, line in enumerate(stream, 1)
                       if i == line_number)
    if str(raw_row["user_id"]) != ACCOUNT:
        raise ValueError("Raw account row differs")
    tweets = raw_row["user_tweets"]
    for loc in locators:
        tweet = tweets[int(loc["tweet_index"])]
        tweet_id = str(tweet.get("id_str") or tweet.get("id"))
        if tweet_id != str(loc["tweet_id"]):
            raise ValueError("Tweet locator differs")
    before = raw_features(tweets)
    if not np.allclose(before, raw_x[target], rtol=1e-5, atol=1e-5):
        raise ValueError("Raw account features do not replay saved features")
    baseline_edge, removed = graph_after_tweet_deletions(event_index, index, set())
    if removed or not torch.equal(baseline_edge, edge):
        raise ValueError("Source events do not replay saved graph")

    model = HypBot(x.shape[1], K=manifest["num_sectors"], gated=False,
                   **PRESETS["fox8"]["model"], graph_width=x.shape[1],
                   sparse_adj=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    threshold = float(manifest["threshold"])
    target_full = next(r for r in full_tweet["rows"] if r["account_id"] == ACCOUNT)
    expected_locators = {(int(r["line"]), int(r["tweet_index"]))
                         for r in target_full["raw_tweet_locators_deleted"]}
    observed_locators = {(int(loc["line"]), int(loc["tweet_index"]))
                         for loc in locators}
    if expected_locators != observed_locators:
        raise ValueError("Full-tweet deletion used a different source set")

    rows = []
    for size in range(len(locators) + 1):
        for subset in combinations(range(len(locators)), size):
            deleted = {(line_number, int(locators[i]["tweet_index"])) for i in subset}
            surviving = [tweet for j, tweet in enumerate(tweets)
                         if (line_number, j) not in deleted]
            after = raw_features(surviving, tweets[0]["user"])
            changed = [FEATURE_FIELDS[j] for j in range(len(FEATURE_FIELDS))
                       if before[j] != after[j]]
            edited_raw_x = raw_x.copy()
            edited_raw_x[target] = after
            edited_x = torch.from_numpy(np.clip(
                scaler.transform(edited_raw_x), -3, 3).astype(np.float32))
            edited_edge, removed_events = graph_after_tweet_deletions(
                event_index, index, deleted)
            if not torch.equal(edited_edge, edge) and size != len(locators):
                raise ValueError("Partial deletion unexpectedly changed the graph")
            if size == len(locators) and edge.shape[1] - edited_edge.shape[1] != 2:
                raise ValueError("Full deletion did not remove exactly one pair")
            with torch.no_grad():
                logit = float(model(edited_x, mean_adjacency(
                    edited_edge, len(ids)))[target].item())
            probability = float(1 / (1 + np.exp(-logit)))
            rows.append({
                "removed_locator_indices": list(subset),
                "removed_tweet_count": size,
                "remaining_tweet_count": len(surviving),
                "changed_feature_columns": changed,
                "graph_pair_present": size < len(locators),
                "removed_event_records": removed_events,
                "target_logit": logit,
                "target_probability": probability,
                "target_class": int(probability >= threshold),
            })
        print(f"evaluated subsets with {size} deleted tweets", flush=True)
    baseline = rows[0]
    complete = rows[-1]
    if (len(rows) != 128
            or abs(baseline["target_logit"] - pointer["frozen_target_logit_by_graph_state"]["pair_present"]) > 1e-6
            or abs(complete["target_logit"] - target_full["tweet_deletion_target_logit"]) > 1e-6
            or baseline["target_class"] != 0
            or complete["target_class"] != 1):
        raise ValueError("Subset endpoints do not replay independent certificates")
    flip = [r for r in rows if r["target_class"] != baseline["target_class"]]
    minimum = min(r["removed_tweet_count"] for r in flip)
    partial = rows[:-1]
    report = {
        "scope": "all subsets of the seven source tweets supporting the sole native graph pair of Fox8 test account 415566147",
        "source_operation": "delete each chosen full tweet; keep account profile; recompute all 31 source features, fixed training scaler, all native graph events and frozen GNN",
        "not_an_observed_source_correction": True,
        "official_paper_hypbot_graph_configuration": False,
        "account_id": ACCOUNT,
        "raw_source_sha256": file_sha256(RAW),
        "source_index_sha256": file_sha256(EVENT_INDEX),
        "feature_csv_sha256": file_sha256(DATA / "features.csv"),
        "model_sha256": file_sha256(checkpoint_path),
        "graph_sha256": graph_meta["graph_sha256"],
        "pointer_subset_certificate_sha256": file_sha256(POINTER_CERT),
        "full_tweet_certificate_sha256": file_sha256(FULL_TWEET_CERT),
        "raw_tweet_locators": locators,
        "original_account_tweet_count": len(tweets),
        "frozen_threshold": threshold,
        "tested_subsets": len(rows),
        "minimum_deleted_full_tweet_count_for_class_flip": minimum,
        "partial_deletion_logit_min": min(r["target_logit"] for r in partial),
        "partial_deletion_logit_max": max(r["target_logit"] for r in partial),
        "partial_deletion_probability_max": max(r["target_probability"] for r in partial),
        "partial_deletion_changed_feature_columns": sorted({
            name for r in partial for name in r["changed_feature_columns"]}),
        "class_flip_subset_counts_by_deleted_tweet_count": dict(sorted(Counter(
            r["removed_tweet_count"] for r in flip).items())),
        "all_subsets": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("all_subsets", "raw_tweet_locators")},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
