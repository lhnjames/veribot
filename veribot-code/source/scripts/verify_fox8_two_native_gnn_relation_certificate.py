"""Independently replay raw pointers, rebuilt graphs, and two GNN logits."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/scripts"))
sys.path.insert(0, str(ROOT / "source/hgt_import/code"))
sys.path.insert(0, str(Path("D:/Code/SAHG/SAHG/HypBot")))
from relnet_reimplementation import GatedRelNet
from hypbot.data import mean_adjacency
from hypbot.model import HypBot
from train import PRESETS
from train_v2_detector import file_sha256, hash_bytes, load_data
from verifiable_bot.evidence_v2 import SourceRecord
from verifiable_bot.fox8_adapter import tweet_features
from verifiable_bot.fox8_relational_graph import typed_native_edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", nargs="?", type=Path,
                        default=ROOT / "research/certificates/fox8_two_native_gnn_relation_evidence.json")
    args = parser.parse_args()
    cert = json.loads(args.certificate.read_text(encoding="utf-8"))
    reports = cert["model_reports"]
    if {r["model_adapter"] for r in reports} != {
            "HypBot-native-graph", "GatedRelNet-native-relations"} or len(reports) != 2:
        raise ValueError("certificate model set differs")
    raw = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    index_path = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
    if file_sha256(raw) != cert["raw_source_sha256"] or \
            file_sha256(index_path) != cert["event_index_sha256"]:
        raise ValueError("raw record or event index bytes differ")
    records = cert["source_records"]
    by_line = defaultdict(list)
    for sid, record in records.items():
        if not record["raw_location"].startswith(str(raw.resolve()) + "#line="):
            raise ValueError("source locator points to another file")
        locator = record["raw_location"].split("#line=", 1)[1]
        line, tail = locator.split(";tweet=", 1)
        ordinal, field = tail.split(";", 1)
        by_line[int(line)].append((sid, record, int(ordinal), field))
    seen = set()
    with gzip.open(raw, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number not in by_line:
                continue
            row = json.loads(line)
            modified_tweets = list(row["user_tweets"])
            for sid, record, ordinal, field in by_line.get(line_number, ()):
                tweet = row["user_tweets"][ordinal]
                if field == "in_reply_to_user_id_str|in_reply_to_user_id":
                    target = tweet.get("in_reply_to_user_id_str") or \
                             tweet.get("in_reply_to_user_id")
                    kind = "reply_target_pointer"
                    modified = dict(modified_tweets[ordinal])
                    modified["in_reply_to_user_id_str"] = None
                    modified["in_reply_to_user_id"] = None
                elif field == "retweeted_status.user.id_str|id":
                    user = (tweet.get("retweeted_status") or {}).get("user") or {}
                    target = user.get("id_str") or user.get("id")
                    kind = "retweet_target_pointer"
                    modified = dict(modified_tweets[ordinal])
                    retweeted = dict(modified["retweeted_status"])
                    changed_user = dict(retweeted["user"])
                    changed_user["id_str"] = None
                    changed_user["id"] = None
                    retweeted["user"] = changed_user
                    modified["retweeted_status"] = retweeted
                else:
                    raise ValueError("unknown raw relation pointer field")
                modified_tweets[ordinal] = modified
                tweet_id = str(tweet.get("id_str") or tweet.get("id") or ordinal)
                expected = SourceRecord(sid, str(row["user_id"]), kind,
                                        {"target_user_id": str(target),
                                         "tweet_id": tweet_id},
                                        record["raw_location"])
                if (expected.sha256 != record["sha256"]
                        or expected.payload != record["payload"]
                        or expected.account_id != record["account_id"]):
                    raise ValueError(f"raw pointer differs: {sid}")
                seen.add(sid)
            if tweet_features(row["user_tweets"]) != tweet_features(modified_tweets):
                raise ValueError("raw pointer edit changed named tweet facts")
    if seen != set(records):
        raise ValueError("not every raw pointer was independently located")
    candidate_ids = tuple(reports[0]["source_ids"])
    control_ids = tuple(reports[0]["controls"][0]["source_ids"])
    if (len(candidate_ids) != len(control_ids)
            or len(candidate_ids) != cert["candidate_pointer_count"]
            or len(set(candidate_ids) | set(control_ids)) != len(records)):
        raise ValueError("candidate/control pointer budget differs")
    if any(tuple(report["source_ids"]) != candidate_ids or
           tuple(report["controls"][0]["source_ids"]) != control_ids
           for report in reports):
        raise ValueError("models used different source operations")
    events = json.loads(index_path.read_text(encoding="utf-8"))["events"]
    event_keys = {}
    for event in events:
        pair = tuple(sorted((event["source_id"], event["target_id"])))
        if pair not in (tuple(cert["candidate_pair"]), tuple(cert["control_pair"])):
            continue
        key = (event["source_id"], event["target_id"], event["relation"])
        ids = {f"fox8:{key[2]}:{key[0]}:{key[1]}:{locator['line']}:{locator['tweet_index']}"
               for locator in event["locators"]}
        event_keys[key] = ids
    if ({sid for key, ids in event_keys.items()
         if tuple(sorted(key[:2])) == tuple(cert["candidate_pair"]) for sid in ids}
            != set(candidate_ids) or
            {sid for key, ids in event_keys.items()
             if tuple(sorted(key[:2])) == tuple(cert["control_pair"]) for sid in ids}
            != set(control_ids)):
        raise ValueError("candidate/control records do not exhaust native graph pairs")
    torch.set_num_threads(4)
    loader = argparse.Namespace(dataset="fox8",
                                data_dir=Path("D:/Code/SAHG/SAHG/code/data/fox8_23"),
                                account_file=None, raw_source=[raw], knn_k=10)
    x, _, _, account_ids, _, _, _, _ = load_data(loader, 0)
    target_index = account_ids.index(cert["target_account_id"])
    hypbot_path = ROOT / "output/hypbot_fox8_native_graph_paired/seed0/model.pt"
    hypbot_saved = json.loads((hypbot_path.parent / "test_predictions.json").read_text())
    hypbot_ckpt = torch.load(hypbot_path, map_location="cpu", weights_only=True)
    hypbot = HypBot(x.shape[1], K=hypbot_saved["manifest"]["num_sectors"],
                    gated=False, **PRESETS["fox8"]["model"],
                    graph_width=x.shape[1], sparse_adj=True)
    hypbot.load_state_dict(hypbot_ckpt["model_state"])
    hypbot.eval()
    rel_path = ROOT / "output/fox8_rel_evidence_gnn/seed0/model.pt"
    rel_ckpt = torch.load(rel_path, map_location="cpu", weights_only=False)
    rel = GatedRelNet(rel_ckpt["input_dim"],
                      hidden_channels=rel_ckpt["hidden_channels"],
                      relation_names=tuple(rel_ckpt["relation_names"]),
                      num_layers=rel_ckpt["num_layers"],
                      dropout=rel_ckpt["dropout"])
    rel.load_state_dict(rel_ckpt["model_state"])
    rel.eval()
    report_by_name = {r["model_adapter"]: r for r in reports}
    models = (("HypBot-native-graph", hypbot, hypbot_path),
              ("GatedRelNet-native-relations", rel, rel_path))
    max_error = 0.0
    for operation, removed in (("before", frozenset()),
                               ("after", frozenset(candidate_ids)),
                               ("control", frozenset(control_ids))):
        excluded = frozenset(key for key, ids in event_keys.items() if ids <= removed)
        typed, typed_hash = typed_native_edges(index_path, account_ids, excluded)
        directed = sorted({tuple(pair) for edges in typed.values()
                           for pair in edges.T.tolist()})
        edge = torch.from_numpy(np.asarray(directed, dtype=np.int64).T.copy())
        native_hash = hash_bytes(edge.numpy().tobytes())
        with torch.inference_mode():
            values = {
                "HypBot-native-graph": float(hypbot(x, mean_adjacency(edge, len(account_ids)))[target_index]),
                "GatedRelNet-native-relations": float(rel(x, typed)[target_index]),
            }
        for name, model, path in models:
            report = report_by_name[name]
            if report["model_sha256"] != file_sha256(path):
                raise ValueError("model checkpoint bytes changed")
            expected = (report["controls"][0]["logit"] if operation == "control"
                        else report[operation]["logit"])
            error = abs(values[name] - expected)
            max_error = max(max_error, error)
            if error > 1e-6:
                raise ValueError(f"{name} {operation} logit did not independently replay")
            if operation != "control":
                graph_hash = native_hash if name == "HypBot-native-graph" else typed_hash
                if graph_hash != report[f"graph_sha256_{operation}"]:
                    raise ValueError("rebuilt graph hash differs from certificate")
    print(json.dumps({"status": "raw_pointers_graphs_and_both_frozen_gnns_independently_replayed",
                      "raw_pointer_count": len(records), "models": len(reports),
                      "maximum_logit_error": max_error}))


if __name__ == "__main__":
    main()
