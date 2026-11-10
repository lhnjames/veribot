"""Trace one complete raw relation-pair edit through two native-graph GNNs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import math
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
from export_hypbot_native_graph_paired import native_symmetric_graph
from train_v2_detector import file_sha256, hash_bytes, load_data
from verifiable_bot.evidence_v2 import AtomicClaim, SourceRecord
from verifiable_bot.fox8_adapter import FEATURE_FIELDS, tweet_features
from verifiable_bot.fox8_relational_graph import typed_native_edges
from verifiable_bot.graph_evidence_architecture import (
    GatedRelNetForwardAdapter, GraphEvidenceInput, HypBotForwardAdapter,
    certify_source_response,
)

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
HYPBOT_RUN = ROOT / "output/hypbot_fox8_native_graph_paired/seed0"
REL_RUN = ROOT / "output/fox8_rel_evidence_gnn/seed0"
CASE = ROOT / "research/certificates/hypbot_fox8_seven_pointer_minimality.json"


def pointer_records(events: list[dict]) -> tuple[dict[str, SourceRecord], dict[tuple[str, str, str], tuple[str, ...]]]:
    sources = {}
    event_ids = {}
    for event in events:
        key = (str(event["source_id"]), str(event["target_id"]), event["relation"])
        ids = []
        for locator in event["locators"]:
            sid = (f"fox8:{key[2]}:{key[0]}:{key[1]}:"
                   f"{locator['line']}:{locator['tweet_index']}")
            location = (f"{RAW.resolve()}#line={locator['line']};"
                        f"tweet={locator['tweet_index']};{locator['field']}")
            sources[sid] = SourceRecord(
                sid, key[0], f"{key[2]}_target_pointer",
                {"target_user_id": key[1], "tweet_id": str(locator["tweet_id"])},
                location)
            ids.append(sid)
        if len(ids) != event["count"] or len(set(ids)) != len(ids):
            raise ValueError("event pointer multiplicity differs from locators")
        event_ids[key] = tuple(ids)
    if len(sources) != sum(map(len, event_ids.values())):
        raise ValueError("raw pointer IDs collide")
    return sources, event_ids


def readback_selected_pointers(sources: dict[str, SourceRecord]) -> None:
    by_line = defaultdict(list)
    for record in sources.values():
        text = record.raw_location.split("#line=", 1)[1]
        line, suffix = text.split(";tweet=", 1)
        tweet, field = suffix.split(";", 1)
        by_line[int(line)].append((record, int(tweet), field))
    seen = set()
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number not in by_line:
                continue
            row = json.loads(line)
            modified_tweets = list(row["user_tweets"])
            for source, ordinal, field in by_line[line_number]:
                tweet = row["user_tweets"][ordinal]
                if source.kind == "reply_target_pointer":
                    target = tweet.get("in_reply_to_user_id_str") or \
                             tweet.get("in_reply_to_user_id")
                    modified = dict(modified_tweets[ordinal])
                    modified["in_reply_to_user_id_str"] = None
                    modified["in_reply_to_user_id"] = None
                else:
                    user = (tweet.get("retweeted_status") or {}).get("user") or {}
                    target = user.get("id_str") or user.get("id")
                    modified = dict(modified_tweets[ordinal])
                    retweeted = dict(modified["retweeted_status"])
                    changed_user = dict(retweeted["user"])
                    changed_user["id_str"] = None
                    changed_user["id"] = None
                    retweeted["user"] = changed_user
                    modified["retweeted_status"] = retweeted
                modified_tweets[ordinal] = modified
                tweet_id = str(tweet.get("id_str") or tweet.get("id") or ordinal)
                if (str(row["user_id"]) != source.account_id
                        or str(target) != source.payload["target_user_id"]
                        or tweet_id != source.payload["tweet_id"]
                        or field not in ("in_reply_to_user_id_str|in_reply_to_user_id",
                                         "retweeted_status.user.id_str|id")):
                    raise ValueError(f"raw pointer differs: {source.source_id}")
                seen.add(source.source_id)
            if tweet_features(row["user_tweets"]) != tweet_features(modified_tweets):
                raise ValueError("target-user pointer edit changed named tweet facts")
    if seen != set(sources):
        raise ValueError("not every selected pointer was read from raw file")


def symmetric_from_typed(typed: dict[str, torch.Tensor]) -> torch.Tensor:
    directed = {tuple(map(int, pair)) for edges in typed.values()
                for pair in edges.T.tolist()}
    if not directed:
        raise ValueError("typed graph has no edges")
    array = np.asarray(sorted(directed), dtype=np.int64).T.copy()
    return torch.from_numpy(array)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-account", type=str)
    parser.add_argument("--candidate-neighbor", type=str)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/certificates/fox8_two_native_gnn_relation_evidence.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    loader = argparse.Namespace(dataset="fox8",
                                data_dir=Path("D:/Code/SAHG/SAHG/code/data/fox8_23"),
                                account_file=None, raw_source=[RAW], knn_k=10)
    x, _, split, ids, _, _, hashes, _ = load_data(loader, 0)
    prior = json.loads(CASE.read_text(encoding="utf-8"))
    target = args.target_account or prior["account_id"]
    neighbor = args.candidate_neighbor or prior["neighbor_id"]
    if bool(args.target_account) != bool(args.candidate_neighbor):
        raise ValueError("target and candidate neighbor must be supplied together")
    if target not in {ids[int(i)] for i in split["test"]}:
        raise ValueError("selected target is outside frozen test partition")
    payload = json.loads(EVENTS.read_text(encoding="utf-8"))
    if payload["raw_source_sha256"] != file_sha256(RAW):
        raise ValueError("raw source differs from event index")
    pair_events = defaultdict(list)
    for event in payload["events"]:
        if event["source_id"] in ids and event["target_id"] in ids:
            pair_events[tuple(sorted((event["source_id"], event["target_id"])))].append(event)
    candidate_pair = tuple(sorted((target, neighbor)))
    candidate_events = pair_events[candidate_pair]
    pointer_budget = sum(event["count"] for event in candidate_events)
    if pointer_budget < 1:
        raise ValueError("selected source pair has no raw pointers")
    # Prefer a different target-incident pair at the same raw-pointer budget.
    # Selection uses source structure only, before either model is queried.
    eligible = [pair for pair in sorted(pair_events)
                if pair != candidate_pair and
                sum(event["count"] for event in pair_events[pair]) == pointer_budget]
    control_pair = next((pair for pair in eligible if target in pair), None)
    if control_pair is None:
        control_pair = next((pair for pair in eligible if target not in pair), None)
    if control_pair is None:
        raise ValueError("no equal-budget raw relation control exists")
    control_scope = "target_incident" if target in control_pair else "nonincident"
    control_events = pair_events[control_pair]
    sources, event_ids = pointer_records(candidate_events + control_events)
    readback_selected_pointers(sources)
    candidate_ids = tuple(sid for event in candidate_events
                          for sid in event_ids[(event["source_id"], event["target_id"],
                                                event["relation"])])
    control_ids = tuple(sid for event in control_events
                        for sid in event_ids[(event["source_id"], event["target_id"],
                                              event["relation"])])
    if len(candidate_ids) != len(control_ids) or len(candidate_ids) != pointer_budget:
        raise ValueError("candidate and control budgets differ")
    claim = AtomicClaim(candidate_ids, "count", pointer_budget)
    original_edge, native_meta = native_symmetric_graph(ROOT, "fox8", ids)
    original_typed, typed_hash = typed_native_edges(EVENTS, ids)
    if not torch.equal(symmetric_from_typed(original_typed), original_edge):
        raise ValueError("typed graph does not reconstruct native HypBot graph")
    hypbot_path = HYPBOT_RUN / "model.pt"
    hypbot_saved = json.loads((HYPBOT_RUN / "test_predictions.json").read_text())
    hypbot_meta = hypbot_saved["manifest"]
    if (hashes["feature_sha256"] != hypbot_meta["node_feature_sha256"]
            or hashes["split_sha256"] != hypbot_meta["split_sha256"]
            or native_meta != hypbot_meta["native_graph"]
            or file_sha256(hypbot_path) != hypbot_meta["model_sha256"]):
        raise ValueError("HypBot frozen contract differs")
    hypbot_ckpt = torch.load(hypbot_path, map_location="cpu", weights_only=True)
    hypbot = HypBot(x.shape[1], K=hypbot_meta["num_sectors"], gated=False,
                    **PRESETS["fox8"]["model"], graph_width=x.shape[1],
                    sparse_adj=True)
    hypbot.load_state_dict(hypbot_ckpt["model_state"])
    hypbot.eval()
    rel_path = REL_RUN / "model.pt"
    rel_saved = json.loads((REL_RUN / "test_predictions.json").read_text())
    rel_ckpt = torch.load(rel_path, map_location="cpu", weights_only=False)
    if (rel_ckpt["feature_sha256"] != hashes["feature_sha256"]
            or rel_ckpt["split_sha256"] != hashes["split_sha256"]
            or rel_ckpt["native_graph_sha256"] != typed_hash
            or rel_ckpt["native_event_index_sha256"] != file_sha256(EVENTS)
            or tuple(rel_ckpt["feature_names"]) != tuple(FEATURE_FIELDS)):
        raise ValueError("relation GNN frozen contract differs")
    rel = GatedRelNet(rel_ckpt["input_dim"],
                      hidden_channels=rel_ckpt["hidden_channels"],
                      relation_names=tuple(rel_ckpt["relation_names"]),
                      num_layers=rel_ckpt["num_layers"], dropout=rel_ckpt["dropout"])
    rel.load_state_dict(rel_ckpt["model_state"])
    rel.eval()

    def active_graph(remaining):
        excluded = frozenset(key for key, pointer_ids in event_ids.items()
                              if all(sid not in remaining for sid in pointer_ids))
        return typed_native_edges(EVENTS, ids, excluded)

    def build_hypbot(remaining):
        typed, _ = active_graph(remaining)
        edge = symmetric_from_typed(typed)
        return GraphEvidenceInput(x, tuple(FEATURE_FIELDS),
                                  mean_adjacency(edge, len(ids)),
                                  hash_bytes(edge.numpy().tobytes()), tuple(ids))

    def build_rel(remaining):
        typed, graph_hash = active_graph(remaining)
        return GraphEvidenceInput(x, tuple(FEATURE_FIELDS), typed,
                                  graph_hash, tuple(ids))

    if build_hypbot(sources).graph_sha256 != native_meta["graph_sha256"] or \
            build_rel(sources).graph_sha256 != typed_hash:
        raise ValueError("no-edit raw pointer graph does not replay training graph")
    reports = [
        certify_source_response(
            account_id=target, claim=claim, sources=sources,
            build_graph=builder, adapter=adapter,
            model_sha256=file_sha256(path),
            control_source_groups=(control_ids,))
        for builder, adapter, path in (
            (build_hypbot, HypBotForwardAdapter(hypbot), hypbot_path),
            (build_rel, GatedRelNetForwardAdapter(rel), rel_path),
        )
    ]
    for report, saved, threshold in zip(
            reports, (hypbot_saved, rel_saved),
            (float(hypbot_meta["threshold"]), 0.5)):
        row = next(item for item in saved["predictions"] if item["account_id"] == target)
        p_before = 1 / (1 + math.exp(-report["before"]["logit"]))
        p_after = 1 / (1 + math.exp(-report["after"]["logit"]))
        if abs(p_before - row["probability"]) > 1e-6:
            raise ValueError("baseline differs from saved frozen prediction")
        report.update({"decision_threshold": threshold,
                       "probability_before": p_before,
                       "probability_after": p_after,
                       "class_before": int(p_before >= threshold),
                       "class_after": int(p_after >= threshold),
                       "saved_prediction_replay_error": abs(p_before-row["probability"])})
    result = {
        "status": "two_native_graph_gnns_traced_from_same_raw_relation_pair",
        "scope": "one selected target relation pair and one source-selected equal-budget control",
        "source_operation": "clear all target-user relation pointers on one pair; retain tweets and node facts",
        "target_account_id": target,
        "candidate_pair": list(candidate_pair),
        "control_pair": list(control_pair),
        "control_scope": control_scope,
        "raw_source_sha256": file_sha256(RAW),
        "event_index_sha256": file_sha256(EVENTS),
        "candidate_pointer_count": len(candidate_ids),
        "control_pointer_count": len(control_ids),
        "source_records": {sid: {"sha256": record.sha256,
                                 "account_id": record.account_id,
                                 "payload": record.payload,
                                 "raw_location": record.raw_location}
                           for sid, record in sources.items()},
        "model_reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": result["status"], "target": target,
                      "candidate_pair": candidate_pair, "control_pair": control_pair,
                      "models": [{"name": r["model_adapter"],
                                  "before": r["class_before"],
                                  "after": r["class_after"],
                                  "delta_logit": r["delta_logit"],
                                  "control_delta_logit": r["controls"][0]["delta_logit"]}
                                 for r in reports]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
