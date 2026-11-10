"""Trace raw Fox8 interaction records through a frozen native-graph HypBot variant."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/scripts"))
sys.path.insert(0, str(ROOT / "source"))
from export_hypbot_native_graph_paired import native_symmetric_graph
from train_v2_detector import file_sha256, load_data
from verifiable_bot.fox8_adapter import tweet_features


RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
EVENT_INDEX = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
RAW_AUDIT = ROOT / "research/certificates/fox8_native_interaction_full_source_audit.json"
RUN = ROOT / "output/hypbot_fox8_native_graph_paired/seed0"
OUT = ROOT / "research/certificates/hypbot_fox8_native_graph_raw_edge_response.json"
REPO = Path("D:/Code/SAHG/SAHG/HypBot")
DATA = Path("D:/Code/SAHG/SAHG/code/data/fox8_23")
SELECTION_SEED = "fox8-native-hypbot-raw-edge-2026-11-04"
SELECTED_ACCOUNT_COUNT = 12


def rank(value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}|{value}".encode()).hexdigest()


def verify_raw_locators(events: list[dict]) -> None:
    by_line = defaultdict(list)
    for event in events:
        locator = event["locators"][0]
        by_line[int(locator["line"])].append(event)
    found = set()
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number not in by_line:
                continue
            row = json.loads(line)
            for event in by_line[line_number]:
                locator = event["locators"][0]
                tweet = row["user_tweets"][int(locator["tweet_index"])]
                tweet_id = str(tweet.get("id_str") or tweet.get("id")
                               or locator["tweet_index"])
                if event["relation"] == "reply":
                    target = tweet.get("in_reply_to_user_id_str") or \
                             tweet.get("in_reply_to_user_id")
                else:
                    target = (tweet.get("retweeted_status") or {}).get("user") or {}
                    target = target.get("id_str") or target.get("id")
                if (str(row["user_id"]) != event["source_id"]
                        or str(target) != event["target_id"]
                        or tweet_id != str(locator["tweet_id"])):
                    raise ValueError(f"Raw interaction locator differs: {event}")
                modified = dict(tweet)
                if event["relation"] == "reply":
                    modified["in_reply_to_user_id_str"] = None
                    modified["in_reply_to_user_id"] = None
                else:
                    retweeted = dict(modified["retweeted_status"])
                    user = dict(retweeted["user"])
                    user["id_str"] = None
                    user["id"] = None
                    retweeted["user"] = user
                    modified["retweeted_status"] = retweeted
                modified_tweets = list(row["user_tweets"])
                modified_tweets[int(locator["tweet_index"])] = modified
                if tweet_features(row["user_tweets"]) != tweet_features(modified_tweets):
                    raise ValueError("Raw target-pointer edit changes HypBot node features")
                found.add((event["source_id"], event["target_id"], event["relation"]))
    if len(found) != len(events):
        raise ValueError("Not all selected raw interaction locators were found")


def main() -> None:
    sys.path.insert(0, str(REPO.resolve()))
    from hypbot.data import mean_adjacency
    from hypbot.model import HypBot
    from train import PRESETS

    torch.set_num_threads(4)
    frozen = json.loads((RUN / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = frozen["manifest"]
    checkpoint_path = RUN / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    args = argparse.Namespace(dataset="fox8", data_dir=DATA, account_file=None,
                              raw_source=[RAW], knn_k=10)
    x, _, split, ids, _, _, hashes, _ = load_data(args, 0)
    edge, graph_meta = native_symmetric_graph(ROOT, "fox8", ids)
    event_index = json.loads(EVENT_INDEX.read_text(encoding="utf-8"))
    raw_audit = json.loads(RAW_AUDIT.read_text(encoding="utf-8"))
    if (file_sha256(checkpoint_path) != manifest["model_sha256"]
            or hashes["feature_sha256"] != manifest["node_feature_sha256"]
            or hashes["split_sha256"] != manifest["split_sha256"]
            or checkpoint["split_sha256"] != hashes["split_sha256"]
            or graph_meta != manifest["native_graph"]
            or file_sha256(RAW) != event_index["raw_source_sha256"]
            or file_sha256(EVENT_INDEX) != graph_meta["source_index_sha256"]
            or raw_audit["status"] != "verified_all_native_interaction_fact_cells_and_event_locators"
            or raw_audit["event_index_sha256"] != file_sha256(EVENT_INDEX)
            or raw_audit["raw_source_sha256"] != file_sha256(RAW)):
        raise ValueError("Frozen GNN, node features or native raw graph differs")
    config = PRESETS["fox8"]
    model = HypBot(x.shape[1], K=manifest["num_sectors"], gated=False,
                   **config["model"], graph_width=x.shape[1], sparse_adj=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    adjacency = mean_adjacency(edge, len(ids))
    with torch.no_grad():
        baseline_logit = model(x, adjacency).cpu().numpy()
        rebuilt_baseline_logit = model(x, mean_adjacency(edge.clone(), len(ids))).cpu().numpy()
        baseline_graph_embedding = model.aggregate(x, adjacency)
        baseline_graph_radius_std = float(model.enc_graph(
            baseline_graph_embedding)[0].std())
        node_readout, _ = model._readout(
            model.enc_node, model.warp_node, model.sectors_node, x)
        def fixed_graph_scale_logits(graph_embedding):
            radius, unit = model.enc_graph(graph_embedding)
            gamma = model.warp_graph(unit)
            entropy, alignment, _ = model.sectors_graph(unit, gamma)
            normalized_radius = radius / (baseline_graph_radius_std + 1e-6)
            normalized_entropy = entropy / (math.log(model.K) + 1e-6)
            graph_readout = torch.stack([
                normalized_radius, normalized_entropy, alignment,
                normalized_radius * alignment, gamma * alignment], dim=-1)
            return model.head(torch.cat((node_readout, graph_readout), dim=-1)
                              ).squeeze(-1).cpu().numpy()
        baseline_fixed_scale_logit = fixed_graph_scale_logits(
            baseline_graph_embedding)
    fixed_scale_baseline_error = float(np.max(np.abs(
        baseline_fixed_scale_logit - baseline_logit)))
    if fixed_scale_baseline_error > 1e-6:
        raise ValueError("Fixed-radius-scale readout does not replay original GNN")
    no_edit_rebuild_error = float(np.max(np.abs(
        baseline_logit - rebuilt_baseline_logit)))
    if no_edit_rebuild_error > 1e-7:
        raise ValueError("No-edit graph reconstruction changed frozen GNN output")
    baseline_probability = 1 / (1 + np.exp(-baseline_logit))
    test = np.asarray(split["test"], dtype=int)
    test_ids = [ids[int(i)] for i in test]
    if test_ids != [row["account_id"] for row in frozen["predictions"]]:
        raise ValueError("Saved test account order differs")
    replay_error = float(np.max(np.abs(
        baseline_probability[test] -
        [row["probability"] for row in frozen["predictions"]])))
    if replay_error > 1e-6:
        raise ValueError("Frozen native-graph HypBot prediction does not replay")
    index = {uid: i for i, uid in enumerate(ids)}
    pair_events = defaultdict(list)
    for event in event_index["events"]:
        source, target = event["source_id"], event["target_id"]
        if source in index and target in index and source != target:
            pair_events[tuple(sorted((source, target)))].append(event)
    candidates = {pair: values[0] for pair, values in pair_events.items()
                  if len(values) == 1 and values[0]["count"] == 1
                  and len(values[0]["locators"]) == 1}
    by_test_account = defaultdict(list)
    test_set = set(test_ids)
    for pair in candidates:
        for uid in pair:
            if uid in test_set:
                by_test_account[uid].append(pair)
    selected = []
    used = set()
    for uid in sorted(by_test_account, key=rank):
        choices = sorted(by_test_account[uid], key=lambda pair: rank("|".join(pair)))
        pair = next((p for p in choices if p not in used), None)
        if pair is None:
            continue
        selected.append((uid, pair, candidates[pair]))
        used.add(pair)
        if len(selected) == SELECTED_ACCOUNT_COUNT:
            break
    if len(selected) != SELECTED_ACCOUNT_COUNT:
        raise ValueError("Insufficient singly sourced test-incident raw edges")
    verify_raw_locators([event for _, _, event in selected])
    original_edge_set = set(map(tuple, edge.T.numpy().tolist()))
    neighbors = defaultdict(set)
    for source, target in original_edge_set:
        neighbors[source].add(target)
    rows = []
    threshold = float(manifest["threshold"])
    for uid, pair, event in selected:
        u, v = index[pair[0]], index[pair[1]]
        if (u, v) not in original_edge_set or (v, u) not in original_edge_set:
            raise ValueError("Selected source edge missing from frozen graph")
        keep = ~(((edge[0] == u) & (edge[1] == v)) |
                 ((edge[0] == v) & (edge[1] == u)))
        edited_edge = edge[:, keep]
        if edited_edge.shape[1] != edge.shape[1] - 2:
            raise ValueError("Unique source edit did not remove both symmetric arcs")
        edited_adjacency = mean_adjacency(edited_edge, len(ids))
        with torch.no_grad():
            edited_logit = model(x, edited_adjacency).cpu().numpy()
            edited_graph_embedding = model.aggregate(x, edited_adjacency)
            edited_graph_radius_std = float(model.enc_graph(
                edited_graph_embedding)[0].std())
            edited_fixed_scale_logit = fixed_graph_scale_logits(
                edited_graph_embedding)
        edited_probability = 1 / (1 + np.exp(-edited_logit))
        within_two = {u, v}
        frontier = {u, v}
        for _ in range(2):
            frontier = set().union(*(neighbors[node] for node in frontier)) - within_two
            within_two.update(frontier)
        outside_two = np.asarray([j for j in range(len(ids)) if j not in within_two],
                                 dtype=np.int64)
        nonlocal_delta = (edited_logit[outside_two] - baseline_logit[outside_two])
        nonlocal_embedding_error = float(torch.max(torch.abs(
            edited_graph_embedding[outside_two] -
            baseline_graph_embedding[outside_two])).item()) if len(outside_two) else 0.0
        nonlocal_fixed_scale_logit_error = float(np.max(np.abs(
            edited_fixed_scale_logit[outside_two] -
            baseline_fixed_scale_logit[outside_two]))) if len(outside_two) else 0.0
        if nonlocal_embedding_error > 1e-7:
            raise ValueError("Two-layer GNN graph embedding changed beyond two hops")
        if nonlocal_fixed_scale_logit_error > 1e-6:
            raise ValueError("Frozen population-scale control changed distant decisions")
        target = index[uid]
        locator = event["locators"][0]
        rows.append({
            "account_id": uid,
            "source_pair": list(pair),
            "source_relation": event["relation"],
            "raw_source_edit": "erase only the target user ID pointer at this tweet locator",
            "node_features_unchanged_by_source_field_edit": True,
            "source_locator": {
                "file_sha256": event_index["raw_source_sha256"],
                "line": locator["line"], "tweet_index": locator["tweet_index"],
                "tweet_id": locator["tweet_id"], "field": locator["field"]},
            "native_graph_arcs_before": edge.shape[1],
            "native_graph_arcs_after": edited_edge.shape[1],
            "original_target_logit": float(baseline_logit[target]),
            "edited_target_logit": float(edited_logit[target]),
            "target_logit_delta": float(edited_logit[target] - baseline_logit[target]),
            "original_target_probability": float(baseline_probability[target]),
            "edited_target_probability": float(edited_probability[target]),
            "original_target_class": int(baseline_probability[target] >= threshold),
            "edited_target_class": int(edited_probability[target] >= threshold),
            "test_accounts_with_abs_logit_change_gt_1e_6": int(np.count_nonzero(
                np.abs(edited_logit[test] - baseline_logit[test]) > 1e-6)),
            "all_accounts_with_abs_logit_change_gt_1e_6": int(np.count_nonzero(
                np.abs(edited_logit - baseline_logit) > 1e-6)),
            "accounts_within_two_hops_of_edited_pair": len(within_two),
            "accounts_outside_two_hops_with_abs_logit_change_gt_1e_6": int(
                np.count_nonzero(np.abs(nonlocal_delta) > 1e-6)),
            "max_abs_logit_change_outside_two_hops": float(np.max(
                np.abs(nonlocal_delta))) if len(nonlocal_delta) else 0.0,
            "max_graph_embedding_change_outside_two_hops": nonlocal_embedding_error,
            "max_logit_change_outside_two_hops_with_original_radius_scale": (
                nonlocal_fixed_scale_logit_error),
            "graph_radius_std_before": baseline_graph_radius_std,
            "graph_radius_std_after": edited_graph_radius_std,
            "max_abs_test_logit_change": float(np.max(np.abs(
                edited_logit[test] - baseline_logit[test]))),
        })
    report = {
        "scope": "twelve prediction-blind raw interaction deletions on frozen Fox8 native-graph HypBot variant",
        "official_paper_hypbot_graph_configuration": False,
        "source_index_sha256": file_sha256(EVENT_INDEX),
        "full_raw_interaction_audit_sha256": file_sha256(RAW_AUDIT),
        "raw_source_sha256": file_sha256(RAW),
        "model_sha256": file_sha256(checkpoint_path),
        "saved_test_predictions_sha256": file_sha256(RUN / "test_predictions.json"),
        "node_feature_sha256": hashes["feature_sha256"],
        "graph_sha256": graph_meta["graph_sha256"],
        "test_accounts": len(test),
        "eligible_unique_event_pairs": len(candidates),
        "test_accounts_with_eligible_incident_edge": len(by_test_account),
        "selection_seed": SELECTION_SEED,
        "selected_raw_edge_operations": len(rows),
        "max_saved_prediction_replay_difference": replay_error,
        "max_no_edit_adjacency_rebuild_logit_difference": no_edit_rebuild_error,
        "max_fixed_radius_scale_original_replay_difference": fixed_scale_baseline_error,
        "source_locators_verified_against_raw_gzip": len(rows),
        "source_field_edits_with_unchanged_node_features": len(rows),
        "target_class_changes": sum(r["original_target_class"] != r["edited_target_class"]
                                    for r in rows),
        "target_logit_changes_gt_1e_6": sum(abs(r["target_logit_delta"]) > 1e-6
                                           for r in rows),
        "source_edits_affecting_accounts_outside_two_hops_gt_1e_6": sum(
            r["accounts_outside_two_hops_with_abs_logit_change_gt_1e_6"] > 0
            for r in rows),
        "max_graph_embedding_change_outside_two_hops": max(
            r["max_graph_embedding_change_outside_two_hops"] for r in rows),
        "max_distant_logit_change_with_original_radius_scale": max(
            r["max_logit_change_outside_two_hops_with_original_radius_scale"]
            for r in rows),
        "rows": rows,
        "interpretation": "The model is a native-edge graph variant of HypBot, not the paper's kNN configuration. Each selected raw tweet target-user-ID pointer uniquely supplies one undirected relation pair. Clearing that pointer leaves the recorded node features unchanged, removes both normalized adjacency arcs, and recomputes predictions with frozen weights. This is inference-time missing-observation sensitivity on a transductive graph, not retraining, social causality or semantic truth. Distant graph embeddings stay fixed while logits change; holding the graph-radius population standard deviation at its original value removes all distant logit changes in these twelve cases.",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
