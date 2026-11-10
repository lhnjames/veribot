"""Test frozen Fox8 native-graph HypBot category response to raw relation edits.

For each test account, clear every raw reply/retweet target-user pointer that
supports a native graph pair incident to it. This is a defined source
counterfactual, not an observed correction of erroneous account data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/scripts"))
sys.path.insert(0, str(ROOT / "source"))
from audit_hypbot_fox8_raw_edge_response import (  # noqa: E402
    DATA, EVENT_INDEX, RAW, RAW_AUDIT, REPO, RUN,
)
from export_hypbot_native_graph_paired import native_symmetric_graph  # noqa: E402
from train_v2_detector import file_sha256, load_data  # noqa: E402
from verifiable_bot.fox8_adapter import tweet_features  # noqa: E402

OUT = ROOT / "research/certificates/hypbot_fox8_account_relation_source_response.json"


def verify_all_raw_pointers(events: list[dict], raw_sha256: str) -> dict:
    """Check every event locator and that clearing pointers preserves node facts."""
    if file_sha256(RAW) != raw_sha256:
        raise ValueError("Raw source hash differs")
    by_line = defaultdict(list)
    for event in events:
        for locator in event["locators"]:
            by_line[int(locator["line"])].append((event, locator))
    checked = 0
    changed_rows = 0
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number not in by_line:
                continue
            row = json.loads(line)
            tweets = list(row["user_tweets"])
            edited = list(tweets)
            for event, locator in by_line[line_number]:
                if str(row["user_id"]) != event["source_id"]:
                    raise ValueError("Event source differs from raw account")
                j = int(locator["tweet_index"])
                original = tweets[j]
                tweet_id = str(original.get("id_str") or original.get("id") or j)
                if tweet_id != str(locator["tweet_id"]):
                    raise ValueError("Event tweet ID differs")
                modified = dict(edited[j])
                if event["relation"] == "reply":
                    target = (original.get("in_reply_to_user_id_str") or
                              original.get("in_reply_to_user_id"))
                    modified["in_reply_to_user_id_str"] = None
                    modified["in_reply_to_user_id"] = None
                elif event["relation"] == "retweet":
                    source_user = (original.get("retweeted_status") or {}).get("user") or {}
                    target = source_user.get("id_str") or source_user.get("id")
                    status = dict(modified["retweeted_status"])
                    user = dict(status["user"])
                    user["id_str"] = None
                    user["id"] = None
                    status["user"] = user
                    modified["retweeted_status"] = status
                else:
                    raise ValueError("Unexpected relation")
                if str(target) != event["target_id"]:
                    raise ValueError("Event target differs from raw pointer")
                edited[j] = modified
                checked += 1
            if tweet_features(tweets) != tweet_features(edited):
                raise ValueError("Pointer edits changed the target account's node facts")
            changed_rows += 1
    expected = sum(len(event["locators"]) for event in events)
    if checked != expected:
        raise ValueError(f"Verified {checked} of {expected} raw pointers")
    return {"event_records": len(events), "raw_locators": checked,
            "raw_account_rows_touched": changed_rows,
            "node_fact_rows_unchanged": changed_rows}


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
        raise ValueError("Frozen model or input contract differs")
    cfg = PRESETS["fox8"]
    model = HypBot(x.shape[1], K=manifest["num_sectors"], gated=False,
                   **cfg["model"], graph_width=x.shape[1], sparse_adj=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        baseline = model(x, mean_adjacency(edge, len(ids))).cpu().numpy()
        baseline_graph_embedding = model.aggregate(x, mean_adjacency(edge, len(ids)))
        radius_std = float(model.enc_graph(baseline_graph_embedding)[0].std())
        node_readout, _ = model._readout(model.enc_node, model.warp_node,
                                        model.sectors_node, x)
        def fixed_scale_logits(graph_embedding):
            radius, unit = model.enc_graph(graph_embedding)
            gamma = model.warp_graph(unit)
            entropy, alignment, _ = model.sectors_graph(unit, gamma)
            r_norm = radius / (radius_std + 1e-6)
            h_norm = entropy / (math.log(model.K) + 1e-6)
            graph_readout = torch.stack([
                r_norm, h_norm, alignment, r_norm * alignment,
                gamma * alignment], dim=-1)
            return model.head(torch.cat((node_readout, graph_readout), dim=-1)
                              ).squeeze(-1).cpu().numpy()
        fixed_baseline = fixed_scale_logits(baseline_graph_embedding)
    fixed_baseline_error = float(np.max(np.abs(fixed_baseline - baseline)))
    if fixed_baseline_error > 1e-6:
        raise ValueError("Fixed-scale baseline does not reproduce GNN scores")
    p0 = 1 / (1 + np.exp(-baseline))
    test = np.asarray(split["test"], dtype=np.int64)
    test_ids = [ids[int(i)] for i in test]
    if test_ids != [row["account_id"] for row in frozen["predictions"]]:
        raise ValueError("Saved test account order differs")
    replay_error = float(np.max(np.abs(p0[test] -
                                   [row["probability"] for row in frozen["predictions"]])))
    if replay_error > 1e-6:
        raise ValueError("Frozen GNN does not reproduce saved predictions")
    index = {uid: i for i, uid in enumerate(ids)}
    pair_events = defaultdict(list)
    for event in event_index["events"]:
        source, target = event["source_id"], event["target_id"]
        if source in index and target in index and source != target:
            pair_events[tuple(sorted((source, target)))].append(event)
    by_account = defaultdict(set)
    for pair in pair_events:
        for uid in pair:
            by_account[uid].add(pair)
    graph_pairs = {tuple(sorted((ids[int(u)], ids[int(v)])))
                   for u, v in edge.T.tolist()}
    if graph_pairs != set(pair_events):
        raise ValueError("Source-index pairs differ from model graph")
    # The union is verified directly against original gzip. A particular
    # account edit uses exactly the subset of these pointers incident to it.
    all_events = [event for values in pair_events.values() for event in values]
    raw_verification = verify_all_raw_pointers(all_events,
                                               event_index["raw_source_sha256"])
    threshold = float(manifest["threshold"])
    rows = []
    for position, uid in enumerate(test_ids, 1):
        pairs = by_account.get(uid, set())
        if pairs:
            remove_vertices = {tuple(sorted((index[a], index[b]))) for a, b in pairs}
            keep = torch.as_tensor([
                tuple(sorted((int(u), int(v)))) not in remove_vertices
                for u, v in edge.T.tolist()], dtype=torch.bool)
            edited_edge = edge[:, keep]
            if edge.shape[1] - edited_edge.shape[1] != 2 * len(pairs):
                raise ValueError("Incident-pair deletion count differs")
            with torch.no_grad():
                edited = model(x, mean_adjacency(edited_edge, len(ids))).cpu().numpy()
            edited_logit = float(edited[index[uid]])
            nonlocal_changed = int(np.count_nonzero(np.abs(edited - baseline) > 1e-6))
        else:
            edited_logit = float(baseline[index[uid]])
            nonlocal_changed = 0
        target = index[uid]
        edited_probability = float(1 / (1 + np.exp(-edited_logit)))
        selected_events = [event for pair in pairs for event in pair_events[pair]]
        rows.append({
            "account_id": uid,
            "incident_unique_undirected_pairs": len(pairs),
            "raw_event_records_cleared": len(selected_events),
            "raw_target_user_pointers_cleared": sum(len(e["locators"]) for e in selected_events),
            "original_logit": float(baseline[target]),
            "edited_logit": edited_logit,
            "logit_delta": edited_logit - float(baseline[target]),
            "original_probability": float(p0[target]),
            "edited_probability": edited_probability,
            "original_class": int(p0[target] >= threshold),
            "edited_class": int(edited_probability >= threshold),
            "all_accounts_with_abs_logit_change_gt_1e_6": nonlocal_changed,
        })
        if position % 50 == 0:
            print(f"evaluated {position}/{len(test_ids)} test accounts", flush=True)
    affected = [r for r in rows if r["incident_unique_undirected_pairs"] > 0]
    flipped = [r for r in rows if r["original_class"] != r["edited_class"]]
    # Only the observed category-changing accounts need this mechanism
    # diagnostic. It changes no raw source and holds the original population
    # radius normalizer fixed while recomputing the local two-layer embedding.
    for row in flipped:
        uid = row["account_id"]
        remove_vertices = {tuple(sorted((index[a], index[b])))
                           for a, b in by_account[uid]}
        keep = torch.as_tensor([
            tuple(sorted((int(u), int(v)))) not in remove_vertices
            for u, v in edge.T.tolist()], dtype=torch.bool)
        with torch.no_grad():
            embedding = model.aggregate(x, mean_adjacency(edge[:, keep], len(ids)))
            fixed_logit = float(fixed_scale_logits(embedding)[index[uid]])
        fixed_probability = float(1 / (1 + np.exp(-fixed_logit)))
        row["edited_target_logit_with_original_population_radius_scale"] = fixed_logit
        row["edited_target_class_with_original_population_radius_scale"] = int(
            fixed_probability >= threshold)
        row["target_graph_embedding_max_abs_change"] = float(torch.max(torch.abs(
            embedding[index[uid]] - baseline_graph_embedding[index[uid]])).item())
    exact_small_degree_subset_audits = []
    for row in flipped:
        uid = row["account_id"]
        pairs = sorted(by_account[uid])
        if len(pairs) > 5:
            continue
        subset_rows = []
        for size in range(len(pairs) + 1):
            for subset in combinations(pairs, size):
                remove_vertices = {tuple(sorted((index[a], index[b])))
                                   for a, b in subset}
                keep = torch.as_tensor([
                    tuple(sorted((int(u), int(v)))) not in remove_vertices
                    for u, v in edge.T.tolist()], dtype=torch.bool)
                with torch.no_grad():
                    logit = float(model(x, mean_adjacency(
                        edge[:, keep], len(ids)))[index[uid]].item())
                probability = float(1 / (1 + np.exp(-logit)))
                subset_rows.append({
                    "removed_pairs": [list(pair) for pair in subset],
                    "removed_pair_count": size,
                    "raw_pointers_cleared": sum(
                        len(e["locators"]) for pair in subset for e in pair_events[pair]),
                    "target_logit": logit,
                    "target_probability": probability,
                    "target_class": int(probability >= threshold),
                    "class_flipped": int(probability >= threshold) != row["original_class"],
                })
        if abs(subset_rows[0]["target_logit"] - row["original_logit"]) > 1e-6:
            raise ValueError("Empty relation subset does not replay baseline")
        if abs(subset_rows[-1]["target_logit"] - row["edited_logit"]) > 1e-6:
            raise ValueError("Full relation subset does not replay source edit")
        flipped_subsets = [s for s in subset_rows if s["class_flipped"]]
        minimum = min(s["removed_pair_count"] for s in flipped_subsets)
        exact_small_degree_subset_audits.append({
            "account_id": uid,
            "selection": "all category-changing test accounts with at most five incident source pairs",
            "tested_subsets": len(subset_rows),
            "minimum_removed_pair_count_for_class_flip": minimum,
            "single_pair_removals_that_flip_class": sum(
                s["class_flipped"] for s in subset_rows
                if s["removed_pair_count"] == 1),
            "single_retained_pairs_that_preserve_original_class": sum(
                not s["class_flipped"] for s in subset_rows
                if s["removed_pair_count"] == len(pairs) - 1),
            "minimum_retained_pairs_to_preserve_original_class": min(
                len(pairs) - s["removed_pair_count"]
                for s in subset_rows if not s["class_flipped"]),
            "minimum_flip_subsets": [s for s in flipped_subsets
                                     if s["removed_pair_count"] == minimum],
            "all_subsets": subset_rows,
        })
    report = {
        "scope": "all Fox8 seed0 test accounts, frozen native-graph HypBot variant",
        "source_operation": "for a target account clear every reply/retweet target-user pointer supporting any observed native graph pair incident to it, including pointers in other accounts' records; keep tweet content and node facts; rebuild symmetric graph",
        "not_an_observed_source_correction": True,
        "official_paper_hypbot_graph_configuration": False,
        "source_index_sha256": file_sha256(EVENT_INDEX),
        "full_raw_interaction_audit_sha256": file_sha256(RAW_AUDIT),
        "raw_source_sha256": file_sha256(RAW),
        "model_sha256": file_sha256(checkpoint_path),
        "saved_test_predictions_sha256": file_sha256(RUN / "test_predictions.json"),
        "node_feature_sha256": hashes["feature_sha256"],
        "graph_sha256": graph_meta["graph_sha256"],
        "max_saved_prediction_replay_difference": replay_error,
        "test_accounts": len(test_ids),
        "test_accounts_with_incident_source_pairs": len(affected),
        "test_accounts_without_incident_source_pairs": len(rows) - len(affected),
        "test_accounts_whose_category_changed": len(flipped),
        "category_changes_preserved_with_original_population_radius_scale": sum(
            r["original_class"] !=
            r["edited_target_class_with_original_population_radius_scale"]
            for r in flipped),
        "max_fixed_scale_baseline_logit_difference": fixed_baseline_error,
        "exact_small_degree_subset_audits": exact_small_degree_subset_audits,
        "test_accounts_with_abs_logit_change_gt_1e_6": sum(
            abs(r["logit_delta"]) > 1e-6 for r in rows),
        "source_locators_verified": raw_verification,
        "threshold": threshold,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "test_accounts", "test_accounts_with_incident_source_pairs",
        "test_accounts_whose_category_changed",
        "category_changes_preserved_with_original_population_radius_scale",
        "test_accounts_with_abs_logit_change_gt_1e_6",
        "source_locators_verified")}, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
