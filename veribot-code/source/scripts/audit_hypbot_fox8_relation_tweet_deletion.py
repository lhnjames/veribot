"""Compare pointer clearing with coherent raw tweet deletion on frozen HypBot.

All tweet records supporting a target's incident native graph pairs are
deleted in memory. Every affected account's 31 source features are rebuilt,
all native relation events from those tweets are removed, and the frozen
Fox8 seed0 native-graph HypBot is replayed with the training scaler fixed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/scripts"))
sys.path.insert(0, str(ROOT / "source"))
from audit_hypbot_fox8_raw_edge_response import DATA, EVENT_INDEX, RAW, REPO, RUN  # noqa: E402
from export_hypbot_native_graph_paired import native_symmetric_graph  # noqa: E402
from train_v2_detector import file_sha256, load_data  # noqa: E402
from verifiable_bot.fox8_adapter import (  # noqa: E402
    FEATURE_FIELDS, profile_features, tweet_features,
)

SOURCE_REPORT = ROOT / "research/certificates/hypbot_fox8_account_relation_source_response.json"
OUT = ROOT / "research/certificates/hypbot_fox8_relation_tweet_deletion.json"
OUT_ALL = ROOT / "research/certificates/hypbot_fox8_all_incident_tweet_deletion.json"


def raw_features(tweets: list[dict], retained_profile: dict | None = None) -> np.ndarray:
    if not tweets and retained_profile is None:
        raise ValueError("Empty tweet sample requires explicit retained profile snapshot")
    # The account profile remains a separate fact even if every sampled tweet
    # is deleted. The raw release embeds the snapshot in each tweet's user.
    profile = tweets[0]["user"] if tweets else retained_profile
    fields = {**profile_features(profile), **tweet_features(tweets)}
    return np.asarray([fields[name] for name in FEATURE_FIELDS], dtype=np.float32)


def graph_after_tweet_deletions(event_index: dict, account_index: dict[str, int],
                                deleted: set[tuple[int, int]]) -> tuple[torch.Tensor, int]:
    observed = set()
    removed_event_records = 0
    for event in event_index["events"]:
        u = account_index.get(event["source_id"])
        v = account_index.get(event["target_id"])
        if u is None or v is None or u == v:
            continue
        surviving = [loc for loc in event["locators"]
                     if (int(loc["line"]), int(loc["tweet_index"])) not in deleted]
        if surviving:
            observed.add((u, v))
        else:
            removed_event_records += 1
    symmetric = sorted(observed | {(v, u) for u, v in observed})
    return torch.as_tensor(np.asarray(symmetric, dtype=np.int64).T.copy()), removed_event_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("low_degree_flips", "all_incident"),
                        default="low_degree_flips")
    options = parser.parse_args()
    sys.path.insert(0, str(REPO.resolve()))
    from hypbot.data import mean_adjacency
    from hypbot.model import HypBot
    from train import PRESETS

    torch.set_num_threads(4)
    pointer = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    event_index = json.loads(EVENT_INDEX.read_text(encoding="utf-8"))
    frozen = json.loads((RUN / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = frozen["manifest"]
    checkpoint_path = RUN / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    args = argparse.Namespace(dataset="fox8", data_dir=DATA, account_file=None,
                              raw_source=[DATA / "features.csv"], knn_k=10)
    x, labels, split, ids, _, _, hashes, scaler = load_data(args, 0)
    edge, graph_meta = native_symmetric_graph(ROOT, "fox8", ids)
    if (file_sha256(RAW) != pointer["raw_source_sha256"]
            or file_sha256(EVENT_INDEX) != pointer["source_index_sha256"]
            or file_sha256(checkpoint_path) != manifest["model_sha256"]
            or pointer["model_sha256"] != manifest["model_sha256"]
            or hashes["feature_sha256"] != manifest["node_feature_sha256"]
            or pointer["node_feature_sha256"] != hashes["feature_sha256"]
            or hashes["split_sha256"] != manifest["split_sha256"]
            or graph_meta != manifest["native_graph"]
            or pointer["graph_sha256"] != graph_meta["graph_sha256"]):
        raise ValueError("Frozen source or native graph differs")
    frame = pd.read_csv(DATA / "features.csv")
    if list(frame.columns) != ["user_id", *FEATURE_FIELDS]:
        raise ValueError("Saved feature columns differ")
    raw_x = frame[list(FEATURE_FIELDS)].fillna(0.0).to_numpy(dtype=np.float32)
    index = {uid: i for i, uid in enumerate(ids)}
    baseline_edge_from_events, n_removed = graph_after_tweet_deletions(
        event_index, index, set())
    if n_removed or not torch.equal(baseline_edge_from_events, edge):
        raise ValueError("Raw event graph does not replay native model graph")
    model = HypBot(x.shape[1], K=manifest["num_sectors"], gated=False,
                   **PRESETS["fox8"]["model"], graph_width=x.shape[1],
                   sparse_adj=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        baseline = model(x, mean_adjacency(edge, len(ids))).cpu().numpy()
    baseline_prob = 1 / (1 + np.exp(-baseline))
    test = np.asarray(split["test"], dtype=np.int64)
    if [ids[int(j)] for j in test] != [r["account_id"] for r in frozen["predictions"]]:
        raise ValueError("Frozen test account order differs")
    replay_error = float(np.max(np.abs(baseline_prob[test] -
                                   [r["probability"] for r in frozen["predictions"]])))
    if replay_error > 1e-6:
        raise ValueError("Frozen native graph checkpoint does not replay")
    pair_events = defaultdict(list)
    for event in event_index["events"]:
        a, b = event["source_id"], event["target_id"]
        if a in index and b in index and a != b:
            pair_events[tuple(sorted((a, b)))].append(event)
    if options.scope == "low_degree_flips":
        selected = [a["account_id"] for a in pointer["exact_small_degree_subset_audits"]]
        if len(selected) != 3:
            raise ValueError("Expected three prior low-degree category-changing accounts")
    else:
        selected = [r["account_id"] for r in pointer["rows"]
                    if r["incident_unique_undirected_pairs"] > 0]
        if len(selected) != pointer["test_accounts_with_incident_source_pairs"]:
            raise ValueError("Full source-response population differs")
    deleted_by_target = {}
    union_lines = set()
    for uid in selected:
        pairs = [pair for pair in pair_events if uid in pair]
        deleted = {(int(loc["line"]), int(loc["tweet_index"]))
                   for pair in pairs for event in pair_events[pair]
                   for loc in event["locators"]}
        deleted_by_target[uid] = deleted
        union_lines.update(line for line, _ in deleted)
    raw_rows = {}
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number in union_lines:
                raw_rows[line_number] = json.loads(line)
            if len(raw_rows) == len(union_lines):
                break
    if set(raw_rows) != union_lines:
        raise ValueError("One or more raw tweet source lines missing")
    threshold = float(manifest["threshold"])
    rows = []
    for uid in selected:
        deleted = deleted_by_target[uid]
        expected_tweet_ids = {}
        for pair, events in pair_events.items():
            if uid not in pair:
                continue
            for event in events:
                for loc in event["locators"]:
                    key = (int(loc["line"]), int(loc["tweet_index"]))
                    previous = expected_tweet_ids.setdefault(key, str(loc["tweet_id"]))
                    if previous != str(loc["tweet_id"]):
                        raise ValueError("One raw tweet locator has conflicting tweet IDs")
        if set(expected_tweet_ids) != deleted:
            raise ValueError("Deleted tweet locators differ from supporting events")
        edited_raw_x = raw_x.copy()
        affected_lines = sorted({line for line, _ in deleted})
        changed_feature_accounts = []
        seen_source_accounts = set()
        for line in affected_lines:
            row = raw_rows[line]
            source_uid = str(row["user_id"])
            if source_uid in seen_source_accounts:
                raise ValueError("Multiple raw rows for one affected source account")
            seen_source_accounts.add(source_uid)
            j = index[source_uid]
            tweets = row["user_tweets"]
            before = raw_features(tweets)
            if not np.allclose(before, raw_x[j], rtol=1e-5, atol=1e-5):
                raise ValueError(f"Raw account {source_uid} does not rebuild CSV")
            remaining = [tweet for k, tweet in enumerate(tweets)
                         if (line, k) not in deleted]
            after = raw_features(remaining, tweets[0]["user"])
            edited_raw_x[j] = after
            changed_feature_accounts.append({
                "account_id": source_uid,
                "raw_gzip_line": line,
                "tweets_before": len(tweets),
                "tweets_after": len(remaining),
                "profile_snapshot_retained_after_empty_sample": len(remaining) == 0,
                "changed_feature_columns": [FEATURE_FIELDS[k] for k in range(len(FEATURE_FIELDS))
                                            if before[k] != after[k]],
            })
        edited_x = torch.from_numpy(np.clip(
            scaler.transform(edited_raw_x), -3, 3).astype(np.float32))
        edited_edge, removed_events = graph_after_tweet_deletions(
            event_index, index, deleted)
        with torch.no_grad():
            edited = model(edited_x, mean_adjacency(edited_edge, len(ids))).cpu().numpy()
            feature_only = model(edited_x, mean_adjacency(edge, len(ids))).cpu().numpy()
        j = index[uid]
        edited_probability = float(1 / (1 + np.exp(-edited[j])))
        feature_only_probability = float(1 / (1 + np.exp(-feature_only[j])))
        prior = next(r for r in pointer["rows"] if r["account_id"] == uid)
        if (abs(prior["original_logit"] - float(baseline[j])) > 1e-6
                or prior["original_class"] != int(baseline_prob[j] >= threshold)):
            raise ValueError("Pointer response baseline differs from this frozen model")
        minimum_arcs_removed = 2 * prior["incident_unique_undirected_pairs"]
        actual_arcs_removed = edge.shape[1] - edited_edge.shape[1]
        if actual_arcs_removed < minimum_arcs_removed:
            raise ValueError("Coherent deletion failed to remove all target incident pairs")
        rows.append({
            "account_id": uid,
            "raw_tweet_locators_deleted": [
                {"source_account_id": str(raw_rows[line]["user_id"]),
                 "line": line, "tweet_index": tweet_index,
                 "tweet_id": str(raw_rows[line]["user_tweets"][tweet_index].get("id_str")
                                 or raw_rows[line]["user_tweets"][tweet_index].get("id")
                                 or tweet_index)}
                for line, tweet_index in sorted(deleted)],
            "raw_tweets_deleted": len(deleted),
            "source_account_rows_affected": len(affected_lines),
            "source_feature_recomputations": changed_feature_accounts,
            "directed_event_records_removed": removed_events,
            "native_graph_arcs_before": edge.shape[1],
            "native_graph_arcs_after": edited_edge.shape[1],
            "additional_arcs_removed_from_other_pairs": actual_arcs_removed - minimum_arcs_removed,
            "original_target_logit": float(baseline[j]),
            "tweet_deletion_target_logit": float(edited[j]),
            "pointer_only_target_logit": prior["edited_logit"],
            "feature_only_target_logit_model_input_control": float(feature_only[j]),
            "graph_path_delta_original_features": float(
                prior["edited_logit"] - baseline[j]),
            "feature_path_delta_original_graph_model_input_control": float(
                feature_only[j] - baseline[j]),
            "feature_path_delta_edited_graph": float(
                edited[j] - prior["edited_logit"]),
            "graph_feature_interaction_logit": float(
                edited[j] - prior["edited_logit"] - feature_only[j] + baseline[j]),
            "original_target_class": int(baseline_prob[j] >= threshold),
            "tweet_deletion_target_class": int(edited_probability >= threshold),
            "pointer_only_target_class": prior["edited_class"],
            "feature_only_target_class_model_input_control": int(
                feature_only_probability >= threshold),
            "tweet_deletion_target_probability": edited_probability,
        })
        if any(loc["tweet_id"] != expected_tweet_ids[(loc["line"], loc["tweet_index"])]
               for loc in rows[-1]["raw_tweet_locators_deleted"]):
            raise ValueError("Deleted raw tweet IDs differ from source event locators")
        if len(rows) % 25 == 0:
            print(f"evaluated {len(rows)}/{len(selected)} accounts", flush=True)
    report = {
        "scope": ("three prior low-degree category-changing Fox8 test accounts"
                  if options.scope == "low_degree_flips"
                  else "all source-incident Fox8 test accounts") +
                 " on frozen native-graph HypBot",
        "operation": "remove all raw tweets supporting any native relation pair incident to target; recompute all affected node features and all surviving native graph pairs",
        "empty_sample_profile_policy": "retain the original account profile snapshot when all sampled tweets are deleted",
        "not_an_observed_source_correction": True,
        "source_report_sha256": file_sha256(SOURCE_REPORT),
        "raw_source_sha256": file_sha256(RAW),
        "event_index_sha256": file_sha256(EVENT_INDEX),
        "feature_csv_sha256": file_sha256(DATA / "features.csv"),
        "model_sha256": file_sha256(checkpoint_path),
        "max_saved_probability_replay_error": replay_error,
        "target_accounts": len(rows),
        "pointer_only_class_changes": sum(
            r["pointer_only_target_class"] != r["original_target_class"] for r in rows),
        "tweet_deletion_class_changes": sum(
            r["tweet_deletion_target_class"] != r["original_target_class"] for r in rows),
        "tweet_deletion_matches_pointer_only_class": sum(
            r["tweet_deletion_target_class"] == r["pointer_only_target_class"] for r in rows),
        "feature_only_model_input_control_class_changes": sum(
            r["feature_only_target_class_model_input_control"] !=
            r["original_target_class"] for r in rows),
        "operations_removing_additional_other_graph_arcs": sum(
            r["additional_arcs_removed_from_other_pairs"] > 0 for r in rows),
        "operations_with_empty_tweet_sample_in_source_account": sum(
            any(s["profile_snapshot_retained_after_empty_sample"]
                for s in r["source_feature_recomputations"]) for r in rows),
        "rows": rows,
    }
    output = OUT if options.scope == "low_degree_flips" else OUT_ALL
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
