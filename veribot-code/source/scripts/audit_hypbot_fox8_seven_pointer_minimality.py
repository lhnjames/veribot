"""Enumerate raw-pointer deletion subsets for one observed Fox8 class flip.

This distinguishes the granularity of a source edit from a binary model edge.
Only the two distinct resulting graphs require a frozen-model evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import gzip
import json
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

OUT = ROOT / "research/certificates/hypbot_fox8_seven_pointer_minimality.json"
SOURCE_RESPONSE = ROOT / "research/certificates/hypbot_fox8_account_relation_source_response.json"
PAIR_MULTIPLICITY = ROOT / "research/certificates/fox8_graph_edge_raw_source_multiplicity.json"
ACCOUNT = "415566147"
NEIGHBOR = "25073877"


def main() -> None:
    sys.path.insert(0, str(REPO.resolve()))
    from hypbot.data import mean_adjacency
    from hypbot.model import HypBot
    from train import PRESETS

    torch.set_num_threads(4)
    response = json.loads(SOURCE_RESPONSE.read_text(encoding="utf-8"))
    multiplicity = json.loads(PAIR_MULTIPLICITY.read_text(encoding="utf-8"))
    event_index = json.loads(EVENT_INDEX.read_text(encoding="utf-8"))
    raw_audit = json.loads(RAW_AUDIT.read_text(encoding="utf-8"))
    frozen = json.loads((RUN / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = frozen["manifest"]
    checkpoint_path = RUN / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    args = argparse.Namespace(dataset="fox8", data_dir=DATA, account_file=None,
                              raw_source=[RAW], knn_k=10)
    x, _, split, ids, _, _, hashes, _ = load_data(args, 0)
    edge, graph_meta = native_symmetric_graph(ROOT, "fox8", ids)
    if (file_sha256(RAW) != event_index["raw_source_sha256"]
            or raw_audit["raw_source_sha256"] != file_sha256(RAW)
            or raw_audit["event_index_sha256"] != file_sha256(EVENT_INDEX)
            or file_sha256(EVENT_INDEX) != graph_meta["source_index_sha256"]
            or file_sha256(checkpoint_path) != manifest["model_sha256"]
            or hashes["feature_sha256"] != manifest["node_feature_sha256"]
            or hashes["split_sha256"] != manifest["split_sha256"]
            or checkpoint["split_sha256"] != hashes["split_sha256"]
            or graph_meta != manifest["native_graph"]
            or response["raw_source_sha256"] != file_sha256(RAW)
            or response["source_index_sha256"] != file_sha256(EVENT_INDEX)
            or response["model_sha256"] != file_sha256(checkpoint_path)
            or response["graph_sha256"] != graph_meta["graph_sha256"]
            or multiplicity["graph_sha256"] != graph_meta["graph_sha256"]):
        raise ValueError("Raw/model/source contract differs from earlier audits")

    index = {uid: i for i, uid in enumerate(ids)}
    pair = tuple(sorted((ACCOUNT, NEIGHBOR)))
    incident = [e for e in event_index["events"]
                if ACCOUNT in (e["source_id"], e["target_id"])
                and e["source_id"] in index and e["target_id"] in index
                and e["source_id"] != e["target_id"]]
    if len(incident) != 1:
        raise ValueError("Target must have exactly one source-backed graph pair")
    event = incident[0]
    if (tuple(sorted((event["source_id"], event["target_id"]))) != pair
            or event["source_id"] != ACCOUNT
            or event["relation"] != "reply"
            or event["count"] != 7
            or len(event["locators"]) != 7):
        raise ValueError("Seven-pointer event differs")
    locators = event["locators"]
    if len({(l["line"], l["tweet_index"]) for l in locators}) != 7:
        raise ValueError("Raw locators are not unique")
    if len({int(l["line"]) for l in locators}) != 1:
        raise ValueError("Raw locators no longer share one account row")
    line_number = int(locators[0]["line"])
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        row = next(json.loads(line) for i, line in enumerate(stream, 1)
                   if i == line_number)
    if str(row["user_id"]) != ACCOUNT:
        raise ValueError("Raw account differs")
    original_tweets = row["user_tweets"]
    for locator in locators:
        tweet = original_tweets[int(locator["tweet_index"])]
        tweet_id = str(tweet.get("id_str") or tweet.get("id"))
        target = tweet.get("in_reply_to_user_id_str") or tweet.get("in_reply_to_user_id")
        if tweet_id != str(locator["tweet_id"]) or str(target) != NEIGHBOR:
            raise ValueError("Locator does not bind to the expected raw reply")

    cfg = PRESETS["fox8"]
    model = HypBot(x.shape[1], K=manifest["num_sectors"], gated=False,
                   **cfg["model"], graph_width=x.shape[1], sparse_adj=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    remove_vertices = tuple(sorted((index[ACCOUNT], index[NEIGHBOR])))
    keep = torch.as_tensor([
        tuple(sorted((int(u), int(v)))) != remove_vertices
        for u, v in edge.T.tolist()], dtype=torch.bool)
    if int((~keep).sum()) != 2:
        raise ValueError("Expected one symmetric graph pair")
    with torch.no_grad():
        original_logits = model(x, mean_adjacency(edge, len(ids))).cpu().numpy()
        deleted_logits = model(x, mean_adjacency(edge[:, keep], len(ids))).cpu().numpy()
    saved_row = next(r for r in response["rows"] if r["account_id"] == ACCOUNT)
    target_index = index[ACCOUNT]
    if (target_index not in set(map(int, split["test"]))
            or abs(float(original_logits[target_index]) - saved_row["original_logit"]) > 1e-6
            or abs(float(deleted_logits[target_index]) - saved_row["edited_logit"]) > 1e-6):
        raise ValueError("Frozen source response is not independently replayed")
    threshold = float(manifest["threshold"])
    class_by_state = {}
    logit_by_state = {"pair_present": float(original_logits[target_index]),
                      "pair_absent": float(deleted_logits[target_index])}
    for state, logit in logit_by_state.items():
        class_by_state[state] = int(1 / (1 + np.exp(-logit)) >= threshold)
    if class_by_state != {"pair_present": 0, "pair_absent": 1}:
        raise ValueError("Expected Human-to-Bot flip no longer occurs")

    subset_counts = Counter()
    flip_counts = Counter()
    graph_state_counts = Counter()
    for size in range(8):
        for subset in combinations(range(7), size):
            edited_tweets = list(original_tweets)
            for locator_index in subset:
                j = int(locators[locator_index]["tweet_index"])
                changed = dict(edited_tweets[j])
                changed["in_reply_to_user_id_str"] = None
                changed["in_reply_to_user_id"] = None
                edited_tweets[j] = changed
            if tweet_features(edited_tweets) != tweet_features(original_tweets):
                raise ValueError("Clearing target-user pointers changed node facts")
            live = sum(
                (edited_tweets[int(l["tweet_index"])].get("in_reply_to_user_id_str")
                 or edited_tweets[int(l["tweet_index"])].get("in_reply_to_user_id"))
                is not None for l in locators)
            if live != 7 - size:
                raise ValueError("Unexpected number of live raw pointers")
            state = "pair_present" if live else "pair_absent"
            # This graph builder deduplicates directed events and symmetrizes
            # them; all partial subsets therefore produce the original graph.
            classification = class_by_state[state]
            subset_counts[size] += 1
            flip_counts[size] += classification != class_by_state["pair_present"]
            graph_state_counts[state] += 1
    if (sum(subset_counts.values()) != 128
            or graph_state_counts != {"pair_present": 127, "pair_absent": 1}
            or flip_counts[7] != 1
            or any(flip_counts[i] for i in range(7))):
        raise ValueError("Seven-source exhaustive result differs")

    affected = [r for r in response["rows"]
                if r["incident_unique_undirected_pairs"]]
    flipped = [r for r in response["rows"]
               if r["original_class"] != r["edited_class"]]
    multi_affected = [r for r in affected if
                      r["raw_target_user_pointers_cleared"] >
                      r["incident_unique_undirected_pairs"]]
    multi_flipped = [r for r in flipped if
                     r["raw_target_user_pointers_cleared"] >
                     r["incident_unique_undirected_pairs"]]
    if (len(affected), len(flipped), len(multi_affected), len(multi_flipped)) != (220, 25, 80, 7):
        raise ValueError("Account-level source multiplicity count differs")
    report = {
        "scope": "one previously observed category-changing Fox8 test account, selected by seven raw pointers supporting its sole graph pair",
        "source_operation": "clear a chosen subset of reply target-user pointers, retain tweets and node facts, then deduplicate and symmetrize native graph",
        "not_an_observed_source_correction": True,
        "official_paper_hypbot_graph_configuration": False,
        "account_id": ACCOUNT,
        "neighbor_id": NEIGHBOR,
        "raw_source_sha256": file_sha256(RAW),
        "source_index_sha256": file_sha256(EVENT_INDEX),
        "full_raw_audit_sha256": file_sha256(RAW_AUDIT),
        "model_sha256": file_sha256(checkpoint_path),
        "saved_predictions_sha256": file_sha256(RUN / "test_predictions.json"),
        "source_response_sha256": file_sha256(SOURCE_RESPONSE),
        "pair_multiplicity_sha256": file_sha256(PAIR_MULTIPLICITY),
        "graph_sha256": graph_meta["graph_sha256"],
        "node_feature_sha256": hashes["feature_sha256"],
        "raw_account_line": line_number,
        "raw_tweet_locators": locators,
        "source_pointers_supporting_pair": 7,
        "encoded_graph_pair_count": 1,
        "enumerated_raw_pointer_subsets": sum(subset_counts.values()),
        "distinct_model_input_graph_states": len(graph_state_counts),
        "subset_counts_by_removed_pointer_count": dict(sorted(subset_counts.items())),
        "class_flip_counts_by_removed_pointer_count": dict(sorted(flip_counts.items())),
        "graph_state_subset_counts": dict(graph_state_counts),
        "minimum_removed_raw_pointer_count_for_class_flip": min(
            k for k, count in flip_counts.items() if count),
        "minimum_removed_encoded_graph_pair_count_for_class_flip": 1,
        "frozen_target_logit_by_graph_state": logit_by_state,
        "frozen_target_class_by_graph_state": class_by_state,
        "threshold": threshold,
        "test_accounts_with_incident_pairs": len(affected),
        "test_accounts_with_multi_supported_incident_pairs": len(multi_affected),
        "test_accounts_with_class_flip": len(flipped),
        "flipped_test_accounts_with_multi_supported_incident_pairs": len(multi_flipped),
        "multi_supported_flipped_account_ids": [r["account_id"] for r in multi_flipped],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("raw_tweet_locators", "multi_supported_flipped_account_ids")},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
