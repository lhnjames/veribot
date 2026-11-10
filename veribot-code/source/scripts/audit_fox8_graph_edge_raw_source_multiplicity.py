"""Measure how many raw interaction records support each Fox8 model graph edge."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/scripts"))
from audit_hypbot_fox8_raw_edge_response import EVENT_INDEX, RAW, RAW_AUDIT  # noqa: E402
from export_hypbot_native_graph_paired import native_symmetric_graph  # noqa: E402
from train_v2_detector import file_sha256  # noqa: E402

OUT = ROOT / "research/certificates/fox8_graph_edge_raw_source_multiplicity.json"
FEATURES = Path("D:/Code/SAHG/SAHG/code/data/fox8_23/features.csv")


def main() -> None:
    payload = json.loads(EVENT_INDEX.read_text(encoding="utf-8"))
    audit = json.loads(RAW_AUDIT.read_text(encoding="utf-8"))
    if (file_sha256(RAW) != payload["raw_source_sha256"]
            or audit["raw_source_sha256"] != payload["raw_source_sha256"]
            or audit["event_index_sha256"] != file_sha256(EVENT_INDEX)
            or audit["status"] != "verified_all_native_interaction_fact_cells_and_event_locators"):
        raise ValueError("Raw interaction source audit differs")
    ids = pd.read_csv(FEATURES, usecols=["user_id"])["user_id"].astype(str).tolist()
    index = {uid: i for i, uid in enumerate(ids)}
    graph, meta = native_symmetric_graph(ROOT, "fox8", ids)
    pair_events = defaultdict(list)
    locator_keys = set()
    for event in payload["events"]:
        source, target = event["source_id"], event["target_id"]
        if source not in index or target not in index or source == target:
            raise ValueError("Source event is not an internal nonself graph event")
        if len(event["locators"]) != event["count"]:
            raise ValueError("Event count differs from raw locator count")
        pair_events[tuple(sorted((source, target)))].append(event)
        for locator in event["locators"]:
            key = (int(locator["line"]), int(locator["tweet_index"]))
            if key in locator_keys:
                raise ValueError("One source tweet is counted for multiple graph edges")
            locator_keys.add(key)
    graph_pairs = {tuple(sorted((ids[int(u)], ids[int(v)])))
                   for u, v in graph.T.tolist()}
    if graph_pairs != set(pair_events) or graph.shape[1] != 2 * len(pair_events):
        raise ValueError("Audited source pairs differ from actual model graph")
    rows = []
    for pair, events in sorted(pair_events.items()):
        locators = [loc for event in events for loc in event["locators"]]
        rows.append({
            "undirected_graph_pair": list(pair),
            "raw_tweet_pointer_count": len(locators),
            "directed_typed_event_records": len(events),
            "relation_types": sorted({event["relation"] for event in events}),
            "direction_count": len({(event["source_id"], event["target_id"])
                                    for event in events}),
            "first_raw_locator": min(locators,
                                     key=lambda loc: (int(loc["line"]),
                                                      int(loc["tweet_index"]))),
        })
    counts = Counter(row["raw_tweet_pointer_count"] for row in rows)
    multi = [row for row in rows if row["raw_tweet_pointer_count"] > 1]
    report = {
        "scope": "every undirected pair in Fox8 native-interaction HypBot graph",
        "raw_source_sha256": file_sha256(RAW),
        "event_index_sha256": file_sha256(EVENT_INDEX),
        "full_source_audit_sha256": file_sha256(RAW_AUDIT),
        "feature_csv_sha256": file_sha256(FEATURES),
        "graph_sha256": meta["graph_sha256"],
        "model_graph_arcs": int(graph.shape[1]),
        "undirected_graph_pairs": len(rows),
        "raw_tweet_pointers": len(locator_keys),
        "single_pointer_pairs": counts[1],
        "multi_pointer_pairs": len(multi),
        "raw_pointers_in_multi_pointer_pairs": sum(
            row["raw_tweet_pointer_count"] for row in multi),
        "max_raw_pointers_supporting_one_graph_pair": max(counts),
        "pairs_with_both_directions_observed": sum(
            row["direction_count"] > 1 for row in rows),
        "pairs_with_multiple_relation_types": sum(
            len(row["relation_types"]) > 1 for row in rows),
        "raw_pointer_count_histogram": {str(k): counts[k] for k in sorted(counts)},
        "all_multi_pointer_pairs": multi,
    }
    if (report["undirected_graph_pairs"] != 4649
            or report["raw_tweet_pointers"] != 5873
            or report["single_pointer_pairs"] != 4262):
        raise ValueError("Full native graph source population differs")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("raw_pointer_count_histogram", "all_multi_pointer_pairs")},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
