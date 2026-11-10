"""Join frozen coalition results to raw source locators and native events."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import gzip
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coalitions", type=Path, required=True)
    parser.add_argument("--locators", type=Path,
                        help="Separate raw-locator certificate; omit when nodes already contain locators")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coalitions = json.loads(args.coalitions.read_text(encoding="utf-8"))
    lookup = None
    if args.locators is not None:
        with gzip.open(args.locators, "rt", encoding="utf-8") as stream:
            locators = json.load(stream)
        for key in ("checkpoint_sha256", "source_sha256", "selected_account_ids"):
            if coalitions[key] != locators[key]:
                raise ValueError(f"coalition and locator contracts differ: {key}")
        lookup = {case["account_id"]: case for case in locators["cases"]}
        if len(lookup) != len(coalitions["cases"]):
            raise ValueError("case populations differ")
    summaries = []
    for case in coalitions["cases"]:
        source_graph = case["source_graph"]
        nodes = source_graph["source_nodes"]
        if lookup is not None:
            reference = lookup[case["account_id"]]
            source_locators = {node["source_id"]: node
                               for node in reference["raw_source_locators"]}
            if (len(nodes) != source_graph["source_node_count"] or
                    len(source_locators) != len(nodes) or
                    len(nodes) != reference["full_dependency_source_count"]):
                raise ValueError("graph source count differs from locator certificate")
            for node in nodes:
                locator = source_locators[node["source_id"]]
                for key, value in locator.items():
                    if key in node and node[key] != value:
                        raise ValueError("graph source locator differs")
                    node[key] = value
            if {node["source_id"] for node in nodes} != set(source_locators):
                raise ValueError("graph source IDs differ from locator certificate")
        else:
            if (len(nodes) != source_graph["source_node_count"] or
                    len(nodes) != case["coalition_audit"]["full_dependency_source_count"] or
                    len({node["source_id"] for node in nodes}) != len(nodes)):
                raise ValueError("embedded graph source counts or IDs differ")
            for node in nodes:
                required = ({"account_id", "raw_line", "field", "value_sha256"}
                            if node["kind"] == "profile" else
                            {"account_id", "raw_line", "tweet_ordinal",
                             "tweet_id", "record_sha256"}
                            if node["kind"] == "tweet" else
                            {"source_account_id", "target_account_id",
                             "relation", "raw_locators"})
                if not required.issubset(node):
                    raise ValueError("embedded source lacks a raw locator")
        tweets = {(node["raw_line"], node["tweet_ordinal"]): node
                  for node in nodes if node["kind"] == "tweet"}
        if len(tweets) != sum(node["kind"] == "tweet" for node in nodes):
            raise ValueError("raw tweet locator is not unique")
        links = []
        for relation in (node for node in nodes if node["kind"] == "relation"):
            for raw_line, ordinal, raw_field in relation["raw_locators"]:
                tweet = tweets.get((raw_line, ordinal))
                if tweet is None:
                    raise ValueError("relation event has no raw tweet source")
                if tweet["account_id"] != relation["source_account_id"]:
                    raise ValueError("relation event and tweet source account differ")
                links.append({
                    "tweet_source_id": tweet["source_id"],
                    "relation_source_id": relation["source_id"],
                    "raw_field": raw_field,
                })
        source_graph["tweet_relation_links"] = links
        adjacency: dict[str, set[str]] = defaultdict(set)
        for node in nodes:
            source = "s:" + node["source_id"]
            owners = ((node["source_account_id"], node["target_account_id"])
                      if node["kind"] == "relation" else (node["account_id"],))
            for owner in owners:
                account = "a:" + owner
                adjacency[source].add(account)
                adjacency[account].add(source)
        start = "a:" + source_graph["root_account_id"]
        seen = {start}
        queue = deque((start,))
        while queue:
            for neighbor in adjacency[queue.popleft()] - seen:
                seen.add(neighbor)
                queue.append(neighbor)
        expected = len(nodes) + len(source_graph["account_nodes"])
        if len(seen) != expected:
            raise ValueError("source graph contains disconnected nodes")
        summaries.append({"account_id": case["account_id"],
                          "source_nodes": len(nodes),
                          "account_nodes": len(source_graph["account_nodes"]),
                          "tweet_relation_links": len(links)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(coalitions, ensure_ascii=False).encode("utf-8")
    with args.output.open("wb") as raw_stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream,
                           mtime=0) as compressed:
            compressed.write(payload)
    print(json.dumps({"output": str(args.output.resolve()),
                      "cases": summaries}), flush=True)


if __name__ == "__main__":
    main()
