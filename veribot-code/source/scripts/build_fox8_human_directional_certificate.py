"""Join a Human directional-support search to its raw source graph."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coalition-report", type=Path, required=True)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.coalition_report.read_text(encoding="utf-8"))
    result = json.loads(args.refinement.read_text(encoding="utf-8"))
    if (result["criterion"] != "directional_human_margin_support" or
            result["stage"] != "single_deletion_stable" or
            result["coalition_report_sha256"] != sha256(args.coalition_report) or
            result["checkpoint_sha256"] != report["checkpoint_sha256"] or
            result["source_sha256"] != report["source_sha256"]):
        raise ValueError("Frozen Human support report contract differs")
    if sha256(args.raw) != result["source_sha256"]:
        raise ValueError("Raw Fox8 archive hash differs")
    case = next((row for row in report["cases"]
                 if row["account_id"] == result["account_id"]), None)
    if case is None or case["predicted_bot"] or case["label"] != 0:
        raise ValueError("Expected a correctly classified Human account")
    threshold = float(result["threshold"])
    buffer = float(result["margin_buffer"])
    if (abs(threshold - float(case["coalition_audit"]["threshold"])) > 1e-6 or
            result["human_gain_over_empty"] < buffer or
            result["human_loss_when_removed_from_full"] < buffer or
            result["retained_only_margin"] >= threshold or
            result["full_margin"] >= threshold):
        raise ValueError("Human directional support criterion failed")
    locators = {row["source_id"]: row for row in result["raw_source_locators"]}
    if (set(locators) != set(result["retained_source_ids"]) or
            len(locators) != result["retained_source_count"]):
        raise ValueError("Refined raw source locators differ")
    graph_nodes = {row["source_id"]: row
                   for row in case["source_graph"]["source_nodes"]}
    for source_id, locator in locators.items():
        node = graph_nodes[source_id]
        for key, value in locator.items():
            if node.get(key) != value:
                raise ValueError(f"Source graph locator differs: {source_id}:{key}")
    wanted_lines = set()
    for node in locators.values():
        if node["kind"] == "profile":
            wanted_lines.add(node["raw_line"])
        elif node["kind"] == "relation":
            wanted_lines.update(item[0] for item in node["raw_locators"])
    raw_rows = {}
    opener = gzip.open if args.raw.suffix == ".gz" else open
    with opener(args.raw, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number in wanted_lines:
                raw_rows[line_number] = json.loads(line)
            if len(raw_rows) == len(wanted_lines):
                break
    if set(raw_rows) != wanted_lines:
        raise ValueError("Raw record line is missing")
    observed = {}
    for source_id, node in locators.items():
        if node["kind"] == "profile":
            row = raw_rows[node["raw_line"]]
            if str(row["user_id"]) != node["account_id"]:
                raise ValueError("Profile raw account differs")
            value = row["user_tweets"][0]["user"][node["field"]]
            serialized = json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    default=str)
            if hashlib.sha256(serialized.encode("utf-8")).hexdigest() != node[
                    "value_sha256"]:
                raise ValueError("Profile raw value hash differs")
            observed[source_id] = {"observed_value": value}
        elif node["kind"] == "relation":
            if node["relation"] != "retweet":
                raise ValueError("Relation raw verifier currently supports retweet")
            tweet_ids = []
            for raw_line, ordinal, _ in node["raw_locators"]:
                row = raw_rows[raw_line]
                tweet = row["user_tweets"][ordinal]
                retweeted_user = tweet["retweeted_status"]["user"]
                target_id = retweeted_user.get("id_str", retweeted_user.get("id"))
                if (str(row["user_id"]) != node["source_account_id"] or
                        str(target_id) != node["target_account_id"]):
                    raise ValueError("Retweet raw relation endpoint differs")
                tweet_ids.append(str(tweet.get("id_str", tweet.get("id", ""))))
            observed[source_id] = {"raw_tweet_ids": tweet_ids}
    last_pass = max(row["pass"] for row in result["trace"])
    final_rows = [row for row in result["trace"] if row["pass"] == last_pass]
    if (len(final_rows) != len(locators) or
            {row["source_id"] for row in final_rows} != set(locators) or
            any(row["accepted"] for row in final_rows)):
        raise ValueError("Final Human source set is not single-deletion stable")

    target = result["account_id"]
    coalition_id = f"coalition:{target}:Human"
    claim_id = f"claim:{target}:Human_directional_support"
    nodes = [
        {"id": f"account:{target}", "type": "account", "account_id": target},
        {"id": coalition_id, "type": "source_coalition",
         "source_count": len(locators)},
        {"id": claim_id, "type": "model_score_support", "class": "Human"},
    ]
    edges = []
    account_ids = {target}
    for source_id in sorted(locators):
        node = dict(locators[source_id])
        node.update(observed[source_id])
        node.update({"id": source_id, "type": "raw_source"})
        nodes.append(node)
        edges.append({"source": source_id, "target": coalition_id,
                      "type": "coalition_member"})
        owners = ((node["source_account_id"], node["target_account_id"])
                  if node["kind"] == "relation" else (node["account_id"],))
        for owner in owners:
            if owner not in account_ids:
                nodes.append({"id": f"account:{owner}", "type": "account",
                              "account_id": owner})
                account_ids.add(owner)
            edges.append({"source": source_id, "target": f"account:{owner}",
                          "type": ("relation_endpoint" if node["kind"] == "relation"
                                   else "profile_owner")})
    for relation in (row for row in locators.values()
                     if row["kind"] == "relation"):
        for profile in (row for row in locators.values()
                        if row["kind"] == "profile" and
                        row["account_id"] in (relation["source_account_id"],
                                              relation["target_account_id"])):
            edges.append({"source": relation["source_id"],
                          "target": profile["source_id"],
                          "type": "relation_links_profile_owner"})
    edges.extend([
        {"source": coalition_id, "target": claim_id,
         "type": "human_gain_over_empty",
         "signed_margin_change": result["human_gain_over_empty"]},
        {"source": coalition_id, "target": claim_id,
         "type": "human_loss_when_removed_from_full",
         "signed_margin_change": result["human_loss_when_removed_from_full"]},
    ])
    output = {
        "account_id": target,
        "checkpoint_sha256": result["checkpoint_sha256"],
        "source_sha256": result["source_sha256"],
        "coalition_report_sha256": result["coalition_report_sha256"],
        "refinement_sha256": sha256(args.refinement),
        "criterion": result["criterion"],
        "margin_buffer": buffer,
        "empty_margin": result["empty_margin"],
        "full_margin": result["full_margin"],
        "retained_only_margin": result["retained_only_margin"],
        "full_without_retained_margin": result["full_without_retained_margin"],
        "human_gain_over_empty": result["human_gain_over_empty"],
        "human_loss_when_removed_from_full":
            result["human_loss_when_removed_from_full"],
        "graph_semantics": (
            "Raw provenance and frozen-model directional score effects; "
            "the criterion does not require a label flip."),
        "nodes": nodes, "edges": edges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False,
                                      sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"account_id": target, "source_count": len(locators),
                      "node_count": len(nodes), "edge_count": len(edges),
                      "sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
