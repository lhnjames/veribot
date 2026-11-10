"""Build a provenance and intervention graph for a frozen Fox8 Bot coalition."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def build_graph(coalition: dict, refinement: dict) -> dict:
    account_id = refinement["account_id"]
    case = next((item for item in coalition["cases"]
                 if item["account_id"] == account_id), None)
    if case is None or not case["predicted_bot"]:
        raise ValueError("Expected a Bot case in the frozen coalition audit")
    if refinement["stage"] != "single_deletion_stable":
        raise ValueError("Refinement has not completed")
    if refinement["checkpoint_sha256"] != coalition["checkpoint_sha256"]:
        raise ValueError("Model checkpoint differs")
    if refinement["source_sha256"] != coalition["source_sha256"]:
        raise ValueError("Raw source differs")
    audit = case["coalition_audit"]
    threshold = float(audit["threshold"])
    buffer = float(refinement["margin_buffer"])
    if abs(threshold - float(refinement["threshold"])) > 1e-6:
        raise ValueError("Decision threshold differs")
    keep = float(refinement["retained_margin"])
    remove = float(refinement["margin_after_removing_refined_set_from_full"])
    if not (keep >= threshold + buffer and remove < threshold - buffer and
            refinement["refined_set_collectively_necessary_in_full"]):
        raise ValueError("Bidirectional intervention contract failed")

    graph_nodes = {node["source_id"]: node
                   for node in case["source_graph"]["source_nodes"]}
    kept_ids = refinement["retained_source_ids"]
    locators = {node["source_id"]: node
                for node in refinement["raw_source_locators"]}
    if len(kept_ids) != len(set(kept_ids)) or set(kept_ids) != set(locators):
        raise ValueError("Refined source identifiers or locators differ")
    if not set(kept_ids).issubset(graph_nodes):
        raise ValueError("Refined source missing from full calculation graph")
    for source_id, locator in locators.items():
        node = graph_nodes[source_id]
        for key, value in locator.items():
            if key in node and node[key] != value:
                raise ValueError(f"Source locator mismatch: {source_id}:{key}")

    root_id = f"account:{account_id}"
    set_id = f"coalition:{account_id}"
    claim_id = f"claim:{account_id}:Bot"
    nodes = [
        {"id": root_id, "type": "account", "account_id": account_id},
        {"id": set_id, "type": "source_coalition",
         "source_count": len(kept_ids)},
        {"id": claim_id, "type": "model_classification", "class": "Bot"},
    ]
    edges = []
    groups = sorted({graph_nodes[source_id]["group"] for source_id in kept_ids})
    for group in groups:
        group_id = f"group:{account_id}:{group}"
        nodes.append({"id": group_id, "type": "source_group", "name": group})
        edges.append({"source": group_id, "target": set_id,
                      "type": "group_in_coalition"})
    for source_id in sorted(kept_ids):
        node = dict(graph_nodes[source_id])
        node.update(locators[source_id])
        node["id"] = source_id
        node["type"] = "raw_source"
        nodes.append(node)
        edges.append({"source": source_id,
                      "target": f"group:{account_id}:{node['group']}",
                      "type": "member_of_group"})
        edges.append({"source": source_id, "target": root_id,
                      "type": "observed_for_account"})
    edges.extend([
        {"source": set_id, "target": claim_id,
         "type": "keep_only_preserves_Bot", "bot_margin": keep},
        {"source": set_id, "target": claim_id,
         "type": "remove_from_full_changes_to_Human", "bot_margin": remove},
    ])
    interactions = [row for row in audit["pair_interactions"]
                    if row["first"] in groups and row["second"] in groups]
    for row in interactions:
        edges.append({"source": f"group:{account_id}:{row['first']}",
                      "target": f"group:{account_id}:{row['second']}",
                      "type": "measured_pair_interaction",
                      "signed_pair_interaction": row["signed_pair_interaction"],
                      "joint_preserves_original_class":
                          row["joint_preserves_original_class"]})
    return {
        "account_id": account_id,
        "checkpoint_sha256": refinement["checkpoint_sha256"],
        "source_sha256": refinement["source_sha256"],
        "threshold": threshold,
        "margin_buffer": buffer,
        "full_input_bot_margin": case["baseline_margin"],
        "retained_only_bot_margin": keep,
        "full_input_without_coalition_bot_margin": remove,
        "source_count": len(kept_ids),
        "single_deletion_stable": True,
        "graph_semantics": (
            "Raw-source provenance, group interaction, and exact frozen-model "
            "intervention effects; it does not establish real-world source truth."),
        "nodes": nodes,
        "edges": edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coalition-report", type=Path, required=True)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coalition = load_json(args.coalition_report)
    refinement = load_json(args.refinement)
    if (refinement["coalition_report_sha256"] != sha256(args.coalition_report)):
        raise ValueError("Coalition report hash differs")
    graph = build_graph(coalition, refinement)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, ensure_ascii=False, sort_keys=True,
                                      indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"account_id": graph["account_id"],
                      "source_count": graph["source_count"],
                      "node_count": len(graph["nodes"]),
                      "edge_count": len(graph["edges"]),
                      "output_sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
