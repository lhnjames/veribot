"""Explicit source-to-input-to-model DAG for a checked VeriBot certificate."""

from __future__ import annotations

from collections import deque
from typing import Any


def build_evidence_dag(report: dict[str, Any]) -> dict[str, Any]:
    cert = report["certificate"]
    records = report.get("selected_source_records", report.get("source_records"))
    candidate_ids = tuple(cert["claim"]["source_ids"])
    control_ids = tuple(cert["controls"][0]["source_ids"])
    nodes: dict[str, dict] = {}
    edges: list[dict[str, str]] = []

    def node(identifier: str, kind: str, **payload: Any) -> None:
        if identifier in nodes:
            raise ValueError(f"duplicate evidence node: {identifier}")
        nodes[identifier] = {"kind": kind, **payload}

    def link(source: str, target: str, relation: str) -> None:
        if source not in nodes or target not in nodes:
            raise ValueError("DAG link refers to a missing node")
        edges.append({"from": source, "to": target, "relation": relation})

    for sid in (*candidate_ids, *control_ids):
        record = records[sid]
        node(f"source:{sid}", "raw_source", source_id=sid,
             location=record["raw_location"], sha256=record["sha256"],
             payload=record["payload"])
        if "bridge_records" in report:
            bridge = report["bridge_records"][sid]
            node(f"bridge:{sid}", "graph_bridge_source",
                 location=bridge["raw_location"], sha256=bridge["sha256"],
                 payload=bridge["payload"])
    node("claim", "recomputed_fact", operation=cert["claim"]["operation"],
         value=cert["claim"]["recomputed_value"])
    for sid in candidate_ids:
        link(f"source:{sid}", "claim", "fact_recomputed_from")
        for position, input_path in enumerate(cert["input_dependencies"][sid]):
            identifier = f"named_input:{sid}:{position}"
            node(identifier, "named_input_fact", input_path=input_path,
                 source_id=sid)
            link(f"source:{sid}", identifier, "deterministically_derived_from")
    node("source_edit", "source_operation", action="remove_candidate_sources",
         source_ids=list(candidate_ids))
    link("claim", "source_edit", "tests_model_dependence_of")
    for state in ("before", "after"):
        trace = cert[state]
        node(f"input:{state}", "model_input",
             sha256=trace["model_input_sha256"],
             graph_rule_sha256=cert["graph_rule_sha256"])
        if state == "before":
            link("claim", f"input:{state}", "partial_input_dependency")
            for sid in candidate_ids:
                for position, _ in enumerate(cert["input_dependencies"][sid]):
                    link(f"named_input:{sid}:{position}", f"input:{state}",
                         "read_as_model_input")
        else:
            link("source_edit", f"input:{state}", "rebuilds_after_edit")
        if "bridge_records" in report:
            for sid in (*candidate_ids, *control_ids):
                link(f"bridge:{sid}", f"input:{state}", "retained_native_graph_relation")
        for branch, stages in (("hgt", trace["hgt_stage_sha256"]),
                               ("relnet", trace["relnet_stage_sha256"])):
            previous = f"input:{state}"
            for stage, fingerprint in stages.items():
                identifier = f"{branch}:{stage}:{state}"
                node(identifier, "model_stage", branch=branch, stage=stage,
                     sha256=fingerprint,
                     model_sha256=cert[f"{branch}_sha256"])
                link(previous, identifier, "computed_by_frozen_branch")
                previous = identifier
            output = f"{branch}:logit:{state}"
            node(output, "branch_logit", branch=branch,
                 value=trace[f"{branch}_logit"])
            link(previous, output, "classified_from_representation")
        node(f"fusion:{state}", "mean_logit_fusion",
             value=trace["fused_logit"])
        link(f"hgt:logit:{state}", f"fusion:{state}", "averaged_into")
        link(f"relnet:logit:{state}", f"fusion:{state}", "averaged_into")
        if "decision_policy" in report:
            policy = report["decision_policy"]
            threshold = float(policy["logit_threshold"])
            node(f"decision:{state}", "threshold_decision",
                 label="Bot" if trace["fused_logit"] >= threshold else "Human",
                 logit_threshold=threshold, policy_sha256=policy["sha256"])
            link(f"fusion:{state}", f"decision:{state}", "thresholded_by_dev_policy")
    node("control_edit", "source_operation", action="remove_control_sources",
         source_ids=list(control_ids))
    for sid in control_ids:
        link(f"source:{sid}", "control_edit", "same_source_budget")
    node("control_response", "measured_response",
         fused_delta_logit=cert["controls"][0]["fused_delta_logit"])
    link("control_edit", "control_response", "frozen_model_recomputed_after")
    node("comparison", "decision_dependence",
         hgt_delta_logit=cert["hgt_delta_logit"],
         relnet_delta_logit=cert["relnet_delta_logit"],
         fused_delta_logit=cert["fused_delta_logit"],
         same_budget_control_delta=cert["controls"][0]["fused_delta_logit"])
    for preceding in ("fusion:before", "fusion:after", "control_response"):
        link(preceding, "comparison", "compared_with")
    if "decision_policy" in report:
        for state in ("before", "after"):
            link(f"decision:{state}", "comparison", "decision_response_compared_with")
    graph = {"account_id": cert["account_id"], "nodes": nodes, "edges": edges,
             "claim_scope": cert["claim_scope"]}
    validate_evidence_dag(graph)
    return graph


def validate_evidence_dag(graph: dict[str, Any]) -> None:
    nodes = graph["nodes"]
    outgoing: dict[str, list[str]] = {key: [] for key in nodes}
    indegree = {key: 0 for key in nodes}
    for edge in graph["edges"]:
        source, target = edge["from"], edge["to"]
        if source not in nodes or target not in nodes:
            raise ValueError("DAG contains an edge with a missing endpoint")
        outgoing[source].append(target)
        indegree[target] += 1
    queue = deque(key for key, count in indegree.items() if count == 0)
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in outgoing[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(nodes):
        raise ValueError("source-to-model evidence graph contains a cycle")
    for state in ("before", "after"):
        hgt = nodes[f"hgt:logit:{state}"]["value"]
        relnet = nodes[f"relnet:logit:{state}"]["value"]
        fused = nodes[f"fusion:{state}"]["value"]
        if abs(fused - 0.5 * (hgt + relnet)) > 1e-6:
            raise ValueError("DAG fusion node differs from branch logits")
        if f"decision:{state}" in nodes:
            decision = nodes[f"decision:{state}"]
            expected = "Bot" if fused >= decision["logit_threshold"] else "Human"
            if decision["label"] != expected:
                raise ValueError("DAG threshold decision differs from fused logit")
    candidate_sources = [key for key, value in nodes.items()
                         if value["kind"] == "raw_source" and
                         key not in {f"source:{sid}" for sid in
                                     nodes["control_edit"]["source_ids"]}]
    for source in candidate_sources:
        reached = set()
        pending = [source]
        while pending:
            current = pending.pop()
            if current in reached:
                continue
            reached.add(current)
            pending.extend(outgoing[current])
        if not {"hgt:logit:after", "relnet:logit:after", "comparison"} <= reached:
            raise ValueError("candidate source does not reach both branch outputs")
