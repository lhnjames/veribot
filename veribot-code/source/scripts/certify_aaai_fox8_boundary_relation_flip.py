"""Certify a fixed-threshold Fox8 relation flip with a same-author control."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "source/scripts"),
                str(ROOT / "source/hgt_import/code")]

from hgt_model import VerifiableHGT
from relnet_reimplementation import GatedRelNet
from certify_aaai_veribot_fox8 import event_record, readback_sources, state_hash
from train_aaai_veribot_fox8 import load_inputs, sha256_file, tensor_hash
from verifiable_bot.aaai_fox8_native_graph import native_relation_edges
from verifiable_bot.aaai_veribot_core import (
    VeriBotDecision, VeriBotGraphInput, certify_aaai_source_response,
)
from verifiable_bot.aaai_provenance import SCHEMA_VERSION, code_manifest
from verifiable_bot.evidence_v2 import AtomicClaim, SourceRecord

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
SCREEN = ROOT / "research/certificates/aaai_veribot_fox8_boundary_relation_screen.json"
OUTPUT = ROOT / "research/certificates/aaai_veribot_fox8_boundary_relation_flip.json"
CODE_FILES = (
    "source/scripts/certify_aaai_fox8_boundary_relation_flip.py",
    "source/scripts/certify_aaai_veribot_fox8.py",
    "source/scripts/train_aaai_veribot_fox8.py",
    "source/verifiable_bot/aaai_veribot_core.py",
    "source/verifiable_bot/aaai_fox8_native_graph.py",
    "source/verifiable_bot/aaai_fox8_feature_schema.py",
    "source/verifiable_bot/evidence_v2.py",
    "source/hgt_import/code/hgt_model.py",
    "source/hgt_import/code/relnet_reimplementation.py",
)


def main() -> None:
    torch.set_num_threads(4)
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if screen["class_changing_operations"] != 1:
        raise ValueError("expected a single graph-source flip in the boundary screen")
    selected = [(row["account_id"], effect) for row in screen["accounts"]
                for effect in row["effects"] if effect["final_class_changed"]]
    if len(selected) != 1:
        raise ValueError("graph-source flip inventory differs")
    account_id, selected_effect = selected[0]
    event_index = json.loads(EVENTS.read_text(encoding="utf-8"))["events"]
    key = tuple(selected_effect["source_key"])
    matched = [event for event in event_index if
               (str(event["source_id"]), str(event["target_id"]), event["relation"]) == key]
    if len(matched) != 1 or matched[0]["locators"][0] != selected_effect["raw_locator"]:
        raise ValueError("screened raw pointer differs from source index")
    candidate = matched[0]
    alternatives = sorted(
        (event for event in event_index
         if event["relation"] == candidate["relation"]
         and str(event["source_id"]) == str(candidate["source_id"])
         and str(event["target_id"]) != str(candidate["target_id"])
         and int(event["count"]) == 1 and len(event["locators"]) == 1),
        key=lambda event: (str(event["target_id"]),
                           int(event["locators"][0]["line"]),
                           int(event["locators"][0]["tweet_index"])))
    if not alternatives:
        raise ValueError("no same-author, same-relation, one-pointer control")
    control = alternatives[0]
    records = {record.source_id: record for record in
               (event_record(candidate), event_record(control))}
    readback_sources(records)
    x, _, split, ids, graph, pack = load_inputs()
    if account_id not in {ids[int(i)] for i in split["test"]}:
        raise ValueError("selected target is outside fixed test")
    checkpoint = torch.load(RUN / "model.pt", map_location="cpu", weights_only=False)
    if checkpoint["input_contract"]["feature_sha256"] != tensor_hash(x):
        raise ValueError("frozen user input differs")
    if any(checkpoint["input_contract"]["graph_sha256"][name] != tensor_hash(edge)
           for name, edge in pack["relnet_edges"].items()):
        raise ValueError("frozen graph differs")
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3)
    relnet = GatedRelNet(x.shape[1], hidden_channels=128,
                         relation_names=tuple(pack["relnet_edges"]),
                         num_layers=2, dropout=0.3)
    hgt.load_state_dict(checkpoint["model_state"]["hgt"])
    relnet.load_state_dict(checkpoint["model_state"]["relnet"])
    decision = VeriBotDecision(
        hgt, relnet,
        hgt_sha256=state_hash(checkpoint["model_state"]["hgt"]),
        relnet_sha256=state_hash(checkpoint["model_state"]["relnet"]))
    keyed = {event_record(event).source_id:
             (str(event["source_id"]), str(event["target_id"]), event["relation"])
             for event in (candidate, control)}
    graph_hashes = pack["contract"]["graph_sha256"]

    def build_input(remaining: dict[str, SourceRecord]) -> VeriBotGraphInput:
        excluded = frozenset(keyed[sid] for sid in records if sid not in remaining)
        typed, meta = native_relation_edges(
            raw_path=RAW, event_index_path=EVENTS,
            account_ids=ids, excluded_events=excluded)
        if not excluded and any(tensor_hash(typed[name]) != graph_hashes[name]
                                for name in typed):
            raise ValueError("baseline raw graph differs from frozen graph")
        paths = {sid: (f"graph.{relation}[{source}->{target}]",
                       f"graph.rev_{relation}[{target}->{source}]")
                 for sid, (source, target, relation) in keyed.items()}
        return VeriBotGraphInput(
            x_dict={"user": x},
            edge_index_dict={("user", name, "user"): edge
                             for name, edge in typed.items()},
            relnet_edges=typed, account_ids=tuple(ids),
            user_fact_names=tuple(pack["contract"]["feature_names"]),
            source_dependencies=paths,
            graph_rule_sha256=sha256(meta["operation"].encode()).hexdigest())

    candidate_id = event_record(candidate).source_id
    control_id = event_record(control).source_id
    claim = AtomicClaim((candidate_id,), "field",
                        records[candidate_id].payload["target_user_id"],
                        field="target_user_id")
    cert = certify_aaai_source_response(
        account_id=account_id, claim=claim, sources=records,
        build_input=build_input, decision=decision,
        control_source_groups=((control_id,),))
    policy = json.loads((RUN / "decision_policy.json").read_text(encoding="utf-8"))
    threshold = float(policy["logit_threshold"])
    if (policy["checkpoint_sha256"] != sha256_file(RUN / "model.pt")
            or screen["threshold_logit"] != threshold
            or (cert["before"]["fused_logit"] >= threshold)
            == (cert["after"]["fused_logit"] >= threshold)):
        raise ValueError("candidate does not replay the saved final-class flip")
    for state in ("before", "after"):
        for branch in ("hgt", "relnet", "fused"):
            if abs(cert[state][f"{branch}_logit"]
                   - selected_effect[state][f"{branch}_logit"]) > 1e-5:
                raise ValueError("boundary scan and full certificate disagree")
    output = {
        "schema_version": SCHEMA_VERSION,
        "code_sha256": code_manifest(ROOT, CODE_FILES),
        "selection_rule": "sole final-class flip among the single-pointer relations of 20 preselected boundary accounts; lexicographically first other same-author, same-relation, one-pointer edge as control",
        "boundary_screen_sha256": sha256_file(SCREEN),
        "raw_source_sha256": sha256_file(RAW),
        "event_index_sha256": sha256_file(EVENTS),
        "checkpoint_sha256": sha256_file(RUN / "model.pt"),
        "decision_policy_sha256": sha256_file(RUN / "decision_policy.json"),
        "threshold_logit": threshold,
        "final_class_before": int(cert["before"]["fused_logit"] >= threshold),
        "final_class_after": int(cert["after"]["fused_logit"] >= threshold),
        "selected_source_records": {
            sid: {"account_id": row.account_id, "kind": row.kind,
                  "payload": dict(row.payload), "raw_location": row.raw_location,
                  "sha256": row.sha256}
            for sid, row in records.items()},
        "source_readback": "verified_from_raw_gzip",
        "certificate": cert,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"account_id": account_id,
                      "candidate_source_id": candidate_id,
                      "control_source_id": control_id,
                      "before": cert["before"]["fused_logit"],
                      "after": cert["after"]["fused_logit"],
                      "control_delta": cert["controls"][0]["fused_delta_logit"],
                      "output": str(OUTPUT)}), flush=True)


if __name__ == "__main__":
    main()
