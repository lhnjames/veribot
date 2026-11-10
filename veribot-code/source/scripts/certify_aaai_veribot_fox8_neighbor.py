"""Trace a raw neighbor profile fact through both frozen VeriBot graph branches."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/scripts"))
sys.path.insert(0, str(ROOT / "source/hgt_import/code"))

from hgt_model import VerifiableHGT
from relnet_reimplementation import GatedRelNet
from certify_aaai_veribot_fox8 import state_hash
from train_aaai_veribot_fox8 import load_inputs, sha256_file, tensor_hash
from verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS, PROFILE_FIELDS
from verifiable_bot.aaai_fox8_raw_features import profile_features, read_first_profile
from verifiable_bot.aaai_veribot_core import (
    VeriBotDecision, VeriBotGraphInput, certify_aaai_source_response,
)
from verifiable_bot.aaai_provenance import SCHEMA_VERSION, code_manifest
from verifiable_bot.evidence_v2 import AtomicClaim, SourceRecord

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
FEATURES = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
AUDIT = ROOT / "research/certificates/aaai_veribot_fox8_all_features_audit.json"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
OUTPUT = ROOT / "research/certificates/aaai_veribot_fox8_neighbor_context.json"
FIELD = "statuses_count"
CODE_FILES = (
    "source/scripts/certify_aaai_veribot_fox8_neighbor.py",
    "source/scripts/train_aaai_veribot_fox8.py",
    "source/verifiable_bot/aaai_fox8_raw_features.py",
    "source/verifiable_bot/aaai_fox8_feature_schema.py",
    "source/verifiable_bot/aaai_fox8_native_graph.py",
    "source/verifiable_bot/aaai_veribot_core.py",
    "source/verifiable_bot/evidence_v2.py",
    "source/hgt_import/code/hgt_model.py",
    "source/hgt_import/code/relnet_reimplementation.py",
)


def select_target_and_neighbors(ids, split, graph, raw_values):
    adjacent = [set() for _ in ids]
    for edge in graph.values():
        for source, target in edge.T.tolist():
            adjacent[source].add(target)
            adjacent[target].add(source)
    eligible = []
    column = FEATURE_FIELDS.index(FIELD)
    for target_index in split["test"]:
        target_index = int(target_index)
        neighbors = sorted((ids[j], j) for j in adjacent[target_index]
                           if j != target_index and raw_values[j, column] > 0)
        if len(neighbors) >= 2:
            eligible.append((ids[target_index], target_index, neighbors[:2]))
    if not eligible:
        raise ValueError("no test account has two eligible raw graph neighbors")
    return min(eligible, key=lambda item: item[0]), len(eligible)


def bridge_record(target: str, neighbor: str, events: dict) -> SourceRecord:
    options = [event for event in events["events"]
               if {str(event["source_id"]), str(event["target_id"])} ==
               {target, neighbor}]
    if not options:
        raise ValueError("neighbor lacks an indexed native relation")
    event = min(options, key=lambda item:
                (item["relation"], str(item["source_id"]), str(item["target_id"])))
    locator = event["locators"][0]
    sid = (f"fox8:bridge:{event['relation']}:{event['source_id']}:"
           f"{event['target_id']}:{locator['line']}:{locator['tweet_index']}")
    return SourceRecord(
        sid, str(event["source_id"]), f"{event['relation']}_target_pointer",
        {"target_user_id": str(event["target_id"]),
         "tweet_id": str(locator["tweet_id"])},
        f"{RAW.resolve()}#line={locator['line']};tweet={locator['tweet_index']};{locator['field']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    x, _, split, ids, graph, pack = load_inputs()
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if (audit["status"] != "verified_raw_to_all_named_features_and_frozen_input" or
            audit["raw_source_sha256"] != sha256_file(RAW) or
            audit["scaled_input_sha256"] != tensor_hash(x)):
        raise ValueError("raw feature audit differs from frozen model input")
    frame = pd.read_csv(FEATURES)
    raw_values = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
    (target, target_index, neighbors), eligible_count = \
        select_target_and_neighbors(ids, split, graph, raw_values)
    events = json.loads(EVENTS.read_text(encoding="utf-8"))
    if events["raw_source_sha256"] != sha256_file(RAW):
        raise ValueError("native relation index differs from raw source")
    sources = {}
    profiles = {}
    source_paths = {}
    bridges = {}
    for uid, index in neighbors:
        profile, line = read_first_profile(RAW, uid)
        computed = profile_features(profile)
        expected = np.asarray([computed[name] for name in PROFILE_FIELDS], dtype=np.float32)
        if not np.allclose(expected, raw_values[index, :len(PROFILE_FIELDS)],
                           rtol=1e-6, atol=1e-5):
            raise ValueError(f"neighbor {uid} profile differs from model features")
        sid = f"fox8:{uid}:profile:{FIELD}"
        sources[sid] = SourceRecord(
            sid, uid, "profile_field", {FIELD: profile[FIELD]},
            f"{RAW.resolve()}#line={line};tweet=0;user.{FIELD}")
        profiles[sid] = (profile, index)
        source_paths[sid] = (f"user[{uid}].{FIELD}",)
        bridges[sid] = bridge_record(target, uid, events)
    checkpoint = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    contract = checkpoint["input_contract"]
    if contract["feature_sha256"] != tensor_hash(x):
        raise ValueError("checkpoint input differs")
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(contract["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(contract["scaler_scale"], dtype=np.float64)
    scaler.n_features_in_ = len(FEATURE_FIELDS)
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
    graph_rule = sha256(pack["contract"]["graph_rule"].encode()).hexdigest()

    def build_input(remaining: dict[str, SourceRecord]) -> VeriBotGraphInput:
        values = raw_values.copy()
        for sid, (profile, index) in profiles.items():
            changed = dict(profile)
            if sid not in remaining:
                changed.pop(FIELD)
            computed = profile_features(changed)
            values[index, :len(PROFILE_FIELDS)] = [computed[name] for name in PROFILE_FIELDS]
        user_x = torch.from_numpy(
            np.clip(scaler.transform(values), -3, 3).astype(np.float32))
        if len(remaining) == len(sources) and tensor_hash(user_x) != contract["feature_sha256"]:
            raise ValueError("raw neighbor facts do not rebuild the frozen input")
        return VeriBotGraphInput(
            x_dict={"user": user_x}, edge_index_dict=graph,
            relnet_edges=pack["relnet_edges"], account_ids=tuple(ids),
            user_fact_names=tuple(FEATURE_FIELDS),
            source_dependencies=source_paths, graph_rule_sha256=graph_rule)

    candidate = f"fox8:{neighbors[0][0]}:profile:{FIELD}"
    control = f"fox8:{neighbors[1][0]}:profile:{FIELD}"
    claim = AtomicClaim((candidate,), "field", sources[candidate].payload[FIELD],
                        field=FIELD)
    cert = certify_aaai_source_response(
        account_id=target, claim=claim, sources=sources,
        build_input=build_input, decision=decision,
        control_source_groups=((control,),))
    saved = json.loads((args.run / "test_predictions.json").read_text(encoding="utf-8"))
    prediction = next(item for item in saved["predictions"]
                      if item["account_id"] == target)
    if any(abs(cert["before"][name] - prediction[name]) > 1e-5
           for name in ("hgt_logit", "relnet_logit", "fused_logit")):
        raise ValueError("target baseline differs from saved test prediction")
    output = {
        "schema_version": SCHEMA_VERSION,
        "code_sha256": code_manifest(ROOT, CODE_FILES),
        "selection_rule": "lexicographically first fixed-test account with two native graph neighbors whose statuses_count is positive; lexicographically first neighbor candidate, second neighbor same-field control",
        "eligible_test_accounts": eligible_count,
        "raw_source_sha256": sha256_file(RAW),
        "event_index_sha256": sha256_file(EVENTS),
        "feature_file_sha256": sha256_file(FEATURES),
        "checkpoint_sha256": sha256_file(args.run / "model.pt"),
        "baseline_scaled_input_sha256": tensor_hash(x),
        "source_records": {sid: {"account_id": record.account_id,
                                 "kind": record.kind, "payload": dict(record.payload),
                                 "raw_location": record.raw_location,
                                 "sha256": record.sha256}
                           for sid, record in sources.items()},
        "bridge_records": {sid: {"account_id": bridge.account_id,
                                 "kind": bridge.kind, "payload": dict(bridge.payload),
                                 "raw_location": bridge.raw_location,
                                 "sha256": bridge.sha256}
                           for sid, bridge in bridges.items()},
        "source_input_paths": source_paths,
        "certificate": cert,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"target_account": target,
                      "candidate_neighbor": neighbors[0][0],
                      "control_neighbor": neighbors[1][0],
                      "eligible_test_accounts": eligible_count,
                      "hgt_delta_logit": cert["hgt_delta_logit"],
                      "relnet_delta_logit": cert["relnet_delta_logit"],
                      "fused_delta_logit": cert["fused_delta_logit"]}), flush=True)


if __name__ == "__main__":
    main()
