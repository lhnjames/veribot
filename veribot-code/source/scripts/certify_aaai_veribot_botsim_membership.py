"""Trace one raw BotSim subreddit membership through frozen HGT and RelNet."""

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
from certify_aaai_veribot_fox8 import state_hash
from train_aaai_veribot_botsim import load_inputs, USERS, FEATURES
from train_aaai_veribot_fox8 import sha256_file, tensor_hash
from verifiable_bot.aaai_botsim_native_graph import build_native_graph, read_memberships
from verifiable_bot.aaai_dense_relnet import DenseGatedRelNet
from verifiable_bot.aaai_veribot_core import (
    VeriBotDecision, VeriBotGraphInput, certify_aaai_source_response,
)
from verifiable_bot.evidence_v2 import AtomicClaim, SourceRecord
from verifiable_bot.aaai_provenance import SCHEMA_VERSION, code_manifest

RUN = ROOT / "output/aaai_veribot_botsim/seed0"
AUDIT = ROOT / "research/certificates/aaai_veribot_botsim_all_features_audit.json"
CODE_FILES = (
    "source/scripts/certify_aaai_veribot_botsim_membership.py",
    "source/scripts/train_aaai_veribot_botsim.py",
    "source/verifiable_bot/aaai_veribot_core.py",
    "source/verifiable_bot/aaai_botsim_raw_features.py",
    "source/verifiable_bot/aaai_botsim_native_graph.py",
    "source/verifiable_bot/aaai_dense_relnet.py",
    "source/verifiable_bot/evidence_v2.py",
    "source/hgt_import/code/hgt_model.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/certificates/aaai_veribot_botsim_membership.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    x_dict, _, split, ids, graph, relnet, metadata, contract = load_inputs()
    audited = json.loads(AUDIT.read_text(encoding="utf-8"))
    if (audited["status"] != "verified_raw_to_all_named_features_and_frozen_input"
            or audited["user_input_sha256"] != tensor_hash(x_dict["user"])):
        raise ValueError("raw-to-feature audit does not match frozen BotSim input")
    raw_ids, subreddits, locations = read_memberships(USERS)
    if raw_ids != ids or subreddits != contract["graph_meta"]["subreddit_names"]:
        raise ValueError("raw membership inventory differs")
    test_ids = {ids[int(i)] for i in split["test"]}
    incident: dict[str, list[str]] = {}
    for uid, sub in locations:
        if uid in test_ids:
            incident.setdefault(uid, []).append(sub)
    choices = [(uid, sorted(set(subs))) for uid, subs in incident.items()
               if len(set(subs)) >= 2]
    if not choices:
        raise ValueError("no fixed test account has two membership sources")
    account_id, pair = sorted(choices)[0]
    target = ids.index(account_id)
    sources = {}
    for sub in pair[:2]:
        sid = f"botsim:{account_id}:subreddit:{sub}"
        sources[sid] = SourceRecord(
            sid, account_id, "subreddit_membership", {"subreddit": sub},
            locations[account_id, sub]["raw_location"])
    checkpoint = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    if checkpoint["input_contract"]["user_input_sha256"] != tensor_hash(x_dict["user"]):
        raise ValueError("checkpoint user features differ")
    hgt = VerifiableHGT(metadata, {name: values.shape[1]
                                   for name, values in x_dict.items()},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3)
    rel = DenseGatedRelNet(x_dict["user"].shape[1], hidden_channels=128,
                           relation_names=tuple(relnet), num_layers=2, dropout=0.3)
    hgt.load_state_dict(checkpoint["model_state"]["hgt"])
    rel.load_state_dict(checkpoint["model_state"]["relnet"])
    decision = VeriBotDecision(
        hgt, rel, hgt_sha256=state_hash(checkpoint["model_state"]["hgt"]),
        relnet_sha256=state_hash(checkpoint["model_state"]["relnet"]))
    frame = pd.read_csv(FEATURES)
    raw_features = frame.to_numpy(dtype=np.float32)
    feature_names = tuple(frame.columns)
    n_sub_col = feature_names.index("n_subreddits")
    if raw_features[target, n_sub_col] != len(incident[account_id]):
        raise ValueError("raw membership count differs from model fact")
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(contract["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(contract["scaler_scale"], dtype=np.float64)
    scaler.n_features_in_ = len(feature_names)
    source_keys = {sid: (account_id, record.payload["subreddit"])
                   for sid, record in sources.items()}
    rule_sha = sha256(contract["graph_meta"]["operation"].encode()).hexdigest()

    def build_input(remaining: dict[str, SourceRecord]) -> VeriBotGraphInput:
        excluded = frozenset(source_keys[sid] for sid in sources
                             if sid not in remaining)
        rebuilt_graph, rebuilt_rel, meta = build_native_graph(USERS, ids, excluded)
        updated = raw_features.copy()
        updated[target, n_sub_col] -= len(excluded)
        user_x = torch.from_numpy(
            np.clip(scaler.transform(updated), -3, 3).astype(np.float32))
        if not excluded:
            if tensor_hash(user_x) != contract["user_input_sha256"]:
                raise ValueError("baseline feature tensor differs from training")
            if any(tensor_hash(rebuilt_graph[key]) != contract["hgt_graph_sha256"][str(key)]
                   for key in rebuilt_graph):
                raise ValueError("baseline HGT graph differs from training")
            if any(tensor_hash(rebuilt_rel[name]) != contract["relnet_operator_sha256"][name]
                   for name in rebuilt_rel):
                raise ValueError("baseline RelNet projection differs from training")
        paths = {
            sid: (f"user[{account_id}].n_subreddits",
                  f"HGT.user_subreddit[{account_id},{sub}]",
                  f"RelNet.shared_subreddit[{account_id},*]")
            for sid, (_, sub) in source_keys.items()
        }
        return VeriBotGraphInput(
            x_dict={"user": user_x, "subreddit": x_dict["subreddit"]},
            edge_index_dict=rebuilt_graph, relnet_edges=rebuilt_rel,
            account_ids=tuple(ids), user_fact_names=feature_names,
            source_dependencies=paths, graph_rule_sha256=rule_sha)

    candidate_id, control_id = tuple(sources)
    claim = AtomicClaim((candidate_id,), "field",
                        sources[candidate_id].payload["subreddit"],
                        field="subreddit")
    cert = certify_aaai_source_response(
        account_id=account_id, claim=claim, sources=sources,
        build_input=build_input, decision=decision,
        control_source_groups=((control_id,),))
    predictions = json.loads((args.run / "test_predictions.json")
                             .read_text(encoding="utf-8"))
    expected = next(row for row in predictions["predictions"]
                    if row["account_id"] == account_id)
    if any(abs(cert["before"][field] - expected[field]) > 1e-5
           for field in ("hgt_logit", "relnet_logit", "fused_logit")):
        raise ValueError("certificate does not replay saved test decision")
    output = {
        "schema_version": SCHEMA_VERSION,
        "code_sha256": code_manifest(ROOT, CODE_FILES),
        "selection_rule": "lexicographically first fixed-test account with at least two raw subreddit memberships; first and second names are candidate and equal-count control",
        "raw_users_sha256": sha256_file(USERS),
        "feature_file_sha256": sha256_file(FEATURES),
        "checkpoint_sha256": sha256_file(args.run / "model.pt"),
        "source_audit_sha256": sha256_file(AUDIT),
        "source_records": {
            sid: {"account_id": row.account_id, "kind": row.kind,
                  "payload": dict(row.payload), "raw_location": row.raw_location,
                  "sha256": row.sha256}
            for sid, row in sources.items()},
        "baseline_raw_n_subreddits": float(raw_features[target, n_sub_col]),
        "certificate": cert,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"account_id": account_id,
                      "candidate_source_id": candidate_id,
                      "control_source_id": control_id,
                      "hgt_delta_logit": cert["hgt_delta_logit"],
                      "relnet_delta_logit": cert["relnet_delta_logit"],
                      "fused_delta_logit": cert["fused_delta_logit"]}), flush=True)


if __name__ == "__main__":
    main()
