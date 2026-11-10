"""Trace an original Fox8 profile fact through both frozen VeriBot branches."""

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
from verifiable_bot.evidence_v2 import AtomicClaim, SourceRecord
from verifiable_bot.aaai_provenance import SCHEMA_VERSION, code_manifest

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
FEATURES = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
DEFAULT_FIELDS = ("statuses_count", "followers_count")
CODE_FILES = (
    "source/scripts/certify_aaai_veribot_fox8_profile.py",
    "source/scripts/train_aaai_veribot_fox8.py",
    "source/verifiable_bot/aaai_veribot_core.py",
    "source/verifiable_bot/aaai_fox8_raw_features.py",
    "source/verifiable_bot/aaai_fox8_native_graph.py",
    "source/verifiable_bot/aaai_fox8_feature_schema.py",
    "source/verifiable_bot/evidence_v2.py",
    "source/hgt_import/code/hgt_model.py",
    "source/hgt_import/code/relnet_reimplementation.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--account-id", type=str)
    parser.add_argument("--candidate-field", type=str, default=DEFAULT_FIELDS[0])
    parser.add_argument("--control-field", type=str, default=DEFAULT_FIELDS[1])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/certificates/aaai_veribot_fox8_profile_fact.json")
    args = parser.parse_args()
    fields = (args.candidate_field, args.control_field)
    if len(set(fields)) != 2 or any(field not in PROFILE_FIELDS for field in fields):
        raise ValueError("candidate and control need distinct named profile facts")
    torch.set_num_threads(4)
    x, _, split, ids, graph, pack = load_inputs()
    frame = pd.read_csv(FEATURES)
    raw_values = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
    test = {ids[int(i)] for i in split["test"]}
    eligible = [uid for uid in sorted(test)
                if all(float(raw_values[ids.index(uid), FEATURE_FIELDS.index(field)]) > 0
                       for field in fields)]
    if not eligible:
        raise ValueError("no test account has both selected positive raw fields")
    if args.account_id is not None and args.account_id not in eligible:
        raise ValueError("requested account is outside the eligible fixed test set")
    account_id = args.account_id or eligible[0]
    target = ids.index(account_id)
    profile, line = read_first_profile(RAW, account_id)
    if any(field not in profile for field in fields):
        raise ValueError("selected raw profile lacks a cited field")
    if not np.allclose(
        np.asarray([profile_features(profile)[name] for name in PROFILE_FIELDS],
                   dtype=np.float32),
        raw_values[target, :len(PROFILE_FIELDS)], rtol=1e-6, atol=1e-5):
        raise ValueError("profile readback differs from training features")
    sources = {}
    for field in fields:
        sid = f"fox8:{account_id}:profile:{field}"
        sources[sid] = SourceRecord(
            sid, account_id, "profile_field", {field: profile[field]},
            f"{RAW.resolve()}#line={line};tweet=0;user.{field}")
    checkpoint = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    contract = checkpoint["input_contract"]
    if contract["feature_sha256"] != tensor_hash(x):
        raise ValueError("checkpoint user inputs differ")
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
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(contract["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(contract["scaler_scale"], dtype=np.float64)
    scaler.n_features_in_ = len(FEATURE_FIELDS)
    graph_rule = sha256(pack["contract"]["graph_rule"].encode()).hexdigest()
    source_paths = {
        sid: tuple(f"user[{account_id}].{name}" for name in
                   ((field, "ff_ratio") if field in ("followers_count", "friends_count")
                    else (field,)))
        for sid, field in ((sid, next(iter(source.payload)))
                           for sid, source in sources.items())}

    def build_input(remaining: dict[str, SourceRecord]) -> VeriBotGraphInput:
        changed_profile = dict(profile)
        for sid in sources:
            if sid not in remaining:
                changed_profile.pop(next(iter(sources[sid].payload)))
        updated = raw_values.copy()
        computed = profile_features(changed_profile)
        updated[target, :len(PROFILE_FIELDS)] = [computed[name] for name in PROFILE_FIELDS]
        user_x = torch.from_numpy(
            np.clip(scaler.transform(updated), -3, 3).astype(np.float32))
        if set(remaining) == set(sources) and tensor_hash(user_x) != contract["feature_sha256"]:
            raise ValueError("baseline raw profile input differs from frozen feature tensor")
        return VeriBotGraphInput(
            x_dict={"user": user_x}, edge_index_dict=graph,
            relnet_edges=pack["relnet_edges"], account_ids=tuple(ids),
            user_fact_names=tuple(FEATURE_FIELDS),
            source_dependencies=source_paths,
            graph_rule_sha256=graph_rule)

    candidate_id = f"fox8:{account_id}:profile:{fields[0]}"
    control_id = f"fox8:{account_id}:profile:{fields[1]}"
    claim = AtomicClaim((candidate_id,), "field",
                        sources[candidate_id].payload[fields[0]], field=fields[0])
    certificate = certify_aaai_source_response(
        account_id=account_id, claim=claim, sources=sources,
        build_input=build_input, decision=decision,
        control_source_groups=((control_id,),))
    saved = json.loads((args.run / "test_predictions.json").read_text(encoding="utf-8"))
    prediction = next(row for row in saved["predictions"]
                      if row["account_id"] == account_id)
    if any(abs(certificate["before"][name] - prediction[name]) > 1e-5
           for name in ("hgt_logit", "relnet_logit", "fused_logit")):
        raise ValueError("profile certificate does not replay saved prediction")
    output = {
        "schema_version": SCHEMA_VERSION,
        "code_sha256": code_manifest(ROOT, CODE_FILES),
        "selection_rule": (
            "explicit account selected from the complete fixed-test profile-response scan and one-field alternative audit"
            if args.account_id else
            "lexicographically first test account with positive statuses_count and followers_count; first fact candidate, second control"),
        "population_scan_sha256": (
            sha256_file(ROOT / "research/certificates/aaai_veribot_fox8_profile_response_population.json")
            if args.account_id else None),
        "alternative_audit_sha256": (
            sha256_file(ROOT / "research/certificates/aaai_veribot_fox8_profile_one_field_alternatives.json")
            if args.account_id and fields[0] != DEFAULT_FIELDS[0] else None),
        "raw_source_sha256": sha256_file(RAW),
        "feature_file_sha256": sha256_file(FEATURES),
        "checkpoint_sha256": sha256_file(args.run / "model.pt"),
        "source_records": {
            sid: {"account_id": record.account_id, "kind": record.kind,
                  "payload": dict(record.payload), "raw_location": record.raw_location,
                  "sha256": record.sha256}
            for sid, record in sources.items()},
        "baseline_raw_input": {name: float(raw_values[target, j])
                               for j, name in enumerate(FEATURE_FIELDS)},
        "baseline_scaled_input_sha256": tensor_hash(x),
        "certificate": certificate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"account_id": account_id,
                      "hgt_delta_logit": certificate["hgt_delta_logit"],
                      "relnet_delta_logit": certificate["relnet_delta_logit"],
                      "fused_delta_logit": certificate["fused_delta_logit"],
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
