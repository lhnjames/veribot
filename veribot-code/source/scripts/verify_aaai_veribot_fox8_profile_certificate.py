"""Independently read profile fields and replay their frozen VeriBot effects."""

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
from train_aaai_veribot_fox8 import load_inputs, sha256_file, tensor_hash
from verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS
from verifiable_bot.aaai_fox8_raw_features import read_first_profile
from verifiable_bot.evidence_v2 import SourceRecord
from verifiable_bot.aaai_provenance import SCHEMA_VERSION, verify_code_manifest
from verifiable_bot.aaai_veribot_core import VeriBotGraphInput

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
FEATURES = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
AUDIT = ROOT / "research/certificates/aaai_veribot_fox8_all_features_audit.json"


def direct_logits(hgt, relnet, user_x, graph, edges, index):
    with torch.inference_mode():
        h = float(hgt({"user": user_x}, graph)["bot_logit"][index])
        r = float(relnet(user_x, edges)[index])
    return {"hgt_logit": h, "relnet_logit": r, "fused_logit": (h + r) / 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", nargs="?", type=Path,
                        default=ROOT / "research/certificates/aaai_veribot_fox8_profile_fact.json")
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = json.loads(args.certificate.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("certificate schema version differs")
    verify_code_manifest(ROOT, report.get("code_sha256"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["status"] != "verified_raw_to_all_named_features_and_frozen_input":
        raise ValueError("all-feature source audit is not verified")
    if (report["raw_source_sha256"] != sha256_file(RAW)
            or report["feature_file_sha256"] != sha256_file(FEATURES)
            or report["checkpoint_sha256"] != sha256_file(args.run / "model.pt")
            or audit["raw_source_sha256"] != report["raw_source_sha256"]
            or audit["checkpoint_sha256"] != report["checkpoint_sha256"]):
        raise ValueError("source, feature table or checkpoint differs")
    x, _, split, ids, graph, pack = load_inputs()
    cert = report["certificate"]
    account_id = cert["account_id"]
    index = ids.index(account_id)
    if index not in set(split["test"]):
        raise ValueError("certified account is outside test split")
    profile, line = read_first_profile(RAW, account_id)
    records = report["source_records"]
    for sid, record in records.items():
        field = next(iter(record["payload"]))
        expected_location = f"{RAW.resolve()}#line={line};tweet=0;user.{field}"
        original = SourceRecord(sid, account_id, "profile_field",
                                {field: profile[field]}, expected_location)
        if (record["raw_location"] != expected_location
                or record["sha256"] != original.sha256
                or record["payload"] != original.payload):
            raise ValueError(f"raw profile field differs: {field}")
    frame = pd.read_csv(FEATURES)
    raw_values = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
    if raw_values.shape != (len(ids), len(FEATURE_FIELDS)):
        raise ValueError("feature table shape differs")
    if any(float(report["baseline_raw_input"][field]) != float(raw_values[index, j])
           for j, field in enumerate(FEATURE_FIELDS)):
        raise ValueError("certificate fact values differ from audited feature table")
    saved = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    contract = saved["input_contract"]
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
    hgt.load_state_dict(saved["model_state"]["hgt"])
    relnet.load_state_dict(saved["model_state"]["relnet"])
    hgt.eval()
    relnet.eval()

    def replay(removed):
        values = raw_values.copy()
        for sid in removed:
            field = next(iter(records[sid]["payload"]))
            values[index, FEATURE_FIELDS.index(field)] = 0.0
            if field in ("followers_count", "friends_count"):
                followers = values[index, FEATURE_FIELDS.index("followers_count")]
                friends = values[index, FEATURE_FIELDS.index("friends_count")]
                values[index, FEATURE_FIELDS.index("ff_ratio")] = (
                    followers / max(friends + 1, 1))
        updated = torch.from_numpy(
            np.clip(scaler.transform(values), -3, 3).astype(np.float32))
        built = VeriBotGraphInput(
            x_dict={"user": updated}, edge_index_dict=graph,
            relnet_edges=pack["relnet_edges"], account_ids=tuple(ids),
            user_fact_names=tuple(FEATURE_FIELDS), source_dependencies={},
            graph_rule_sha256=cert["graph_rule_sha256"])
        return (direct_logits(hgt, relnet, updated, graph,
                              pack["relnet_edges"], index),
                tensor_hash(updated), built.fingerprint())

    candidate = tuple(cert["claim"]["source_ids"])
    control = tuple(cert["controls"][0]["source_ids"])
    if len(candidate) != len(control) or len(candidate) != 1 or set(candidate) & set(control):
        raise ValueError("candidate and control budgets differ")
    before, before_hash, before_fingerprint = replay(())
    after, _, after_fingerprint = replay(candidate)
    control_logits, _, _ = replay(control)
    if before_hash != contract["feature_sha256"]:
        raise ValueError("raw fact baseline does not match frozen model input")
    if (before_fingerprint != cert["before"]["model_input_sha256"] or
            after_fingerprint != cert["after"]["model_input_sha256"]):
        raise ValueError("edited complete model input differs")
    if (cert["claim"]["operation"] != "field" or
            cert["claim"]["recomputed_value"] !=
            records[candidate[0]]["payload"][next(iter(records[candidate[0]]["payload"]))]):
        raise ValueError("profile source claim differs")
    maximum = 0.0
    for stage, observed in (("before", before), ("after", after)):
        for key in ("hgt_logit", "relnet_logit", "fused_logit"):
            difference = abs(observed[key] - cert[stage][key])
            maximum = max(maximum, difference)
            if difference > 1e-5:
                raise ValueError(f"{stage} {key} differs from certificate")
    for branch in ("hgt", "relnet", "fused"):
        key = f"{branch}_logit"
        reported = cert["controls"][0][f"{branch}_delta_logit"]
        difference = abs(control_logits[key] - before[key] - reported)
        maximum = max(maximum, difference)
        if difference > 1e-5:
            raise ValueError(f"control {branch} differs from certificate")
    result = {"status": "verified_raw_profile_fact_and_frozen_two_branch_replay",
              "certificate_sha256": sha256_file(args.certificate),
              "account_id": account_id, "candidate_source_id": candidate[0],
              "control_source_id": control[0],
              "max_output_difference": maximum,
              "source_audit_sha256": sha256(AUDIT.read_bytes()).hexdigest()}
    destination = args.certificate.with_name(args.certificate.stem + "_verification.json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
