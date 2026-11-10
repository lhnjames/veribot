"""Enumerate all one-field raw-profile alternatives for the fixed boundary case."""

from __future__ import annotations

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
from verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS, PROFILE_FIELDS
from verifiable_bot.aaai_fox8_raw_features import (
    RAW_PROFILE_FIELDS, profile_features, read_first_profile,
)
from verifiable_bot.evidence_v2 import SourceRecord

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
FEATURES = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
SCAN = ROOT / "research/certificates/aaai_veribot_fox8_profile_response_population.json"
AUDIT = ROOT / "research/certificates/aaai_veribot_fox8_all_features_audit.json"
OUTPUT = ROOT / "research/certificates/aaai_veribot_fox8_profile_one_field_alternatives.json"


def main() -> None:
    torch.set_num_threads(4)
    population = json.loads(SCAN.read_text(encoding="utf-8"))
    if population["summary"]["candidate_class_changes"] != 1:
        raise ValueError("expected exactly one boundary account from the fixed scan")
    selected = [row["account_id"] for row in population["accounts"]
                if row["effects"]["statuses_count"]["fused_class_changed"]]
    if len(selected) != 1:
        raise ValueError("population scan and boundary account count differ")
    account_id = selected[0]
    x, _, split, ids, graph, pack = load_inputs()
    target = ids.index(account_id)
    if target not in set(split["test"]):
        raise ValueError("boundary account is outside fixed test split")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if (audit["scaled_input_sha256"] != tensor_hash(x)
            or audit["checkpoint_sha256"] != sha256_file(RUN / "model.pt")):
        raise ValueError("source feature audit differs from frozen model")
    frame = pd.read_csv(FEATURES)
    raw = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
    profile, line = read_first_profile(RAW, account_id)
    baseline_profile = profile_features(profile)
    if not np.array_equal(
        np.asarray([baseline_profile[name] for name in PROFILE_FIELDS], dtype=np.float32),
        raw[target, :len(PROFILE_FIELDS)]):
        raise ValueError("raw first profile differs from model facts")
    checkpoint = torch.load(RUN / "model.pt", map_location="cpu", weights_only=False)
    contract = checkpoint["input_contract"]
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
    hgt.eval()
    relnet.eval()

    def forward(user_x):
        with torch.inference_mode():
            a = float(hgt({"user": user_x}, graph)["bot_logit"][target])
            b = float(relnet(user_x, pack["relnet_edges"])[target])
        return a, b, (a + b) / 2

    baseline = forward(x)
    results = []
    for field in RAW_PROFILE_FIELDS:
        if field not in profile:
            continue
        sid = f"fox8:{account_id}:profile:{field}"
        record = SourceRecord(sid, account_id, "profile_field", {field: profile[field]},
                              f"{RAW.resolve()}#line={line};tweet=0;user.{field}")
        changed_profile = dict(profile)
        changed_profile.pop(field)
        edited_facts = profile_features(changed_profile)
        changed_columns = [name for name in PROFILE_FIELDS
                           if np.float32(edited_facts[name]) != np.float32(baseline_profile[name])]
        if not changed_columns:
            results.append({"field": field, "source_id": sid,
                            "source_sha256": record.sha256,
                            "raw_location": record.raw_location,
                            "changed_model_facts": [],
                            "fused_delta_logit": 0.0,
                            "class_changed": False})
            continue
        edited_raw = raw[target:target + 1].copy()
        edited_raw[0, :len(PROFILE_FIELDS)] = [edited_facts[name] for name in PROFILE_FIELDS]
        edited_x = x.clone()
        edited_x[target:target + 1] = torch.from_numpy(
            np.clip(scaler.transform(edited_raw), -3, 3).astype(np.float32))
        observed = forward(edited_x)
        results.append({
            "field": field, "source_id": sid,
            "source_sha256": record.sha256,
            "raw_location": record.raw_location,
            "raw_value": profile[field],
            "changed_model_facts": changed_columns,
            "edited_input_sha256": tensor_hash(edited_x),
            "hgt_delta_logit": observed[0] - baseline[0],
            "relnet_delta_logit": observed[1] - baseline[1],
            "fused_delta_logit": observed[2] - baseline[2],
            "class_changed": (baseline[2] >= 0) != (observed[2] >= 0),
        })
    flipping = [row for row in results if row["class_changed"]]
    if not any(row["field"] == "statuses_count" for row in flipping):
        raise ValueError("full one-field enumeration did not replay the scan's class flip")
    output = {
        "selection_rule": "the sole statuses_count class-changing account in the fixed 455-account scan",
        "account_id": account_id,
        "raw_profile_line": line,
        "baseline": {"hgt_logit": baseline[0], "relnet_logit": baseline[1],
                     "fused_logit": baseline[2]},
        "checkpoint_sha256": sha256_file(RUN / "model.pt"),
        "population_scan_sha256": sha256_file(SCAN),
        "raw_source_sha256": sha256_file(RAW),
        "evaluated_present_raw_fields": len(results),
        "single_field_class_changing_alternatives": [row["field"] for row in flipping],
        "fields": results,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"account_id": account_id,
                      "evaluated_present_raw_fields": len(results),
                      "single_field_class_changing_alternatives":
                          output["single_field_class_changing_alternatives"]}), flush=True)


if __name__ == "__main__":
    main()
