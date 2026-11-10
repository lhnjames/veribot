"""Measure frozen two-branch responses to two raw profile facts on fixed test accounts."""

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
from verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS

FEATURES = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
AUDIT = ROOT / "research/certificates/aaai_veribot_fox8_all_features_audit.json"
OUTPUT = ROOT / "research/certificates/aaai_veribot_fox8_profile_response_population.json"


def logits(hgt, relnet, x, graph, edges, target):
    with torch.inference_mode():
        h = float(hgt({"user": x}, graph)["bot_logit"][target])
        r = float(relnet(x, edges)[target])
    return h, r, (h + r) / 2


def main() -> None:
    torch.set_num_threads(4)
    x, _, split, ids, graph, pack = load_inputs()
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    saved = torch.load(RUN / "model.pt", map_location="cpu", weights_only=False)
    if (audit["status"] != "verified_raw_to_all_named_features_and_frozen_input"
            or audit["checkpoint_sha256"] != sha256_file(RUN / "model.pt")
            or audit["scaled_input_sha256"] != tensor_hash(x)):
        raise ValueError("all-feature raw source audit does not match this model")
    frame = pd.read_csv(FEATURES)
    if list(frame.columns) != ["user_id", *FEATURE_FIELDS] or \
            frame["user_id"].astype(str).tolist() != ids:
        raise ValueError("feature account order differs")
    values = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
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
    with torch.inference_mode():
        baseline_h = hgt({"user": x}, graph)["bot_logit"].numpy()
        baseline_r = relnet(x, pack["relnet_edges"]).numpy()
    predictions = json.loads((RUN / "test_predictions.json").read_text(encoding="utf-8"))
    by_id = {row["account_id"]: row for row in predictions["predictions"]}
    if any(abs(float(baseline_h[i]) - by_id[ids[int(i)]]["hgt_logit"]) > 1e-5 or
           abs(float(baseline_r[i]) - by_id[ids[int(i)]]["relnet_logit"]) > 1e-5
           for i in split["test"]):
        raise ValueError("saved test predictions do not replay")
    status_col = FEATURE_FIELDS.index("statuses_count")
    follower_col = FEATURE_FIELDS.index("followers_count")
    ratio_col = FEATURE_FIELDS.index("ff_ratio")
    results = []
    skipped = {"zero_statuses": 0, "zero_followers": 0}
    for position, index in enumerate(map(int, split["test"]), 1):
        status = float(values[index, status_col])
        followers = float(values[index, follower_col])
        if status <= 0 or followers <= 0:
            skipped["zero_statuses"] += int(status <= 0)
            skipped["zero_followers"] += int(followers <= 0)
            continue
        before = (float(baseline_h[index]), float(baseline_r[index]),
                  float((baseline_h[index] + baseline_r[index]) / 2))
        effects = {}
        for field in ("statuses_count", "followers_count"):
            changed_raw = values[index:index + 1].copy()
            column = FEATURE_FIELDS.index(field)
            changed_raw[0, column] = 0
            if field == "followers_count":
                changed_raw[0, ratio_col] = 0
            edited_x = x.clone()
            edited_x[index:index + 1] = torch.from_numpy(
                np.clip(scaler.transform(changed_raw), -3, 3).astype(np.float32))
            observed = logits(hgt, relnet, edited_x, graph,
                              pack["relnet_edges"], index)
            effects[field] = {
                "hgt_delta_logit": observed[0] - before[0],
                "relnet_delta_logit": observed[1] - before[1],
                "fused_delta_logit": observed[2] - before[2],
                "fused_class_changed": (before[2] >= 0) != (observed[2] >= 0),
                "edited_feature_sha256": tensor_hash(edited_x),
            }
        uid = ids[index]
        results.append({
            "account_id": uid, "raw_profile_line": audit["first_record_locations"][uid],
            "statuses_count": status, "followers_count": followers,
            "baseline_hgt_logit": before[0],
            "baseline_relnet_logit": before[1],
            "baseline_fused_logit": before[2],
            "effects": effects,
        })
        if position % 50 == 0:
            print(json.dumps({"processed_test_position": position,
                              "eligible_so_far": len(results)}), flush=True)
    if not results:
        raise ValueError("no eligible raw profile fact edits")
    candidate = np.asarray([row["effects"]["statuses_count"]["fused_delta_logit"]
                            for row in results])
    control = np.asarray([row["effects"]["followers_count"]["fused_delta_logit"]
                          for row in results])
    summary = {
        "fixed_test_unique_accounts": len(split["test"]),
        "eligible_positive_both_fields": len(results),
        "skipped": skipped,
        "candidate_nonzero_fused_response": int(np.count_nonzero(np.abs(candidate) > 1e-6)),
        "control_nonzero_fused_response": int(np.count_nonzero(np.abs(control) > 1e-6)),
        "candidate_class_changes": sum(row["effects"]["statuses_count"]["fused_class_changed"]
                                       for row in results),
        "control_class_changes": sum(row["effects"]["followers_count"]["fused_class_changed"]
                                     for row in results),
        "both_single_field_edits_change_class": sum(
            row["effects"]["statuses_count"]["fused_class_changed"] and
            row["effects"]["followers_count"]["fused_class_changed"]
            for row in results),
        "candidate_larger_absolute_delta_than_control": int(
            np.count_nonzero(np.abs(candidate) > np.abs(control))),
        "candidate_median_absolute_delta": float(np.median(np.abs(candidate))),
        "control_median_absolute_delta": float(np.median(np.abs(control))),
        "candidate_max_absolute_delta": float(np.max(np.abs(candidate))),
        "control_max_absolute_delta": float(np.max(np.abs(control))),
    }
    output = {"selection_rule": "all fixed test accounts with positive raw statuses_count and followers_count",
              "source_audit_sha256": sha256_file(AUDIT),
              "checkpoint_sha256": sha256_file(RUN / "model.pt"),
              "feature_file_sha256": sha256_file(FEATURES),
              "operation": "remove exactly one raw first-profile field; recompute derived ff_ratio when followers_count is removed; graph unchanged",
              "summary": summary, "accounts": results}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
