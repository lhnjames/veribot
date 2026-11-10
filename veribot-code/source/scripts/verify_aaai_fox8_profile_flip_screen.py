"""Re-read raw Fox8 profiles and replay the fixed-threshold source screen."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "source/scripts"),
                str(ROOT / "source/hgt_import/code")]

from hgt_model import VerifiableHGT
from relnet_reimplementation import GatedRelNet
from train_aaai_veribot_fox8 import load_inputs, sha256_file, tensor_hash
from verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS, PROFILE_FIELDS
from verifiable_bot.aaai_fox8_raw_features import RAW_PROFILE_FIELDS, profile_features
from verifiable_bot.evidence_v2 import SourceRecord

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
FEATURES = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
SCREEN = ROOT / "research/certificates/aaai_veribot_fox8_fixed_threshold_profile_screen.json"
OUTPUT = ROOT / "research/certificates/aaai_veribot_fox8_fixed_threshold_profile_screen_verification.json"


def close(actual: float, expected: float, tolerance: float = 1e-5) -> float:
    difference = abs(actual - expected)
    if difference > tolerance:
        raise ValueError(f"frozen output mismatch: {actual} vs {expected}")
    return difference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", nargs="?", type=Path, default=SCREEN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    policy = json.loads((RUN / "decision_policy.json").read_text(encoding="utf-8"))
    saved = json.loads((RUN / "test_predictions.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(RUN / "model.pt", map_location="cpu", weights_only=False)
    for key, path in (("raw_source_sha256", RAW),
                      ("feature_file_sha256", FEATURES),
                      ("checkpoint_sha256", RUN / "model.pt"),
                      ("policy_sha256", RUN / "decision_policy.json"),
                      ("predictions_sha256", RUN / "test_predictions.json")):
        if screen[key] != sha256_file(path):
            raise ValueError(f"input file fingerprint differs: {key}")
    threshold = float(policy["logit_threshold"])
    if screen["threshold_logit"] != threshold:
        raise ValueError("decision threshold differs")
    by_id = {row["account_id"]: row for row in saved["predictions"]}
    selected = sorted(by_id, key=lambda uid:
                      (abs(float(by_id[uid]["fused_logit"]) - threshold), uid))[
                          :screen["top_n"]]
    if selected != [row["account_id"] for row in screen["accounts"]]:
        raise ValueError("boundary-account selection differs")
    targets = set(selected)
    profiles = {}
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid in targets and uid not in profiles:
                profiles[uid] = (row["user_tweets"][0]["user"], line_no)
    if set(profiles) != targets:
        raise ValueError("raw account inventory differs")

    x, _, split, ids, graph, pack = load_inputs()
    if targets - {ids[int(i)] for i in split["test"]}:
        raise ValueError("screen includes a non-test account")
    if checkpoint["input_contract"]["feature_sha256"] != tensor_hash(x):
        raise ValueError("checkpoint input fingerprint differs")
    frame = pd.read_csv(FEATURES)
    if frame["user_id"].astype(str).tolist() != ids or \
            frame.columns.tolist() != ["user_id", *FEATURE_FIELDS]:
        raise ValueError("feature table differs")
    raw = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
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

    def forward(features: torch.Tensor, index: int) -> dict[str, float]:
        with torch.inference_mode():
            h = float(hgt({"user": features}, graph)["bot_logit"][index])
            r = float(relnet(features, pack["relnet_edges"])[index])
        return {"hgt_logit": h, "relnet_logit": r,
                "fused_logit": float(0.5 * (torch.tensor(h, dtype=torch.float32)
                                            + torch.tensor(r, dtype=torch.float32)))}

    max_difference = 0.0
    operations = 0
    flips = []
    for entry in screen["accounts"]:
        uid = entry["account_id"]
        index = ids.index(uid)
        profile, line_no = profiles[uid]
        if str(profile.get("id_str", profile.get("id"))) != uid or \
                line_no != entry["raw_profile_line"]:
            raise ValueError(f"raw profile locator differs: {uid}")
        facts = profile_features(profile)
        expected_profile = np.asarray([facts[name] for name in PROFILE_FIELDS],
                                      dtype=np.float32)
        if not np.array_equal(expected_profile, raw[index, :len(PROFILE_FIELDS)]):
            raise ValueError(f"raw profile facts differ: {uid}")
        before = forward(x, index)
        if int(entry["label"]) != int(by_id[uid]["label"]):
            raise ValueError("saved test label differs")
        for key in before:
            max_difference = max(max_difference, close(before[key], entry["before"][key]),
                                 close(before[key], float(by_id[uid][key])))
        max_difference = max(max_difference, close(
            before["fused_logit"] - threshold, entry["margin_logit"]))
        expected_fields = [name for name in RAW_PROFILE_FIELDS if name in profile]
        if expected_fields != [row["field"] for row in entry["fields"]]:
            raise ValueError(f"field inventory differs: {uid}")
        for item in entry["fields"]:
            field = item["field"]
            record = SourceRecord(
                f"fox8:{uid}:profile:{field}", uid, "profile_field",
                {field: profile[field]},
                f"{RAW.resolve()}#line={line_no};tweet=0;user.{field}")
            if (item["source_id"] != record.source_id or
                    item["source_sha256"] != record.sha256 or
                    item["raw_location"] != record.raw_location or
                    item["raw_value"] != profile[field]):
                raise ValueError(f"source record differs: {uid}/{field}")
            reduced = dict(profile)
            reduced.pop(field)
            recomputed = profile_features(reduced)
            changed = [name for name in PROFILE_FIELDS
                       if np.float32(recomputed[name]) != np.float32(facts[name])]
            if changed != item["changed_model_facts"]:
                raise ValueError(f"dependent facts differ: {uid}/{field}")
            if changed:
                edited_raw = raw[index:index + 1].copy()
                edited_raw[0, :len(PROFILE_FIELDS)] = [recomputed[name]
                                                       for name in PROFILE_FIELDS]
                edited = x.clone()
                edited[index:index + 1] = torch.from_numpy(np.clip(
                    scaler.transform(edited_raw), -3, 3).astype(np.float32))
            else:
                edited = x
            if tensor_hash(edited) != item["edited_input_sha256"]:
                raise ValueError(f"edited input fingerprint differs: {uid}/{field}")
            after = forward(edited, index) if changed else before
            for key in after:
                max_difference = max(max_difference, close(after[key], item["after"][key]))
            max_difference = max(max_difference, close(
                after["fused_logit"] - before["fused_logit"],
                item["fused_delta_logit"]))
            class_changed = ((after["fused_logit"] >= threshold)
                             != (before["fused_logit"] >= threshold))
            if class_changed != item["class_changed"]:
                raise ValueError(f"final class response differs: {uid}/{field}")
            if class_changed:
                flips.append({"account_id": uid, "field": field,
                              "fused_delta_logit": item["fused_delta_logit"]})
            operations += 1
        print(json.dumps({"verified_account_id": uid,
                          "verified_operations": operations}), flush=True)
    if operations != screen["single_source_operations"] or \
            flips != screen["single_source_flips"]:
        raise ValueError("population summary differs")
    result = {"status": "verified_raw_sources_inputs_frozen_outputs_and_final_classes",
              "screen_sha256": sha256_file(args.screen),
              "checkpoint_sha256": sha256_file(RUN / "model.pt"),
              "verified_accounts": len(selected),
              "verified_single_source_operations": operations,
              "verified_single_source_flips": len(flips),
              "max_output_difference": max_difference}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
