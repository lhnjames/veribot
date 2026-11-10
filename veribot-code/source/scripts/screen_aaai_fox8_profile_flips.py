"""Screen raw Fox8 profile-field deletions against the frozen final decision.

Accounts are chosen by baseline distance to the saved dev-only threshold,
before any source edits are evaluated. This is an exploratory boundary screen,
not a test-population estimate of explanation quality.
"""

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
OUTPUT = ROOT / "research/certificates/aaai_veribot_fox8_fixed_threshold_profile_screen.json"


def first_profiles(account_ids: set[str]) -> dict[str, tuple[dict, int]]:
    found: dict[str, tuple[dict, int]] = {}
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid in account_ids and uid not in found:
                profile = row["user_tweets"][0]["user"]
                if str(profile.get("id_str", profile.get("id"))) != uid:
                    raise ValueError(f"profile identity mismatch for {uid}")
                found[uid] = (profile, line_no)
    if set(found) != account_ids:
        raise ValueError(f"missing {len(account_ids - set(found))} selected profiles")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.top_n <= 0:
        raise ValueError("top-n must be positive")
    torch.set_num_threads(4)
    x, _, split, ids, graph, pack = load_inputs()
    checkpoint_path = RUN / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy = json.loads((RUN / "decision_policy.json").read_text(encoding="utf-8"))
    predictions = json.loads((RUN / "test_predictions.json").read_text(encoding="utf-8"))
    if (policy["checkpoint_sha256"] != sha256_file(checkpoint_path)
            or policy["saved_predictions_sha256"] != sha256_file(RUN / "test_predictions.json")
            or checkpoint["input_contract"]["feature_sha256"] != tensor_hash(x)):
        raise ValueError("saved model, policy, predictions, and inputs disagree")
    threshold = float(policy["logit_threshold"])
    by_id = {row["account_id"]: row for row in predictions["predictions"]}
    test_ids = {ids[int(index)] for index in split["test"]}
    if set(by_id) != test_ids:
        raise ValueError("saved prediction inventory differs from fixed test")
    selected = sorted(by_id, key=lambda uid:
                      (abs(float(by_id[uid]["fused_logit"]) - threshold), uid))[:args.top_n]
    profiles = first_profiles(set(selected))
    frame = pd.read_csv(FEATURES)
    if frame["user_id"].astype(str).tolist() != ids or \
            frame.columns.tolist() != ["user_id", *FEATURE_FIELDS]:
        raise ValueError("named feature table differs from frozen input")
    raw_features = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(checkpoint["input_contract"]["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(checkpoint["input_contract"]["scaler_scale"], dtype=np.float64)
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

    def forward(user_x: torch.Tensor, target: int) -> dict[str, float]:
        with torch.inference_mode():
            h = float(hgt({"user": user_x}, graph)["bot_logit"][target])
            r = float(relnet(user_x, pack["relnet_edges"])[target])
        return {"hgt_logit": h, "relnet_logit": r, "fused_logit":
                float(0.5 * (torch.tensor(h, dtype=torch.float32) +
                             torch.tensor(r, dtype=torch.float32)))}

    results = []
    for rank, uid in enumerate(selected, 1):
        target = ids.index(uid)
        profile, line_no = profiles[uid]
        facts = profile_features(profile)
        expected = np.asarray([facts[name] for name in PROFILE_FIELDS], dtype=np.float32)
        if not np.array_equal(expected, raw_features[target, :len(PROFILE_FIELDS)]):
            raise ValueError(f"raw profile facts differ for {uid}")
        baseline = forward(x, target)
        saved = by_id[uid]
        if any(abs(baseline[key] - float(saved[key])) > 1e-5 for key in baseline):
            raise ValueError(f"frozen forward differs from saved prediction for {uid}")
        initial_class = int(baseline["fused_logit"] >= threshold)
        fields = []
        for field in RAW_PROFILE_FIELDS:
            if field not in profile:
                continue
            record = SourceRecord(
                f"fox8:{uid}:profile:{field}", uid, "profile_field",
                {field: profile[field]},
                f"{RAW.resolve()}#line={line_no};tweet=0;user.{field}")
            changed_profile = dict(profile)
            changed_profile.pop(field)
            recomputed = profile_features(changed_profile)
            changed_columns = [name for name in PROFILE_FIELDS
                               if np.float32(recomputed[name]) != np.float32(facts[name])]
            if changed_columns:
                edited_raw = raw_features[target:target + 1].copy()
                edited_raw[0, :len(PROFILE_FIELDS)] = [recomputed[name]
                                                       for name in PROFILE_FIELDS]
                edited_x = x.clone()
                edited_x[target:target + 1] = torch.from_numpy(
                    np.clip(scaler.transform(edited_raw), -3, 3).astype(np.float32))
                observed = forward(edited_x, target)
                edited_hash = tensor_hash(edited_x)
            else:
                observed = baseline
                edited_hash = tensor_hash(x)
            fields.append({
                "field": field, "source_id": record.source_id,
                "source_sha256": record.sha256,
                "raw_location": record.raw_location,
                "raw_value": profile[field],
                "changed_model_facts": changed_columns,
                "edited_input_sha256": edited_hash,
                "after": observed,
                "fused_delta_logit": observed["fused_logit"] - baseline["fused_logit"],
                "class_changed": int(observed["fused_logit"] >= threshold) != initial_class,
            })
        results.append({"rank_by_margin": rank, "account_id": uid,
                        "label": int(saved["label"]), "raw_profile_line": line_no,
                        "before": baseline, "margin_logit": baseline["fused_logit"] - threshold,
                        "fields": fields})
        print(json.dumps({"rank": rank, "account_id": uid,
                          "single_source_flips": sum(row["class_changed"] for row in fields)}),
              flush=True)
    flips = [{"account_id": row["account_id"], "field": item["field"],
              "fused_delta_logit": item["fused_delta_logit"]}
             for row in results for item in row["fields"] if item["class_changed"]]
    output = {
        "selection_rule": "closest fixed-test fused logits to saved dev-only threshold, before source-edit inspection",
        "scope": "exploratory boundary subset; not a full-test prevalence estimate",
        "operation": "remove one raw first-profile field; recompute all dependent profile facts; graph unchanged",
        "top_n": len(selected), "fixed_test_n": len(test_ids),
        "threshold_logit": threshold,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "policy_sha256": sha256_file(RUN / "decision_policy.json"),
        "predictions_sha256": sha256_file(RUN / "test_predictions.json"),
        "raw_source_sha256": sha256_file(RAW),
        "feature_file_sha256": sha256_file(FEATURES),
        "single_source_operations": sum(len(row["fields"]) for row in results),
        "single_source_flips": flips,
        "accounts": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"single_source_operations": output["single_source_operations"],
                      "single_source_flips": flips,
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
