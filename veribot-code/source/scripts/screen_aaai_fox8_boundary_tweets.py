"""Screen all graph-neutral raw tweet deletions for preselected Fox8 boundary accounts."""

from __future__ import annotations

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
from certify_aaai_veribot_fox8_tweet import has_internal_relation
from train_aaai_veribot_fox8 import load_inputs, sha256_file, tensor_hash
from verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS, TWEET_FIELDS
from verifiable_bot.aaai_fox8_raw_features import account_features
from verifiable_bot.evidence_v2 import SourceRecord, digest

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
FEATURES = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
BOUNDARY = ROOT / "research/certificates/aaai_veribot_fox8_fixed_threshold_profile_screen.json"
OUTPUT = ROOT / "research/certificates/aaai_veribot_fox8_boundary_tweet_screen.json"


def main() -> None:
    torch.set_num_threads(4)
    boundary = json.loads(BOUNDARY.read_text(encoding="utf-8"))
    selected = [row["account_id"] for row in boundary["accounts"]]
    x, _, split, ids, graph, pack = load_inputs()
    if set(selected) - {ids[int(i)] for i in split["test"]}:
        raise ValueError("boundary accounts leave the fixed test split")
    checkpoint = torch.load(RUN / "model.pt", map_location="cpu", weights_only=False)
    policy = json.loads((RUN / "decision_policy.json").read_text(encoding="utf-8"))
    saved = json.loads((RUN / "test_predictions.json").read_text(encoding="utf-8"))
    if (checkpoint["input_contract"]["feature_sha256"] != tensor_hash(x)
            or policy["checkpoint_sha256"] != sha256_file(RUN / "model.pt")
            or policy["saved_predictions_sha256"] != sha256_file(RUN / "test_predictions.json")
            or boundary["threshold_logit"] != policy["logit_threshold"]):
        raise ValueError("frozen input, output, and threshold differ")
    by_id = {row["account_id"]: row for row in saved["predictions"]}
    frame = pd.read_csv(FEATURES)
    if frame.columns.tolist() != ["user_id", *FEATURE_FIELDS] or \
            frame["user_id"].astype(str).tolist() != ids:
        raise ValueError("feature table differs from frozen account order")
    raw = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    contract = checkpoint["input_contract"]
    scaler.mean_ = np.asarray(contract["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(contract["scaler_scale"], dtype=np.float64)
    scaler.n_features_in_ = len(FEATURE_FIELDS)
    rows = {}
    wanted = set(selected)
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid in wanted and uid not in rows:
                rows[uid] = (row, line_no)
    if set(rows) != wanted:
        raise ValueError("raw source lacks a selected account")
    account_set = set(ids)
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

    threshold = float(policy["logit_threshold"])
    result_rows = []
    for rank, uid in enumerate(selected, 1):
        index = ids.index(uid)
        row, line_no = rows[uid]
        tweets = row["user_tweets"]
        baseline_facts = account_features(row)
        expected = np.asarray([baseline_facts[name] for name in FEATURE_FIELDS],
                              dtype=np.float32)
        if not np.allclose(expected, raw[index], rtol=1e-6, atol=1e-5):
            raise ValueError(f"raw account facts differ: {uid}")
        before = forward(x, index)
        if any(abs(before[name] - float(by_id[uid][name])) > 1e-5 for name in before):
            raise ValueError(f"saved frozen prediction differs: {uid}")
        effects = []
        for ordinal in range(1, len(tweets)):
            tweet = tweets[ordinal]
            if has_internal_relation(tweet, uid, account_set):
                continue
            remaining = tweets[:ordinal] + tweets[ordinal + 1:]
            computed = account_features({**row, "user_tweets": remaining})
            changed_names = [name for name in TWEET_FIELDS
                             if np.float32(computed[name]) != np.float32(baseline_facts[name])]
            if "n_tweets" not in changed_names:
                raise ValueError("source deletion did not change tweet count")
            edited_raw = raw[index:index + 1].copy()
            edited_raw[0] = [computed[name] for name in FEATURE_FIELDS]
            edited = x.clone()
            edited[index:index + 1] = torch.from_numpy(np.clip(
                scaler.transform(edited_raw), -3, 3).astype(np.float32))
            after = forward(edited, index)
            source = SourceRecord(
                f"fox8:{uid}:tweet:{line_no}:{ordinal}", uid, "raw_tweet",
                {"tweet_id": str(tweet.get("id_str") or tweet.get("id")),
                 "tweet_sha256": digest(tweet),
                 "starts_rt": (tweet.get("text") or "").startswith("RT ")},
                f"{RAW.resolve()}#line={line_no};tweet={ordinal}")
            effects.append({
                "ordinal": ordinal, "source_id": source.source_id,
                "source_sha256": source.sha256, "raw_location": source.raw_location,
                "tweet_sha256": source.payload["tweet_sha256"],
                "starts_rt": source.payload["starts_rt"],
                "changed_model_facts": changed_names,
                "edited_input_sha256": tensor_hash(edited),
                "after": after,
                "fused_delta_logit": after["fused_logit"] - before["fused_logit"],
                "final_class_changed": ((after["fused_logit"] >= threshold)
                                        != (before["fused_logit"] >= threshold)),
            })
        result_rows.append({"rank_by_margin": rank, "account_id": uid,
                            "label": int(by_id[uid]["label"]),
                            "raw_account_line": line_no, "raw_tweet_count": len(tweets),
                            "before": before, "effects": effects})
        print(json.dumps({"rank": rank, "account_id": uid,
                          "eligible_tweets": len(effects),
                          "class_flips": sum(e["final_class_changed"] for e in effects)}),
              flush=True)
    all_effects = [effect for row in result_rows for effect in row["effects"]]
    output = {
        "scope": "all nonfirst tweets without internal model graph pointers from 20 preselected fixed-test boundary accounts",
        "operation": "remove one complete raw tweet; recompute all 8 tweet facts; keep raw profile and native graph unchanged",
        "selection_rule": "accounts ranked by saved fused-logit distance to dev threshold before tweet edits",
        "boundary_screen_sha256": sha256_file(BOUNDARY),
        "raw_source_sha256": sha256_file(RAW),
        "feature_file_sha256": sha256_file(FEATURES),
        "checkpoint_sha256": sha256_file(RUN / "model.pt"),
        "decision_policy_sha256": sha256_file(RUN / "decision_policy.json"),
        "threshold_logit": threshold,
        "selected_accounts": len(selected),
        "eligible_operations": len(all_effects),
        "score_changing_operations_gt_1e-6": sum(
            abs(effect["fused_delta_logit"]) > 1e-6 for effect in all_effects),
        "class_changing_operations": sum(effect["final_class_changed"]
                                         for effect in all_effects),
        "accounts": result_rows,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({key: output[key] for key in
                      ("selected_accounts", "eligible_operations",
                       "score_changing_operations_gt_1e-6",
                       "class_changing_operations")}), flush=True)


if __name__ == "__main__":
    main()
