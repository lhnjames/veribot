"""Search raw profile-source coalitions for both predicted classes on Fox8."""

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
from train_aaai_veribot_fox8 import load_inputs, sha256_file, tensor_hash
from verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS, PROFILE_FIELDS
from verifiable_bot.aaai_fox8_raw_features import RAW_PROFILE_FIELDS, profile_features
from verifiable_bot.source_coalition_search import (
    check_forward_contexts, find_minimal_flip_groups,
)

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
FEATURES = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
SCREEN = ROOT / "research/certificates/aaai_veribot_fox8_fixed_threshold_profile_screen.json"
OUTPUT = ROOT / "research/certificates/aaai_veribot_fox8_dual_class_profile_coalitions.json"


def main() -> None:
    torch.set_num_threads(4)
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if (screen["raw_source_sha256"] != sha256_file(RAW) or
            screen["checkpoint_sha256"] != sha256_file(RUN / "model.pt")):
        raise ValueError("screen is not bound to current source and model")
    threshold = float(screen["threshold_logit"])
    selected = {}
    for account in screen["accounts"]:
        predicted_bot = account["before"]["fused_logit"] >= threshold
        if predicted_bot not in selected and int(predicted_bot) == account["label"]:
            selected[predicted_bot] = account
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise ValueError("the existing boundary screen lacks both correct classes")

    x, _, split, ids, graph, pack = load_inputs()
    test_ids = {ids[int(i)] for i in split["test"]}
    checkpoint = torch.load(RUN / "model.pt", map_location="cpu", weights_only=False)
    contract = checkpoint["input_contract"]
    if contract["feature_sha256"] != tensor_hash(x):
        raise ValueError("checkpoint input differs")
    frame = pd.read_csv(FEATURES)
    if frame["user_id"].astype(str).tolist() != ids:
        raise ValueError("feature table order differs")
    raw_values = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
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

    wanted = {account["account_id"] for account in selected.values()}
    raw_profiles = {}
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            parsed = json.loads(line)
            uid = str(parsed["user_id"])
            if uid in wanted and uid not in raw_profiles:
                raw_profiles[uid] = (parsed["user_tweets"][0]["user"], line_number)
    if set(raw_profiles) != wanted:
        raise ValueError("selected raw profiles missing")

    outcomes = []
    for target_bot, account in selected.items():
        uid = account["account_id"]
        if uid not in test_ids:
            raise ValueError("selected account outside fixed test split")
        profile, line_number = raw_profiles[uid]
        if line_number != account["raw_profile_line"]:
            raise ValueError("raw profile location differs")
        index = ids.index(uid)
        facts = profile_features(profile)
        if not np.array_equal(
                np.asarray([facts[name] for name in PROFILE_FIELDS], dtype=np.float32),
                raw_values[index, :len(PROFILE_FIELDS)]):
            raise ValueError("profile facts differ from model input")
        field_ids = tuple(field for field in RAW_PROFILE_FIELDS if field in profile)
        screen_by_field = {item["field"]: item for item in account["fields"]}
        if set(field_ids) != set(screen_by_field):
            raise ValueError("source inventory differs from existing screen")
        cache = {}

        def evaluate(removed: frozenset[str]) -> dict:
            if removed in cache:
                return cache[removed]
            changed = {key: value for key, value in profile.items() if key not in removed}
            rebuilt = profile_features(changed)
            raw_row = raw_values[index:index + 1].copy()
            raw_row[0, :len(PROFILE_FIELDS)] = [rebuilt[name] for name in PROFILE_FIELDS]
            edited = x.clone()
            edited[index:index + 1] = torch.from_numpy(np.clip(
                scaler.transform(raw_row), -3, 3).astype(np.float32))
            with torch.inference_mode():
                h = float(hgt({"user": edited}, graph)["bot_logit"][index])
                r = float(relnet(edited, pack["relnet_edges"])[index])
            fused = float(0.5 * (torch.tensor(h, dtype=torch.float32) +
                                 torch.tensor(r, dtype=torch.float32)))
            cache[removed] = {"hgt_logit": h, "relnet_logit": r,
                              "fused_logit": fused, "predicted_bot": fused >= threshold,
                              "model_input_sha256": tensor_hash(edited)}
            return cache[removed]

        baseline = evaluate(frozenset())
        if baseline["predicted_bot"] != target_bot or any(
                abs(baseline[name] - account["before"][name]) > 1e-5
                for name in ("hgt_logit", "relnet_logit", "fused_logit")):
            raise ValueError("baseline differs from existing screen")
        search = find_minimal_flip_groups(
            field_ids, lambda removed: evaluate(removed)["fused_logit"],
            threshold, max_evaluations=10000)
        for field in field_ids:
            observed = evaluate(frozenset({field}))
            if abs(observed["fused_logit"] - screen_by_field[field]["after"]["fused_logit"]) > 1e-5:
                raise ValueError(f"single-source screen differs for {field}")
        for found in search["minimal_groups"]:
            group = tuple(found["sources"])
            found["source_records"] = [
                {key: screen_by_field[field][key] for key in
                 ("source_id", "source_sha256", "raw_location", "raw_value")}
                for field in group]
            found["after_removal"] = evaluate(frozenset(group))
            found["forward_contexts"] = check_forward_contexts(
                group, [field for field in field_ids if field not in group],
                lambda removed: evaluate(removed)["fused_logit"], threshold,
                target_bot=target_bot)
        outcomes.append({"account_id": uid, "label": account["label"],
                         "predicted_bot": target_bot,
                         "selection_rank_by_threshold_margin": account["rank_by_margin"],
                         "raw_profile_line": line_number,
                         "baseline": baseline, "search": search,
                         "unique_source_reconstructions": len(cache)})
        print(json.dumps({"account_id": uid, "predicted_bot": target_bot,
                          "minimum_size": search["minimum_cardinality"],
                          "groups": len(search["minimal_groups"]),
                          "source_reconstructions": len(cache)}), flush=True)
    OUTPUT.write_text(json.dumps({
        "scope": "first correctly predicted account of each class in existing margin screen; exploratory",
        "selection_rule": "first correct Human and first correct Bot in fixed margin-ranked screen, before coalition edits",
        "operation": "delete raw profile fields; recompute all dependent profile facts; native graph fixed",
        "raw_source_sha256": sha256_file(RAW),
        "screen_sha256": sha256_file(SCREEN),
        "checkpoint_sha256": sha256_file(RUN / "model.pt"),
        "threshold_logit": threshold,
        "accounts": outcomes,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT)}), flush=True)


if __name__ == "__main__":
    main()
