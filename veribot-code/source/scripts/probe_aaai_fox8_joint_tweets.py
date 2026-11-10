"""Exploratory source-consistent joint tweet probe on one fixed Fox8 boundary account."""

from __future__ import annotations

import gzip
from itertools import combinations
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
from verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS
from verifiable_bot.aaai_fox8_raw_features import account_features
from verifiable_bot.source_coalition_search import (
    check_forward_contexts, find_minimal_flip_groups,
)

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
FEATURES = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
SCREEN = ROOT / "research/certificates/aaai_veribot_fox8_boundary_tweet_screen.json"
OUTPUT = ROOT / "research/certificates/aaai_veribot_fox8_joint_tweet_probe.json"


def main() -> None:
    torch.set_num_threads(4)
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if screen["raw_source_sha256"] != sha256_file(RAW) or \
            screen["checkpoint_sha256"] != sha256_file(RUN / "model.pt"):
        raise ValueError("source or checkpoint changed since single-source screen")
    chosen = next(a for a in screen["accounts"]
                  if 12 <= len(a["effects"]) <= 20
                  and a["before"]["fused_logit"] >= screen["threshold_logit"]
                  and not any(e["final_class_changed"] for e in a["effects"]))
    uid = chosen["account_id"]
    ordinals = sorted(e["ordinal"] for e in chosen["effects"])
    single = {e["ordinal"]: e["after"]["fused_logit"] -
              chosen["before"]["fused_logit"] for e in chosen["effects"]}
    x, _, split, ids, graph, pack = load_inputs()
    if uid not in {ids[int(i)] for i in split["test"]}:
        raise ValueError("account not in fixed test split")
    index = ids.index(uid)
    checkpoint = torch.load(RUN / "model.pt", map_location="cpu", weights_only=False)
    if checkpoint["input_contract"]["feature_sha256"] != tensor_hash(x):
        raise ValueError("baseline input differs from checkpoint")
    policy = json.loads((RUN / "decision_policy.json").read_text(encoding="utf-8"))
    threshold = float(policy["logit_threshold"])
    if threshold != screen["threshold_logit"]:
        raise ValueError("decision threshold changed")
    frame = pd.read_csv(FEATURES)
    if frame["user_id"].astype(str).tolist() != ids:
        raise ValueError("account order differs")
    raw_values = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
    row = None
    line_number = None
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            parsed = json.loads(line)
            if str(parsed["user_id"]) == uid:
                row, line_number = parsed, n
                break
    if row is None or line_number != chosen["raw_account_line"]:
        raise ValueError("source location differs")
    baseline_facts = account_features(row)
    expected = np.asarray([baseline_facts[name] for name in FEATURE_FIELDS], dtype=np.float32)
    if not np.allclose(expected, raw_values[index], rtol=1e-6, atol=1e-5):
        raise ValueError("source facts differ from input table")
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
    cache = {}

    def evaluate(removed: frozenset[int]) -> dict[str, object]:
        if removed in cache:
            return cache[removed]
        tweets = [tweet for i, tweet in enumerate(row["user_tweets"])
                  if i not in removed]
        facts = account_features({**row, "user_tweets": tweets})
        edited_raw = raw_values[index:index + 1].copy()
        edited_raw[0] = [facts[name] for name in FEATURE_FIELDS]
        edited = x.clone()
        edited[index:index + 1] = torch.from_numpy(np.clip(
            scaler.transform(edited_raw), -3, 3).astype(np.float32))
        with torch.inference_mode():
            h = float(hgt({"user": edited}, graph)["bot_logit"][index])
            r = float(relnet(edited, pack["relnet_edges"])[index])
        fused = float(0.5 * (torch.tensor(h, dtype=torch.float32)
                             + torch.tensor(r, dtype=torch.float32)))
        result = {"hgt_logit": h, "relnet_logit": r, "fused_logit": fused,
                  "model_input_sha256": tensor_hash(edited),
                  "predicted_bot": fused >= threshold}
        cache[removed] = result
        return result

    baseline = evaluate(frozenset())
    if any(abs(baseline[k] - chosen["before"][k]) > 1e-5
           for k in ("hgt_logit", "relnet_logit", "fused_logit")):
        raise ValueError("baseline prediction differs from single-source screen")
    for ordinal in ordinals:
        observed = evaluate(frozenset({ordinal}))["fused_logit"] - baseline["fused_logit"]
        if abs(observed - single[ordinal]) > 1e-5:
            raise ValueError(f"single-source effect differs: {ordinal}")

    paths = {}
    for direction in ("bot_support", "human_counterevidence"):
        picked = frozenset()
        steps = []
        for _ in ordinals:
            candidates = [(ordinal, evaluate(picked | {ordinal}))
                          for ordinal in ordinals if ordinal not in picked]
            ordinal, response = min(
                candidates, key=lambda z: (z[1]["fused_logit"], z[0])) \
                if direction == "bot_support" else max(
                    candidates, key=lambda z: (z[1]["fused_logit"], -z[0]))
            picked = picked | {ordinal}
            actual_delta = response["fused_logit"] - baseline["fused_logit"]
            additive_delta = sum(single[i] for i in picked)
            steps.append({"added_ordinal": ordinal,
                          "removed_ordinals": sorted(picked),
                          "source_ids": [f"fox8:{uid}:tweet:{line_number}:{i}"
                                         for i in sorted(picked)],
                          "after": response,
                          "actual_delta_logit": actual_delta,
                          "sum_single_delta_logit": additive_delta,
                          "interaction_residual_logit": actual_delta - additive_delta})
            print(json.dumps({"direction": direction, "size": len(picked),
                              "fused_logit": response["fused_logit"],
                              "predicted_bot": response["predicted_bot"]}), flush=True)
            if direction == "bot_support" and not response["predicted_bot"]:
                break
        paths[direction] = steps
    chosen_triple = tuple(paths["bot_support"][2]["removed_ordinals"])
    triple_rows = []
    for triple in combinations(ordinals, 3):
        response = evaluate(frozenset(triple))
        triple_rows.append({"ordinals": list(triple),
                            "fused_logit": response["fused_logit"],
                            "predicted_bot": response["predicted_bot"]})
    sorted_triples = sorted(triple_rows, key=lambda z: (z["fused_logit"], z["ordinals"]))
    chosen_rank = next(i for i, z in enumerate(sorted_triples, 1)
                       if z["ordinals"] == list(chosen_triple))
    triple_comparison = {
        "all_same_account_same_size_groups": len(triple_rows),
        "class_flipping_groups": sum(not z["predicted_bot"] for z in triple_rows),
        "chosen_group": list(chosen_triple),
        "chosen_rank_lowest_fused_logit": chosen_rank,
        "minimum_fused_logit": sorted_triples[0]["fused_logit"],
        "median_fused_logit": sorted_triples[len(sorted_triples) // 2]["fused_logit"],
        "maximum_fused_logit": sorted_triples[-1]["fused_logit"],
        "all_groups": sorted_triples,
    }
    coalition_search = find_minimal_flip_groups(
        ordinals, lambda removed: evaluate(removed)["fused_logit"], threshold,
        max_evaluations=10000)
    if coalition_search["minimum_cardinality"] != 3 or not coalition_search["all_minimum_groups_found"]:
        raise ValueError("independent-size coalition search differs from exhaustive triple screen")
    if {tuple(z["sources"]) for z in coalition_search["minimal_groups"]} != {
            tuple(z["ordinals"]) for z in triple_rows if not z["predicted_bot"]}:
        raise ValueError("minimum flip groups differ from exhaustive comparison")
    for found in coalition_search["minimal_groups"]:
        group = tuple(found["sources"])
        found["single_effect_sum_logit"] = sum(single[i] for i in group)
        found["joint_effect_logit"] = found["after_removal_logit"] - baseline["fused_logit"]
        found["interaction_residual_logit"] = (
            found["joint_effect_logit"] - found["single_effect_sum_logit"])
        found["forward_contexts"] = check_forward_contexts(
            group, [i for i in ordinals if i not in group],
            lambda removed: evaluate(removed)["fused_logit"], threshold,
            target_bot=coalition_search["baseline_bot"])
    result = {"scope": "one preselected fixed-test boundary account, exploratory greedy joint source edits",
              "selection_rule": "first account in prior single-source boundary screen with 12-20 eligible tweets, bot prediction, and no single-tweet class flip",
              "account_id": uid, "raw_account_line": line_number,
              "raw_source_sha256": sha256_file(RAW),
              "single_screen_sha256": sha256_file(SCREEN),
              "checkpoint_sha256": sha256_file(RUN / "model.pt"),
              "threshold_logit": threshold, "eligible_ordinals": ordinals,
              "baseline": baseline, "single_delta_logit": single,
              "paths": paths, "triple_comparison": triple_comparison,
              "coalition_search": coalition_search,
              "unique_frozen_forwards": len(cache)}
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT),
                      "unique_frozen_forwards": len(cache)}), flush=True)


if __name__ == "__main__":
    main()
