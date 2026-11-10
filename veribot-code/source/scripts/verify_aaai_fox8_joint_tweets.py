"""Independently replay the Fox8 joint-tweet probe from raw records and weights."""

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
from certify_aaai_veribot_fox8_tweet import has_internal_relation
from train_aaai_veribot_fox8 import load_inputs, sha256_file, tensor_hash
from verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS
from verifiable_bot.aaai_fox8_raw_features import account_features

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
FEATURES = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
PROBE = ROOT / "research/certificates/aaai_veribot_fox8_joint_tweet_probe.json"
OUTPUT = ROOT / "research/certificates/aaai_veribot_fox8_joint_tweet_probe_verification.json"


def main() -> None:
    torch.set_num_threads(4)
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    if probe["raw_source_sha256"] != sha256_file(RAW) or \
            probe["checkpoint_sha256"] != sha256_file(RUN / "model.pt"):
        raise ValueError("source or checkpoint hash differs")
    x, _, split, ids, graph, pack = load_inputs()
    uid = probe["account_id"]
    if uid not in {ids[int(i)] for i in split["test"]}:
        raise ValueError("account outside fixed test split")
    index = ids.index(uid)
    ordinals = sorted(int(i) for i in probe["eligible_ordinals"])
    row = None
    line_number = None
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            candidate = json.loads(line)
            if str(candidate["user_id"]) == uid:
                row, line_number = candidate, n
                break
    if row is None or line_number != probe["raw_account_line"]:
        raise ValueError("raw account location differs")
    if any(has_internal_relation(row["user_tweets"][i], uid, set(ids))
           for i in ordinals):
        raise ValueError("a claimed graph-neutral post has an internal relation")
    frame = pd.read_csv(FEATURES)
    if frame["user_id"].astype(str).tolist() != ids:
        raise ValueError("feature table account order differs")
    raw_matrix = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
    baseline_facts = account_features(row)
    if not np.allclose(
            np.asarray([baseline_facts[name] for name in FEATURE_FIELDS], dtype=np.float32),
            raw_matrix[index], rtol=1e-6, atol=1e-5):
        raise ValueError("raw account facts differ from input table")
    checkpoint = torch.load(RUN / "model.pt", map_location="cpu", weights_only=False)
    contract = checkpoint["input_contract"]
    if contract["feature_sha256"] != tensor_hash(x):
        raise ValueError("baseline model input differs")
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
    threshold = float(probe["threshold_logit"])

    def replay(removed: tuple[int, ...]) -> dict[str, float | bool]:
        missing = set(removed)
        remaining = [tweet for i, tweet in enumerate(row["user_tweets"])
                     if i not in missing]
        facts = account_features({**row, "user_tweets": remaining})
        edited_row = np.asarray([[facts[name] for name in FEATURE_FIELDS]], dtype=np.float32)
        edited = x.clone()
        edited[index] = torch.from_numpy(np.clip(
            scaler.transform(edited_row)[0], -3, 3).astype(np.float32))
        with torch.inference_mode():
            h = float(hgt({"user": edited}, graph)["bot_logit"][index])
            r = float(relnet(edited, pack["relnet_edges"])[index])
        fused = float(0.5 * (torch.tensor(h, dtype=torch.float32) +
                             torch.tensor(r, dtype=torch.float32)))
        return {"hgt_logit": h, "relnet_logit": r, "fused_logit": fused,
                "predicted_bot": fused >= threshold}

    def check(actual: dict, expected: dict) -> float:
        difference = max(abs(actual[k] - expected[k])
                         for k in ("hgt_logit", "relnet_logit", "fused_logit"))
        if difference > 1e-5 or actual["predicted_bot"] != expected["predicted_bot"]:
            raise ValueError(f"frozen forward differs: {difference}")
        return difference

    max_difference = check(replay(()), probe["baseline"])
    for ordinal in ordinals:
        observed = replay((ordinal,))["fused_logit"] - probe["baseline"]["fused_logit"]
        if abs(observed - float(probe["single_delta_logit"][str(ordinal)])) > 1e-5:
            raise ValueError(f"single-source response differs: {ordinal}")
    rows = probe["triple_comparison"]["all_groups"]
    if len(rows) != len(list(combinations(ordinals, 3))) or \
            {tuple(z["ordinals"]) for z in rows} != set(combinations(ordinals, 3)):
        raise ValueError("triple inventory is incomplete")
    flip_count = 0
    for position, saved in enumerate(rows, 1):
        actual = replay(tuple(saved["ordinals"]))
        difference = abs(actual["fused_logit"] - saved["fused_logit"])
        max_difference = max(max_difference, difference)
        if difference > 1e-5 or actual["predicted_bot"] != saved["predicted_bot"]:
            raise ValueError(f"triple response differs at row {position}")
        flip_count += not actual["predicted_bot"]
        if position % 100 == 0:
            print(json.dumps({"checked_triples": position}), flush=True)
    if flip_count != probe["triple_comparison"]["class_flipping_groups"]:
        raise ValueError("triple flip count differs")
    chosen = tuple(probe["triple_comparison"]["chosen_group"])
    max_difference = max(max_difference, check(
        replay(chosen), probe["paths"]["bot_support"][2]["after"]))
    anchored_kept = frozenset((0, *chosen))
    if set(range(1, len(row["user_tweets"]))) != set(ordinals):
        raise ValueError("retention check needs a complete graph-neutral tweet pool")
    chosen_retained = replay(tuple(i for i in range(len(row["user_tweets"]))
                                   if i not in anchored_kept))
    anchor_only = replay(tuple(ordinals))
    paired_contexts = []
    for extra in (i for i in ordinals if i not in chosen):
        with_group = replay((extra,))
        without_group = replay(tuple(sorted((extra, *chosen))))
        paired_contexts.append({
            "other_removed_ordinal": extra,
            "with_group": with_group,
            "without_group": without_group,
            "group_add_back_delta_logit": (with_group["fused_logit"] -
                                           without_group["fused_logit"]),
            "group_add_back_changes_class": (with_group["predicted_bot"] !=
                                             without_group["predicted_bot"]),
        })
    pair_rows = [{"ordinals": list(pair), "response": replay(pair)}
                 for pair in combinations(chosen, 2)]
    all_pair_flip_count = 0
    for pair in combinations(ordinals, 2):
        all_pair_flip_count += not replay(pair)["predicted_bot"]
    search = probe["coalition_search"]
    if (search["minimum_cardinality"] != 3 or
            not search["minimum_cardinality_proven"] or
            not search["all_minimum_groups_found"] or
            search["baseline_bot"] is not True or
            search["search_universe"] != ordinals or
            search["unique_evaluations"] != 1 + len(ordinals) +
            len(list(combinations(ordinals, 2))) + len(rows)):
        raise ValueError("automatic coalition-search scope or proof differs")
    independently_flipping = {
        tuple(saved["ordinals"]) for saved in rows if not saved["predicted_bot"]}
    if {tuple(found["sources"]) for found in search["minimal_groups"]} != independently_flipping:
        raise ValueError("automatic search missed a minimum alternative")
    checked_forward_contexts = 0
    for found in search["minimal_groups"]:
        group = tuple(found["sources"])
        observed = replay(group)
        max_difference = max(max_difference, abs(
            observed["fused_logit"] - found["after_removal_logit"]))
        if observed["predicted_bot"] or max_difference > 1e-5:
            raise ValueError("minimum group response differs")
        expected_others = set(ordinals) - set(group)
        forward = found["forward_contexts"]
        if ({c["other_removed_source"] for c in forward["contexts"]} != expected_others
                or forward["context_count"] != len(expected_others)):
            raise ValueError("forward context inventory differs")
        changed = 0
        for context in forward["contexts"]:
            other = context["other_removed_source"]
            without = replay(tuple(sorted((*group, other))))
            with_group = replay((other,))
            differences = (
                abs(without["fused_logit"] - context["without_group_logit"]),
                abs(with_group["fused_logit"] - context["with_group_logit"]),
            )
            max_difference = max(max_difference, *differences)
            if (max(differences) > 1e-5 or
                    without["predicted_bot"] != context["without_group_bot"] or
                    with_group["predicted_bot"] != context["with_group_bot"]):
                raise ValueError("forward context response differs")
            effect = with_group["fused_logit"] - without["fused_logit"]
            if abs(effect - context["signed_addition_effect_toward_target"]) > 1e-5:
                raise ValueError("forward context effect differs")
            class_changed = with_group["predicted_bot"] and not without["predicted_bot"]
            if class_changed != context["addition_changes_to_target"]:
                raise ValueError("forward context class change differs")
            changed += class_changed
            checked_forward_contexts += 1
        if changed != forward["class_change_count"]:
            raise ValueError("forward context aggregate differs")
    result = {"status": "verified_from_raw_sources_and_frozen_models",
              "probe_sha256": sha256_file(PROBE), "account_id": uid,
              "checked_single_sources": len(ordinals),
              "checked_same_size_triples": len(rows),
              "class_flipping_triples": flip_count,
              "chosen_group_all_proper_pairs": pair_rows,
              "chosen_group_pair_flips": sum(not p["response"]["predicted_bot"]
                                             for p in pair_rows),
              "all_same_account_pair_groups": len(list(combinations(ordinals, 2))),
              "all_pair_class_flips": all_pair_flip_count,
              "automatically_found_minimum_groups": len(search["minimal_groups"]),
              "checked_forward_contexts_all_minimum_groups": checked_forward_contexts,
              "retention_context": {
                  "invariant_profile_anchor_ordinal": 0,
                  "chosen_plus_anchor_retained_tweet_count": len(anchored_kept),
                  "chosen_plus_anchor_response": chosen_retained,
                  "anchor_only_response": anchor_only,
              },
              "paired_add_back_contexts": paired_contexts,
              "paired_add_back_class_changes": sum(
                  p["group_add_back_changes_class"] for p in paired_contexts),
              "maximum_output_difference": max_difference}
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "chosen_group_all_proper_pairs"}),
          flush=True)


if __name__ == "__main__":
    main()
