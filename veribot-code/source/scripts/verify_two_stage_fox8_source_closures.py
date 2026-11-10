"""Verify raw-source locators and exact RelNet scores in two-stage certificates."""

from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "source/scripts"),
                str(ROOT / "source/hgt_import/code")]

from hgt_model import VerifiableHGT
from relnet_reimplementation import GatedRelNet
from train_aaai_veribot_fox8 import load_inputs, sha256_file
from verifiable_bot.fox8_atomic_sources import (
    graph_from_relation_atoms, load_account_atoms, load_relation_atoms,
)
from verifiable_bot.fox8_two_stage_inputs import ALL_FACTS, Fox8TwoStageInputs
from verifiable_bot.fox8_two_stage_model import HGTSourceSelector, TwoStageFox8
from verifiable_bot.fox8_two_stage_intervention import TwoStageIntervention

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
BASE = ROOT / "output/aaai_veribot_fox8/seed0/model.pt"


def _canonical_hash(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def _raw_lines(needed: set[int]) -> dict[int, dict]:
    found = {}
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line_number in needed:
                found[line_number] = json.loads(line)
            if len(found) == len(needed):
                break
    if set(found) != needed:
        raise ValueError("certificate raw line is absent")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=ROOT / "output/two_stage_fox8_evidence/pilot_replay")
    args = parser.parse_args()
    probe_file = args.run / "full_path_source_group_probe.json"
    probe = json.loads(probe_file.read_text(encoding="utf-8"))
    checkpoint_file = args.run / "model.pt"
    saved = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if (probe["checkpoint_sha256"] != sha256_file(checkpoint_file) or
            probe["raw_source_sha256"] != sha256_file(RAW)):
        raise ValueError("source certificate does not match model or raw file")
    needed = set()
    for account in probe["accounts"]:
        for closure in account["class_source_closures"].values():
            for record in closure["source_records"]:
                if record["kind"] == "native_relation_event":
                    needed.update(item["line"] for item in record["raw_locators"])
                else:
                    needed.add(int(record["raw_location"].split(";", 1)[0][5:]))
    raw = _raw_lines(needed)
    x, _, _, ids, _, pack = load_inputs()
    base = torch.load(BASE, map_location="cpu", weights_only=False)
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3).eval()
    hgt.load_state_dict(base["model_state"]["hgt"])
    accounts = load_account_atoms(RAW, ids)
    relations = load_relation_atoms(EVENTS, RAW, ids)
    bank = Fox8TwoStageInputs.build(accounts, relations, ids)
    model = TwoStageFox8(
        HGTSourceSelector(context_dim=128),
        GatedRelNet(len(ALL_FACTS), hidden_channels=128,
                     relation_names=tuple(bank.full_graph),
                     num_layers=2, dropout=0.3, out_channels=2),
        torch.tensor(saved["scaler_mean"]),
        torch.tensor(saved["scaler_scale"])).eval()
    model.load_state_dict(saved["model_state"])
    oracle = TwoStageIntervention(
        hgt=hgt, model=model, batch=bank,
        base_mean=np.asarray(base["input_contract"]["scaler_mean"]),
        base_scale=np.asarray(base["input_contract"]["scaler_scale"]))
    lookup = oracle.lookup
    relation_by_id = {item.source_id: item for item in relations}
    max_score_difference = 0.0
    max_fact_difference = 0.0
    checked = 0
    checked_intervention_groups = 0
    checked_group_members = 0
    for account in probe["accounts"]:
        uid = account["account_id"]
        target = ids.index(uid)
        for class_index, name in ((0, "human"), (1, "bot")):
            closure = account["class_source_closures"][name]
            if closure["account_id"] != uid or closure["class_hypothesis"].lower() != name:
                raise ValueError("certificate class or target differs")
            masks = {"profile": torch.zeros(len(bank.profile_account)),
                     "tweet": torch.zeros(len(bank.tweet_account)),
                     "relation": torch.zeros(len(bank.relations))}
            seen = set()
            for record in closure["source_records"]:
                source_id = record["source_id"]
                if source_id in seen or source_id not in lookup:
                    raise ValueError("duplicate or unknown certificate source")
                seen.add(source_id)
                kind, position = lookup[source_id]
                masks[kind][position] = 1
                if record["kind"] == "profile_field":
                    line, suffix = record["raw_location"].split(";tweet=0;user.")
                    line_number = int(line[5:])
                    field = suffix
                    value = raw[line_number]["user_tweets"][0]["user"][field]
                    if value != record["value"] or _canonical_hash(value) != record[
                            "value_sha256"]:
                        raise ValueError("profile source value or hash differs")
                elif record["kind"] == "tweet":
                    line, tweet = record["raw_location"].split(";tweet=")
                    item = raw[int(line[5:])]["user_tweets"][int(tweet)]
                    if (_canonical_hash(item) != record["record_sha256"] or
                            str(item.get("id_str", item.get("id", ""))) !=
                            record["tweet_id"]):
                        raise ValueError("tweet source value or hash differs")
                elif record["kind"] == "native_relation_event":
                    atom = relation_by_id[source_id]
                    expected = [(line, tweet, field) for line, tweet, field
                                in atom.raw_locators]
                    observed = [(item["line"], item["tweet"], item["field"])
                                for item in record["raw_locators"]]
                    if expected != observed:
                        raise ValueError("relation locators differ")
                    for line, tweet_index, _ in expected:
                        item = raw[line]["user_tweets"][tweet_index]
                        if atom.relation == "reply":
                            target_id = item.get("in_reply_to_user_id_str") or item.get(
                                "in_reply_to_user_id")
                        else:
                            target_user = (item.get("retweeted_status") or {}).get("user") or {}
                            target_id = target_user.get("id_str") or target_user.get("id")
                        if str(target_id) != atom.target_account_id:
                            raise ValueError("raw relation pointer differs")
                else:
                    raise ValueError("unknown certificate source kind")
                checked += 1
            facts = bank.exact_view(masks["profile"], masks["tweet"])
            target_facts = np.asarray([account["target_selected_view_facts"][name][key]
                                       for key in ALL_FACTS], dtype=np.float32)
            difference = float(np.max(np.abs(facts[target] - target_facts)))
            max_fact_difference = max(max_fact_difference, difference)
            if difference > 1e-5:
                raise ValueError("source closure target facts differ")
            scaled = torch.from_numpy(np.clip(
                (facts - np.asarray(saved["scaler_mean"])) /
                np.asarray(saved["scaler_scale"]), -3, 3).astype(np.float32))
            relation_ids = [item.source_id for item, keep in
                            zip(relations, masks["relation"].tolist()) if keep]
            graph = graph_from_relation_atoms(relations, ids, relation_ids)
            with torch.inference_mode():
                observed_score = float(model.relnet(scaled, graph)[target, class_index])
            expected_score = float(account["baseline"][f"{name}_score"])
            difference = abs(observed_score - expected_score)
            max_score_difference = max(max_score_difference, difference)
            if difference > 1e-5:
                raise ValueError("source closure RelNet score differs")
        baseline = oracle.predict(uid)
        if abs(baseline["margin"] - account["baseline"]["margin"]) > 1e-5:
            raise ValueError("full-path baseline margin differs")
        candidates = oracle.target_source_ids(uid)
        if candidates != tuple(account["search"]["search_universe"]):
            raise ValueError("source search universe differs")
        for solution in account["search"]["solutions"]:
            group = frozenset(solution["sources"])
            after = oracle.predict(uid, group)["margin"]
            if (abs(after - solution["after_removal_logit"]) > 1e-5 or
                    (after >= probe["threshold_dev_only"]) ==
                    account["predicted_bot"]):
                raise ValueError("full-path class-changing group differs")
            checked_intervention_groups += 1
            if solution["locally_minimal_checked"]:
                for member in group:
                    subset = group - {member}
                    subset_margin = oracle.predict(uid, subset)["margin"]
                    if (subset_margin >= probe["threshold_dev_only"]) != account[
                            "predicted_bot"]:
                        raise ValueError("claimed locally minimal group has redundant member")
                    checked_group_members += 1
    result = {"status": "verified_raw_sources_two_hop_scores_and_full_path_groups",
              "probe_sha256": sha256_file(probe_file),
              "checked_source_records": checked,
              "checked_account_class_closures": 2 * len(probe["accounts"]),
              "checked_full_path_flip_groups": checked_intervention_groups,
              "checked_locally_minimal_group_members": checked_group_members,
              "maximum_fact_difference": max_fact_difference,
              "maximum_score_difference": max_score_difference}
    output = args.run / "full_path_source_group_probe_verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
