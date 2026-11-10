"""Independently rebuild selected graph closures and their RelNet scores."""

from __future__ import annotations

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
from verify_two_stage_fox8_source_closures import _canonical_hash, _raw_lines
from verifiable_bot.fox8_atomic_sources import (
    graph_from_relation_atoms, load_account_atoms, load_relation_atoms,
)
from verifiable_bot.fox8_two_stage_inputs import ALL_FACTS, Fox8TwoStageInputs
from verifiable_bot.fox8_two_stage_intervention import TwoStageIntervention
from verifiable_bot.fox8_two_stage_model import HGTSourceSelector, TwoStageFox8


def main() -> None:
    torch.set_num_threads(4)
    run = ROOT / "output/two_stage_fox8_evidence/pilot_replay"
    probe_file = run / "native_graph_source_probe.json"
    probe = json.loads(probe_file.read_text(encoding="utf-8"))
    raw_path = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    events_path = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
    checkpoint = run / "model.pt"
    base_path = ROOT / "output/aaai_veribot_fox8/seed0/model.pt"
    if probe["raw_source_sha256"] != sha256_file(raw_path) or probe[
            "checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("graph closure source or model differs")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    x, _, _, ids, _, pack = load_inputs()
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3).eval()
    hgt.load_state_dict(base["model_state"]["hgt"])
    accounts = load_account_atoms(raw_path, ids)
    relations = load_relation_atoms(events_path, raw_path, ids)
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
    needed = set()
    for account in probe["accounts"].values():
        for record in account["selected_class_relnet_closure"]["source_records"]:
            if record["kind"] == "native_relation_event":
                needed.update(item["line"] for item in record["raw_locators"])
            else:
                needed.add(int(record["raw_location"].split(";", 1)[0][5:]))
    raw_lines = _raw_lines(needed)
    relation_by_id = {item.source_id: item for item in relations}
    max_score_error = 0.0
    source_count = 0
    graph_count = 0
    replay_count = 0
    for name, account in probe["accounts"].items():
        uid = account["account_id"]
        class_index = int(name == "bot")
        closure = account["selected_class_relnet_closure"]
        if closure["account_id"] != uid or closure[
                "class_hypothesis"].lower() != name:
            raise ValueError("graph closure account or class differs")
        masks = {"profile": torch.zeros(len(bank.profile_account)),
                 "tweet": torch.zeros(len(bank.tweet_account)),
                 "relation": torch.zeros(len(bank.relations))}
        seen = set()
        for record in closure["source_records"]:
            source_id = record["source_id"]
            if source_id in seen or source_id not in oracle.lookup:
                raise ValueError("duplicate or absent source in graph closure")
            seen.add(source_id)
            kind, position = oracle.lookup[source_id]
            masks[kind][position] = 1
            if record["kind"] == "profile_field":
                line, field = record["raw_location"].split(";tweet=0;user.")
                value = raw_lines[int(line[5:])]["user_tweets"][0]["user"][field]
                if value != record["value"] or _canonical_hash(value) != record[
                        "value_sha256"]:
                    raise ValueError("graph closure profile raw value differs")
            elif record["kind"] == "tweet":
                line, ordinal = record["raw_location"].split(";tweet=")
                item = raw_lines[int(line[5:])]["user_tweets"][int(ordinal)]
                if (_canonical_hash(item) != record["record_sha256"] or
                        str(item.get("id_str", item.get("id", ""))) !=
                        record["tweet_id"]):
                    raise ValueError("graph closure tweet raw record differs")
            elif record["kind"] == "native_relation_event":
                relation = relation_by_id[source_id]
                locators = [(item["line"], item["tweet"], item["field"])
                            for item in record["raw_locators"]]
                if list(relation.raw_locators) != locators:
                    raise ValueError("graph closure relation locators differ")
                for line, ordinal, _ in locators:
                    item = raw_lines[line]["user_tweets"][ordinal]
                    if relation.relation == "reply":
                        target = (item.get("in_reply_to_user_id_str") or
                                  item.get("in_reply_to_user_id"))
                    else:
                        user = ((item.get("retweeted_status") or {}).get("user") or {})
                        target = user.get("id_str") or user.get("id")
                    if str(target) != relation.target_account_id:
                        raise ValueError("graph closure raw relation pointer differs")
                graph_count += 1
            else:
                raise ValueError("unknown graph closure source kind")
            source_count += 1
        facts = bank.exact_view(masks["profile"], masks["tweet"])
        scaled = torch.from_numpy(np.clip(
            (facts - np.asarray(saved["scaler_mean"])) /
            np.asarray(saved["scaler_scale"]), -3, 3).astype(np.float32))
        relation_ids = [relation.source_id for relation, keep in zip(
            relations, masks["relation"].tolist()) if keep]
        graph = graph_from_relation_atoms(relations, ids, relation_ids)
        with torch.inference_mode():
            score = float(model.relnet(scaled, graph)[
                ids.index(uid), class_index])
        expected = account["baseline"][f"{name}_score"]
        max_score_error = max(max_score_error, abs(score - expected))
        if abs(score - expected) > 1e-5:
            raise ValueError("graph closure score differs from full model")
        for effect in account["incident_relation_removal_effects"]:
            after = oracle.predict(uid, frozenset((effect["source_id"],)))
            if abs(after["margin"] - effect["after_removal"]["margin"]) > 1e-5:
                raise ValueError("graph intervention replay differs")
            replay_count += 1
    result = {
        "status": "verified_graph_closure_raw_sources_scores_and_interventions",
        "probe_sha256": sha256_file(probe_file),
        "checked_source_records": source_count,
        "checked_native_relation_records": graph_count,
        "checked_full_path_relation_interventions": replay_count,
        "maximum_relnet_score_difference": max_score_error,
    }
    output = run / "native_graph_closure_verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
