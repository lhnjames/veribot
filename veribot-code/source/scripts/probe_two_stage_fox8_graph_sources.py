"""Probe native graph-source effects for both Fox8 prediction classes."""

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
from verifiable_bot.fox8_atomic_sources import load_account_atoms, load_relation_atoms
from verifiable_bot.fox8_two_stage_inputs import ALL_FACTS, Fox8TwoStageInputs
from verifiable_bot.fox8_two_stage_certificate import relnet_source_closure
from verifiable_bot.fox8_two_stage_intervention import TwoStageIntervention
from verifiable_bot.fox8_two_stage_model import HGTSourceSelector, TwoStageFox8


def main() -> None:
    torch.set_num_threads(4)
    run = ROOT / "output/two_stage_fox8_evidence/pilot_replay"
    raw = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    events = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
    base_path = ROOT / "output/aaai_veribot_fox8/seed0/model.pt"
    checkpoint = run / "model.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["source_sha256"] != sha256_file(raw) or saved[
            "event_index_sha256"] != sha256_file(events):
        raise ValueError("source or relation index differs from checkpoint")
    evaluation = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    threshold = float(evaluation["threshold_selected_on_dev_only"])
    x, _, split, ids, graph, pack = load_inputs()
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3).eval()
    hgt.load_state_dict(base["model_state"]["hgt"])
    accounts = load_account_atoms(raw, ids)
    relations = load_relation_atoms(events, raw, ids)
    bank = Fox8TwoStageInputs.build(accounts, relations, ids)
    relnet = GatedRelNet(len(ALL_FACTS), hidden_channels=128,
                         relation_names=tuple(bank.full_graph),
                         num_layers=2, dropout=0.3, out_channels=2)
    model = TwoStageFox8(HGTSourceSelector(context_dim=128), relnet,
                         torch.tensor(saved["scaler_mean"]),
                         torch.tensor(saved["scaler_scale"])).eval()
    model.load_state_dict(saved["model_state"])
    oracle = TwoStageIntervention(
        hgt=hgt, model=model, batch=bank,
        base_mean=np.asarray(base["input_contract"]["scaler_mean"]),
        base_scale=np.asarray(base["input_contract"]["scaler_scale"]))
    with torch.inference_mode():
        context = hgt({"user": x}, graph)["user_emb"]
        full = model(context, bank)
    test_ids = {ids[int(i)] for i in split["test"]}
    choices = {}
    for row in sorted(report["test_predictions"],
                      key=lambda item: (abs(item["bot_score"] - item[
                          "human_score"] - threshold), item["account_id"])):
        uid = row["account_id"]
        if uid not in test_ids:
            continue
        predicted_bot = row["bot_score"] - row["human_score"] >= threshold
        name = "bot" if predicted_bot else "human"
        if name in choices:
            continue
        closure = relnet_source_closure(bank, uid, full["hard_selection"],
                                        int(predicted_bot))
        if closure["source_counts"]["native_relation_event"] == 0:
            continue
        incident = [item for item in relations if item.source_account_id == uid
                    or item.target_account_id == uid]
        baseline = oracle.predict(uid)
        if abs(baseline["margin"] - (row["bot_score"] - row[
                "human_score"])) > 1e-5:
            raise ValueError("graph probe baseline differs")
        effects = []
        for item in incident:
            after = oracle.predict(uid, frozenset((item.source_id,)))
            effects.append({
                "source_id": item.source_id,
                "relation": item.relation,
                "source_account_id": item.source_account_id,
                "target_account_id": item.target_account_id,
                "raw_locators": [{"line": line, "tweet": ordinal, "field": field}
                                 for line, ordinal, field in item.raw_locators],
                "after_removal": after,
                "margin_change": after["margin"] - baseline["margin"],
                "changes_class": ((after["margin"] >= threshold) != predicted_bot),
            })
        choices[name] = {
            "account_id": uid, "predicted_class": name,
            "label_for_evaluation_only": row["label"],
            "baseline": baseline,
            "selected_class_relnet_closure": closure,
            "incident_native_relation_count": len(incident),
            "incident_relation_removal_effects": effects,
        }
        print(json.dumps({"class": name, "account_id": uid,
                          "selected_graph_sources": closure[
                              "source_counts"]["native_relation_event"],
                          "incident": len(incident)}), flush=True)
        if len(choices) == 2:
            break
    if set(choices) != {"human", "bot"}:
        raise ValueError("could not find native graph examples for both classes")
    output = run / "native_graph_source_probe.json"
    output.write_text(json.dumps({
        "scope": "nearest-boundary fixed-test account per predicted class with a selected native relation in the two-hop RelNet closure",
        "checkpoint_sha256": sha256_file(checkpoint),
        "raw_source_sha256": sha256_file(raw),
        "threshold_dev_only": threshold,
        "accounts": choices,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
