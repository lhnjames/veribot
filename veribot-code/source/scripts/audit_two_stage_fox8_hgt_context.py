"""Certify raw-source reachability of the frozen HGT selector context."""

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
from train_aaai_veribot_fox8 import load_inputs, sha256_file
from verifiable_bot.fox8_atomic_sources import load_account_atoms, load_relation_atoms
from verifiable_bot.fox8_hgt_context_closure import (
    hgt_context_closure, verify_hgt_context_locality,
)
from verifiable_bot.fox8_two_stage_inputs import Fox8TwoStageInputs


def main() -> None:
    torch.set_num_threads(4)
    raw = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    events = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
    base_path = ROOT / "output/aaai_veribot_fox8/seed0/model.pt"
    run = ROOT / "output/two_stage_fox8_evidence/pilot_replay"
    group_probe = json.loads((run / "full_path_source_group_probe.json").read_text(
        encoding="utf-8"))
    graph_probe = json.loads((run / "native_graph_source_probe.json").read_text(
        encoding="utf-8"))
    if (group_probe["raw_source_sha256"] != sha256_file(raw) or
            graph_probe["raw_source_sha256"] != sha256_file(raw)):
        raise ValueError("HGT context source differs from probes")
    x, _, _, ids, graph, pack = load_inputs()
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3).eval()
    hgt.load_state_dict(base["model_state"]["hgt"])
    accounts = load_account_atoms(raw, ids)
    relations = load_relation_atoms(events, raw, ids)
    bank = Fox8TwoStageInputs.build(accounts, relations, ids)
    if any(not torch.equal(graph[("user", name, "user")], edges)
           for name, edges in bank.full_graph.items()):
        raise ValueError("HGT graph and raw relation graph differ")
    mean = np.asarray(base["input_contract"]["scaler_mean"])
    scale = np.asarray(base["input_contract"]["scaler_scale"])
    cases = []
    for account in group_probe["accounts"]:
        for name in ("human", "bot"):
            cases.append((account["account_id"], name,
                          account["class_source_closures"][name][
                              "receptive_field_accounts"]))
    for name, account in graph_probe["accounts"].items():
        cases.append((account["account_id"], name,
                      account["selected_class_relnet_closure"][
                          "receptive_field_accounts"]))
    output_cases = []
    for uid, name, seeds in cases:
        closure, restricted, nodes = hgt_context_closure(
            bank, graph, seeds)
        hgt_error = verify_hgt_context_locality(
            hgt, x, graph, restricted, nodes, seeds, bank.ids)
        positions = np.asarray(sorted(nodes), dtype=np.int64)
        raw_facts = np.asarray([bank.accounts[i].features() for i in positions],
                               dtype=np.float32)
        raw_scaled = np.clip((raw_facts - mean) / scale, -3, 3)
        feature_error = float(np.max(np.abs(raw_scaled - x[positions].numpy())))
        if feature_error > 1e-5:
            raise ValueError("HGT raw context features differ from frozen input")
        closure["target_account_id"] = uid
        closure["class_hypothesis"] = name
        closure["maximum_raw_hgt_input_difference"] = feature_error
        closure["maximum_local_hgt_embedding_difference"] = hgt_error
        output_cases.append(closure)
        print(json.dumps({
            "account": uid, "class": name, "seed_accounts": len(seeds),
            "context_accounts": closure["context_account_count"],
            "raw_tweets": closure["raw_tweet_source_count"],
            "maximum_hgt_embedding_difference": hgt_error,
        }), flush=True)
    output = run / "hgt_context_source_closure_audit.json"
    output.write_text(json.dumps({
        "status": "verified_two_layer_hgt_context_raw_source_reachability",
        "raw_source_sha256": sha256_file(raw),
        "hgt_context_checkpoint_sha256": sha256_file(base_path),
        "cases": output_cases,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": len(output_cases)}))


if __name__ == "__main__":
    main()
