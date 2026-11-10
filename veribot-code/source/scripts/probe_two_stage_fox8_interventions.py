"""Find source groups that change the trained two-stage Fox8 decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
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
from verifiable_bot.source_coalition_search import find_budgeted_flip_groups

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
BASE = ROOT / "output/aaai_veribot_fox8/seed0/model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=ROOT / "output/two_stage_fox8_evidence/pilot_replay")
    parser.add_argument("--max-evaluations", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint_path = args.run / "model.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy = json.loads((args.run / "evaluation.json").read_text(encoding="utf-8"))
    report = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    threshold = float(policy["threshold_selected_on_dev_only"])
    if (saved["source_sha256"] != sha256_file(RAW) or
            saved["event_index_sha256"] != sha256_file(EVENTS) or
            saved["hgt_context_checkpoint_sha256"] != sha256_file(BASE) or
            report["checkpoint_sha256"] != sha256_file(checkpoint_path)):
        raise ValueError("model, source, or run report differs")
    x, _, split, ids, graph, pack = load_inputs()
    base = torch.load(BASE, map_location="cpu", weights_only=False)
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3).eval()
    hgt.load_state_dict(base["model_state"]["hgt"])
    accounts = load_account_atoms(RAW, ids)
    relations = load_relation_atoms(EVENTS, RAW, ids)
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
        full_context = hgt({"user": x}, graph)["user_emb"]
        full_result = model(full_context, bank)
    n_tweets = pd.read_csv(ROOT / "data/processed/aaai_veribot/fox8_23/features.csv")
    counts = dict(zip(n_tweets["user_id"].astype(str), n_tweets["n_tweets"]))
    eligible = []
    test_ids = {ids[int(i)] for i in split["test"]}
    for row in report["test_predictions"]:
        uid = row["account_id"]
        if uid not in test_ids or counts[uid] > 20:
            continue
        margin = float(row["bot_score"] - row["human_score"])
        predicted_bot = margin >= threshold
        eligible.append((abs(margin - threshold), uid, predicted_bot, row, margin))
    chosen = [min((item for item in eligible if item[2] == direction),
                  key=lambda item: (item[0], item[1]))
              for direction in (False, True)]
    results = []
    for _, uid, predicted_bot, saved_row, saved_margin in chosen:
        baseline = oracle.predict(uid)
        if abs(baseline["margin"] - saved_margin) > 1e-5:
            raise ValueError("reconstructed baseline differs from saved prediction")
        candidates = oracle.target_source_ids(uid)
        search = find_budgeted_flip_groups(
            candidates, lambda removed: oracle.predict(uid, removed)["margin"],
            threshold, max_evaluations=args.max_evaluations)
        position = ids.index(uid)
        if abs(float(full_result["logits"][position, 1] -
                     full_result["logits"][position, 0]) - baseline["margin"]) > 1e-5:
            raise ValueError("source intervention baseline differs from ordinary forward")
        closures = {
            name: relnet_source_closure(bank, uid,
                                        full_result["hard_selection"], class_index)
            for class_index, name in ((0, "human"), (1, "bot"))
        }
        views = {}
        for class_index, name in ((0, "human"), (1, "bot")):
            selected = full_result["hard_selection"]
            facts = bank.exact_view(selected["profile"][:, class_index],
                                    selected["tweet"][:, class_index])[position]
            views[name] = dict(zip(ALL_FACTS, (float(value) for value in facts)))
        results.append({
            "account_id": uid,
            "selection_rule": "closest saved decision margin per predicted class among fixed-test accounts with at most 20 source tweets",
            "label_for_evaluation_only": saved_row["label"],
            "predicted_bot": predicted_bot,
            "source_tweet_count": int(counts[uid]),
            "baseline": baseline,
            "candidate_source_count": len(candidates),
            "class_source_closures": closures,
            "target_selected_view_facts": views,
            "search": search,
        })
        print(json.dumps({"account_id": uid, "predicted_bot": predicted_bot,
                          "candidate_sources": len(candidates),
                          "evaluations": search["unique_evaluations"],
                          "solutions": len(search["solutions"])}), flush=True)
    output = args.run / "full_path_source_group_probe.json"
    output.write_text(json.dumps({
        "scope": "two preselected fixed-test boundary accounts; model-level source-channel interventions",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "raw_source_sha256": sha256_file(RAW),
        "threshold_dev_only": threshold,
        "max_evaluations_per_account": args.max_evaluations,
        "accounts": results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
