"""Test forward sufficiency and removal necessity on identical raw sources.

For each predeclared diagnostic account, this script runs the frozen complete
HGT-selector -> evidence-view RelNet pipeline after keeping or withholding
the exact emitted source sets. Both class hypotheses are included. The two
candidate sets differ in scope: selected RelNet sources alone, or those sources
plus the full two-layer HGT selector context dependency closure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
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
from verifiable_bot.fox8_hgt_context_closure import hgt_context_closure
from verifiable_bot.fox8_two_stage_certificate import relnet_source_closure
from verifiable_bot.fox8_two_stage_inputs import ALL_FACTS, Fox8TwoStageInputs
from verifiable_bot.fox8_two_stage_intervention import TwoStageIntervention
from verifiable_bot.fox8_two_stage_model import HGTSourceSelector, TwoStageFox8


def _summary(result: dict, threshold: float) -> dict:
    return {**result, "predicted_bot": result["margin"] >= threshold}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--account-id", action="append")
    parser.add_argument("--sample-per-class", type=int, default=0)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--sampling-seed", type=int, default=20261109)
    parser.add_argument("--output-name", default="sufficiency_necessity_diagnostic.json")
    parser.add_argument("--random-controls", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20261109)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but no GPU is available")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    raw = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    events = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
    base_path = ROOT / "output/aaai_veribot_fox8/seed0/model.pt"
    checkpoint_path = args.run / "model.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (saved["source_sha256"] != sha256_file(raw) or
            saved["event_index_sha256"] != sha256_file(events) or
            saved["hgt_context_checkpoint_sha256"] != sha256_file(base_path)):
        raise ValueError("model/source hash contract differs")
    threshold = float(saved["dev_logit_threshold"])
    x, y, split, ids, graph, pack = load_inputs()
    if bool(args.account_id) == bool(args.sample_per_class):
        raise ValueError("choose explicit account IDs or a per-class sample")
    if args.sample_per_class:
        sampler = random.Random(args.sampling_seed)
        selected_ids = tuple(
            ids[int(i)] for cls in (0, 1)
            for i in sampler.sample(
                [int(j) for j in split[args.split] if int(y[j]) == cls],
                args.sample_per_class))
    else:
        selected_ids = tuple(dict.fromkeys(args.account_id))
    evaluation_ids = {ids[int(i)] for i in split[args.split]}
    if any(uid not in evaluation_ids for uid in selected_ids):
        raise ValueError(f"diagnostic account is not in the {args.split} set")
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3).eval().to(device)
    hgt.load_state_dict(base["model_state"]["hgt"])
    accounts = load_account_atoms(raw, ids)
    relations = load_relation_atoms(events, raw, ids)
    bank = Fox8TwoStageInputs.build(accounts, relations, ids)
    relnet = GatedRelNet(len(ALL_FACTS), hidden_channels=128,
                         relation_names=tuple(bank.full_graph),
                         num_layers=2, dropout=0.3, out_channels=2)
    model = TwoStageFox8(HGTSourceSelector(context_dim=128), relnet,
                         torch.tensor(saved["scaler_mean"]),
                         torch.tensor(saved["scaler_scale"])).eval().to(device)
    model.load_state_dict(saved["model_state"])
    oracle = TwoStageIntervention(
        hgt=hgt, model=model, batch=bank,
        base_mean=np.asarray(base["input_contract"]["scaler_mean"]),
        base_scale=np.asarray(base["input_contract"]["scaler_scale"]))
    with torch.inference_mode():
        context = hgt({"user": x.to(device)}, {
            key: edges.to(device) for key, edges in graph.items()
        })["user_emb"]
        emitted = model(context, bank)
    source_universe = frozenset(oracle.lookup)
    control_rng = random.Random(args.random_seed)
    cases = []
    output = args.run / args.output_name
    if output.parent != args.run or output.suffix != ".json":
        raise ValueError("output name must be a JSON file directly inside run directory")
    header = {
        "scope": f"predeclared {args.split} diagnostic accounts; exact model dependence, not identity truth",
        "split": args.split,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "raw_sha256": sha256_file(raw),
        "threshold_selected_on_dev": threshold,
        "device": str(device),
        "sampling_seed": args.sampling_seed if args.sample_per_class else None,
        "sample_per_class": args.sample_per_class,
        "selected_account_ids": selected_ids,
        "random_controls_per_account": args.random_controls,
        "random_seed": args.random_seed if args.random_controls else None,
    }
    for uid in selected_ids:
        closure = {name: relnet_source_closure(
            bank, uid, emitted["hard_selection"], class_index)
            for class_index, name in ((0, "human"), (1, "bot"))}
        stage2 = frozenset(record["source_id"] for item in closure.values()
                           for record in item["source_records"])
        seed_accounts = sorted({account_id for item in closure.values()
                                for account_id in item["receptive_field_accounts"]})
        hgt_closure, _, context_nodes = hgt_context_closure(
            bank, graph, seed_accounts)
        full = set(stage2)
        for i in context_nodes:
            account = bank.accounts[i]
            full.update(account.source_id("profile", field)
                        for field in account.profile_field_ids)
            full.update(account.source_id("tweet", ordinal)
                        for ordinal in account.tweet_ordinals)
        full.update(hgt_closure["native_relation_source_ids"])
        full = frozenset(full)
        if full - source_universe:
            raise ValueError("dependency closure contains an unknown source")
        baseline = _summary(oracle.predict(uid), threshold)
        original_class = baseline["predicted_bot"]
        scores = {
            "stage2_only": {
                "keep": _summary(oracle.predict(uid, source_universe - stage2),
                                 threshold),
                "remove": _summary(oracle.predict(uid, stage2), threshold),
                "source_count": len(stage2),
            },
            "stage2_plus_hgt_context": {
                "keep": _summary(oracle.predict(uid, source_universe - full),
                                 threshold),
                "remove": _summary(oracle.predict(uid, full), threshold),
                "source_count": len(full),
                "hgt_context_account_count": hgt_closure["context_account_count"],
                "hgt_context_inventory_sha256": hgt_closure[
                    "raw_source_inventory_sha256"],
            },
        }
        for scope in scores.values():
            scope["sufficient_for_original_class"] = (
                scope["keep"]["predicted_bot"] == original_class)
            scope["collectively_necessary_for_original_class"] = (
                scope["remove"]["predicted_bot"] != original_class)
        case = {"account_id": uid, "baseline": baseline,
                "two_hypotheses_included": True,
                "candidate_source_counts": {
                    "selected_relnet": len(stage2),
                    "with_hgt_context": len(full),
                    "all_available": len(source_universe)},
                "tests": scores}
        if args.random_controls:
            selected_by_type = {kind: sum(oracle.lookup[source][0] == kind
                                          for source in stage2)
                                for kind in ("profile", "tweet", "relation")}
            pool_by_type = {kind: sorted(source for source in full
                                         if oracle.lookup[source][0] == kind)
                            for kind in selected_by_type}
            controls = []
            for repeat in range(args.random_controls):
                random_set = frozenset(
                    source for kind, n in selected_by_type.items()
                    for source in control_rng.sample(pool_by_type[kind], n))
                control_keep = _summary(
                    oracle.predict(uid, source_universe - random_set), threshold)
                control_remove = _summary(oracle.predict(uid, random_set), threshold)
                controls.append({
                    "repeat": repeat,
                    "same_type_source_counts": selected_by_type,
                    "overlap_with_selected": len(random_set & stage2),
                    "keep": control_keep,
                    "remove": control_remove,
                    "sufficient_for_original_class":
                        control_keep["predicted_bot"] == original_class,
                    "collectively_necessary_for_original_class":
                        control_remove["predicted_bot"] != original_class,
                })
            case["random_same_type_within_hgt_context_controls"] = controls
        cases.append(case)
        output.write_text(json.dumps({**header, "completed_cases": len(cases),
                                      "cases": cases}, ensure_ascii=False,
                                     indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"account_id": uid, "original_bot": original_class,
                          "stage2_sufficient": scores["stage2_only"][
                              "sufficient_for_original_class"],
                          "stage2_necessary": scores["stage2_only"][
                              "collectively_necessary_for_original_class"],
                          "full_sufficient": scores[
                              "stage2_plus_hgt_context"][
                                  "sufficient_for_original_class"],
                          "full_necessary": scores[
                              "stage2_plus_hgt_context"][
                                  "collectively_necessary_for_original_class"]}),
              flush=True)
    print(json.dumps({"output": str(output), "case_count": len(cases)}))


if __name__ == "__main__":
    main()
