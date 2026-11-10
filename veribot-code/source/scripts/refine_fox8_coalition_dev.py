"""Refine a Bot source-group coalition with exact raw-source interventions."""

from __future__ import annotations

import argparse
import hashlib
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
from verifiable_bot.aaai_fox8_raw_features import RAW_PROFILE_FIELDS
from verifiable_bot.fox8_atomic_sources import load_account_atoms, load_relation_atoms
from verifiable_bot.fox8_target_evidence_intervention import TargetEvidenceIntervention
from verifiable_bot.fox8_target_evidence_model import (
    HGTTargetFactSelector, HGTTargetRelationSelector, TargetEvidenceFox8,
)
from verifiable_bot.fox8_two_stage_inputs import ALL_FACTS, Fox8TwoStageInputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--coalition-report", type=Path, required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--minimal-set-index", type=int, default=0)
    parser.add_argument("--union-sufficient-seeds", action="store_true",
                        help="Merge observed alternative sufficient paths before refinement")
    parser.add_argument("--margin-buffer", type=float, default=0.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.margin_buffer < 0:
        raise ValueError("margin buffer must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
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
        raise ValueError("source contract differs from frozen model")
    coalition_file = json.loads(args.coalition_report.read_text(encoding="utf-8"))
    if coalition_file["checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise ValueError("coalition and frozen model differ")
    case = next((x for x in coalition_file["cases"]
                 if x["account_id"] == args.account_id), None)
    if case is None or not case["predicted_bot"]:
        raise ValueError("account must be a verified Bot coalition case")
    coarse = case["coalition_audit"]
    if args.union_sufficient_seeds:
        seeds = coarse.get("observed_nondominated_sufficient_group_sets", [])
        if not seeds or any(not row["groups"] for row in seeds):
            raise ValueError("No nonempty alternative sufficient paths to merge")
        chosen = {name for row in seeds for name in row["groups"]}
        names = [name for name in coarse["group_names"] if name in chosen]
    elif "searched_bidirectional_group_sets" in coarse:
        names = coarse["searched_bidirectional_group_sets"][
            args.minimal_set_index]["groups"]
    else:
        names = coarse["minimal_sufficient_group_sets"][args.minimal_set_index]
    group_ids = coarse["group_source_ids"]
    full = frozenset().union(*(frozenset(v) for v in group_ids.values()))
    retained = frozenset().union(*(frozenset(group_ids[name]) for name in names))
    x, y, split, ids, graph, pack = load_inputs()
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3).to(device).eval()
    hgt.load_state_dict(base["model_state"]["hgt"])
    bank = Fox8TwoStageInputs.build(
        load_account_atoms(raw, ids), load_relation_atoms(events, raw, ids), ids)
    relnet = GatedRelNet(len(ALL_FACTS), hidden_channels=128,
                         relation_names=tuple(bank.full_graph),
                         num_layers=1, dropout=0.3, out_channels=2)
    selector_type = (HGTTargetRelationSelector if saved.get("relation_gates", False)
                     else HGTTargetFactSelector)
    model = TargetEvidenceFox8(
        selector_type(context_dim=128), relnet,
        torch.tensor(saved["scaler_mean"]),
        torch.tensor(saved["scaler_scale"]),
        centered_evidence=saved.get("centered_evidence", False),
        sample_training_gates=saved.get("sample_training_gates", True),
        bot_only_support=saved.get("bot_only_support", False),
        selected_neighbor_sources=saved.get(
            "selected_neighbor_sources", False)).to(device).eval()
    model.load_state_dict(saved["model_state"])
    oracle = TargetEvidenceIntervention(
        hgt=hgt, model=model, batch=bank,
        base_mean=np.asarray(base["input_contract"]["scaler_mean"]),
        base_scale=np.asarray(base["input_contract"]["scaler_scale"]))
    threshold = float(saved["dev_logit_threshold"])
    if args.verify_only:
        report = json.loads(args.output.read_text(encoding="utf-8"))
        if (report["stage"] != "single_deletion_stable" or
                report["checkpoint_sha256"] != sha256_file(checkpoint_path) or
                report["source_sha256"] != sha256_file(raw) or
                float(report.get("margin_buffer", 0.0)) != args.margin_buffer):
            raise ValueError("final refinement contract differs")
        final_sources = frozenset(report["retained_source_ids"])
        if abs(oracle.predict(args.account_id, full - final_sources)["margin"] -
               report["retained_margin"]) > 1e-4:
            raise ValueError("refined coalition does not replay")
        removed_from_full = oracle.predict(args.account_id,
                                           final_sources)["margin"]
        report["margin_after_removing_refined_set_from_full"] = removed_from_full
        report["refined_set_collectively_necessary_in_full"] = (
            removed_from_full < threshold - args.margin_buffer)
        args.output.write_text(json.dumps(report, indent=2) + "\n",
                               encoding="utf-8")
        print(json.dumps({"account_id": args.account_id,
                          "refined_source_count": len(final_sources),
                          "removed_from_full_margin": removed_from_full,
                          "collectively_necessary_in_full":
                              report["refined_set_collectively_necessary_in_full"]}),
              flush=True)
        return

    def score(sources: frozenset[str]) -> float:
        return oracle.predict(args.account_id, full - sources)["margin"]

    initial_margin = score(retained)
    recorded = next((row["margin"] for row in coarse["coalitions"]
                     if set(row["groups"]) == set(names)), None)
    if recorded is not None and abs(initial_margin - recorded) > 1e-4:
        raise ValueError("coarse coalition did not replay exactly")
    if len(full) != coarse["full_dependency_source_count"]:
        raise ValueError("coarse dependency source IDs are incomplete")
    initial_removed_margin = oracle.predict(args.account_id, retained)["margin"]
    group_expansion_trace = []
    if args.union_sufficient_seeds:
        while (initial_margin < threshold + args.margin_buffer or
               initial_removed_margin >= threshold - args.margin_buffer):
            expansions = []
            for name in coarse["ranked_by_full_context_removal_loss"]:
                if name in names:
                    continue
                candidate = retained | frozenset(group_ids[name])
                kept = score(candidate)
                removed = oracle.predict(args.account_id, candidate)["margin"]
                group_expansion_trace.append({
                    "added_group": name, "retained_source_count": len(candidate),
                    "retained_margin": kept,
                    "margin_after_removal_from_full": removed})
                if kept >= threshold + args.margin_buffer:
                    expansions.append((removed, len(candidate), name, candidate,
                                       kept))
            if not expansions:
                raise ValueError("No larger source union preserves Bot with margin")
            removed, _, name, retained, kept = min(expansions)
            names.append(name)
            initial_margin, initial_removed_margin = kept, removed
    if (initial_margin < threshold + args.margin_buffer or
            initial_removed_margin >= threshold - args.margin_buffer):
        raise ValueError("coarse coalition is not bidirectional with margin")
    trace = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(stage: str) -> None:
        payload = {
            "account_id": args.account_id,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "source_sha256": sha256_file(raw),
            "coalition_report_sha256": sha256_file(args.coalition_report),
            "threshold": threshold,
            "margin_buffer": args.margin_buffer,
            "coarse_group_names": names,
            "search_origin": ("union_of_observed_sufficient_paths"
                              if args.union_sufficient_seeds else
                              "single_group_search_candidate"),
            "group_expansion_trace": group_expansion_trace,
            "coarse_source_count": sum(len(group_ids[name]) for name in names),
            "coarse_margin": initial_margin,
            "coarse_margin_after_removal_from_full": initial_removed_margin,
            "stage": stage,
            "retained_source_ids": sorted(retained),
            "retained_source_count": len(retained),
            "retained_margin": score(retained),
            "trace": trace,
        }
        if stage == "single_deletion_stable":
            removed_from_full = oracle.predict(args.account_id, retained)["margin"]
            payload["margin_after_removing_refined_set_from_full"] = removed_from_full
            payload["refined_set_collectively_necessary_in_full"] = (
                removed_from_full < threshold - args.margin_buffer)
            if not payload["refined_set_collectively_necessary_in_full"]:
                raise ValueError("final coalition lost full-context necessity")
            locators = []
            for source_id in sorted(retained):
                kind, position = oracle.lookup[source_id]
                locator = {"source_id": source_id, "kind": kind}
                if kind == "profile":
                    owner = int(bank.profile_account[position])
                    field = RAW_PROFILE_FIELDS[int(bank.profile_field[position])]
                    value = json.dumps(bank.accounts[owner].profile[field],
                                       ensure_ascii=False, sort_keys=True,
                                       default=str)
                    locator.update({
                        "account_id": bank.ids[owner],
                        "raw_line": bank.accounts[owner].line_number,
                        "field": field,
                        "value_sha256": hashlib.sha256(
                            value.encode("utf-8")).hexdigest(),
                    })
                elif kind == "tweet":
                    owner = int(bank.tweet_account[position])
                    tweet = bank.accounts[owner].tweets[
                        position - int(bank.tweet_offsets[owner])]
                    locator.update({
                        "account_id": bank.ids[owner],
                        "raw_line": tweet.line_number,
                        "tweet_ordinal": tweet.ordinal,
                        "tweet_id": tweet.tweet_id,
                        "record_sha256": tweet.record_sha256,
                    })
                else:
                    relation = bank.relations[position]
                    locator.update({
                        "source_account_id": relation.source_account_id,
                        "target_account_id": relation.target_account_id,
                        "relation": relation.relation,
                        "raw_locators": relation.raw_locators,
                    })
                locators.append(locator)
            payload["raw_source_locators"] = locators
        args.output.write_text(json.dumps(payload, indent=2) + "\n",
                               encoding="utf-8")

    def try_remove(stage: str, part_name: str,
                   part: frozenset[str]) -> bool:
        nonlocal retained
        active = retained & part
        if not active:
            return False
        candidate = retained - active
        margin = score(candidate)
        sufficient = margin >= threshold + args.margin_buffer
        full_removal_margin = (oracle.predict(args.account_id, candidate)["margin"]
                               if sufficient else None)
        necessary = (full_removal_margin is not None and
                     full_removal_margin < threshold - args.margin_buffer)
        removed = sufficient and necessary
        trace.append({"stage": stage, "part": part_name,
                      "tested_source_count": len(active),
                      "margin_after_removal": margin,
                      "candidate_sufficient": sufficient,
                      "candidate_removed_from_full_margin": full_removal_margin,
                      "candidate_collectively_necessary_in_full": necessary,
                      "accepted": removed})
        if removed:
            retained = candidate
        save(stage)
        return removed

    profile = [source_id for source_id in retained
               if oracle.lookup[source_id][0] == "profile"]
    tweets = [source_id for source_id in retained
              if oracle.lookup[source_id][0] == "tweet"]
    other = [source_id for source_id in retained
             if oracle.lookup[source_id][0] not in ("profile", "tweet")]
    for source_id in sorted(profile):
        try_remove("profile_field", source_id, frozenset((source_id,)))
    tweet_types: dict[str, set[str]] = {}
    for source_id in tweets:
        _, position = oracle.lookup[source_id]
        owner = int(bank.tweet_account[position])
        tweet = bank.accounts[owner].tweets[
            position - int(bank.tweet_offsets[owner])]
        kind = ("retweet" if tweet.is_retweet else
                "reply" if tweet.is_reply else
                "quote" if tweet.is_quote else "original")
        tweet_types.setdefault(kind, set()).add(source_id)
    for kind, sources in sorted(tweet_types.items()):
        try_remove("tweet_type", kind, frozenset(sources))
    for source_id in sorted(other):
        try_remove("other_atom", source_id, frozenset((source_id,)))
    # Revisit individual retained atoms until no single deletion preserves Bot.
    passes = 0
    while True:
        passes += 1
        changed = False
        for source_id in sorted(retained):
            changed |= try_remove("atomic_pass_" + str(passes), source_id,
                                  frozenset((source_id,)))
        if not changed:
            break
    save("single_deletion_stable")
    print(json.dumps({"account_id": args.account_id,
                      "coarse_count": sum(len(group_ids[name]) for name in names),
                      "retained_count": len(retained),
                      "retained_margin": score(retained),
                      "threshold": threshold,
                      "atomic_passes": passes,
                      "queries": len(trace)}), flush=True)


if __name__ == "__main__":
    main()
