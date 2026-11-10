"""Compare a refined source coalition with same-owner, same-kind controls."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from verifiable_bot.aaai_fox8_raw_features import RAW_PROFILE_FIELDS
from verifiable_bot.fox8_atomic_sources import load_account_atoms, load_relation_atoms
from verifiable_bot.fox8_target_evidence_intervention import TargetEvidenceIntervention
from verifiable_bot.fox8_target_evidence_model import (
    HGTTargetFactSelector, HGTTargetRelationSelector, TargetEvidenceFox8,
)
from verifiable_bot.fox8_two_stage_inputs import ALL_FACTS, Fox8TwoStageInputs


def source_bucket(node: dict) -> tuple:
    if node["kind"] == "profile":
        return ("profile", node["account_id"],
                node.get("profile_value_type", "any"))
    if node["kind"] == "tweet":
        return ("tweet", node["account_id"],
                node.get("tweet_type", "any"),
                node.get("has_url", "any"))
    return ("relation", node["relation"], node["source_account_id"],
            node["target_account_id"])


def build_controls(source_nodes: list[dict], selected: frozenset[str],
                   count: int, seed: int) -> tuple[list[frozenset[str]], list[str]]:
    by_id = {node["source_id"]: node for node in source_nodes}
    if len(by_id) != len(source_nodes) or not selected.issubset(by_id):
        raise ValueError("Selected sources differ from the full provenance graph")
    chosen: dict[tuple, list[str]] = defaultdict(list)
    available: dict[tuple, list[str]] = defaultdict(list)
    for source_id in sorted(by_id):
        bucket = source_bucket(by_id[source_id])
        (chosen if source_id in selected else available)[bucket].append(source_id)
    fixed = []
    rng = random.Random(seed)
    controls = []
    seen = set()
    for _ in range(count * 100):
        control = set()
        for bucket, source_ids in chosen.items():
            pool = available[bucket]
            if len(pool) < len(source_ids):
                control.update(source_ids)
                fixed.extend(source_ids)
            else:
                control.update(rng.sample(pool, len(source_ids)))
        key = frozenset(control)
        if key not in seen and key != selected:
            seen.add(key)
            controls.append(key)
        if len(controls) == count:
            break
    if len(controls) != count:
        raise ValueError("Insufficient distinct matched controls")
    return controls, sorted(set(fixed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--coalition-report", type=Path, required=True)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--control-count", type=int, default=20)
    parser.add_argument("--sampling-seed", type=int, default=20261114)
    parser.add_argument("--matching", choices=("kind_owner", "semantic_shape"),
                        default="kind_owner")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.control_count < 1:
        raise ValueError("control count must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    raw = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    events = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
    base_path = ROOT / "output/aaai_veribot_fox8/seed0/model.pt"
    checkpoint_path = args.run / "model.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    report = json.loads(args.coalition_report.read_text(encoding="utf-8"))
    refined = json.loads(args.refinement.read_text(encoding="utf-8"))
    if (report["split"] != "dev" or
            report["checkpoint_sha256"] != sha256_file(checkpoint_path) or
            report["source_sha256"] != sha256_file(raw) or
            refined["checkpoint_sha256"] != report["checkpoint_sha256"] or
            refined["source_sha256"] != report["source_sha256"] or
            refined["coalition_report_sha256"] !=
            sha256_file(args.coalition_report) or
            refined["stage"] != "single_deletion_stable"):
        raise ValueError("Frozen model and source contracts differ")
    case = next((row for row in report["cases"]
                 if row["account_id"] == refined["account_id"]), None)
    if case is None or bool(case["predicted_bot"]) != bool(case["label"]):
        raise ValueError("Expected a correctly predicted dev account")
    full = frozenset().union(*(frozenset(values) for values in
                               case["coalition_audit"]["group_source_ids"].values()))
    selected = frozenset(refined["retained_source_ids"])
    if not selected.issubset(full):
        raise ValueError("Refined sources are outside the prediction dependency")
    x, y, split, ids, graph, pack = load_inputs()
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3).to(device).eval()
    hgt.load_state_dict(base["model_state"]["hgt"])
    bank = Fox8TwoStageInputs.build(
        load_account_atoms(raw, ids), load_relation_atoms(events, raw, ids), ids)
    relnet = GatedRelNet(len(ALL_FACTS), hidden_channels=128,
                         relation_names=tuple(bank.full_graph), num_layers=1,
                         dropout=0.3, out_channels=2)
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
    source_nodes = case["source_graph"]["source_nodes"]
    if args.matching == "semantic_shape":
        source_nodes = [dict(node) for node in source_nodes]
        for node in source_nodes:
            kind, position = oracle.lookup[node["source_id"]]
            if kind == "profile":
                owner = int(bank.profile_account[position])
                field = RAW_PROFILE_FIELDS[int(bank.profile_field[position])]
                value = bank.accounts[owner].profile[field]
                node["profile_value_type"] = type(value).__name__
            elif kind == "tweet":
                owner = int(bank.tweet_account[position])
                tweet = bank.accounts[owner].tweets[
                    position - int(bank.tweet_offsets[owner])]
                node["tweet_type"] = (
                    "retweet" if tweet.is_retweet else
                    "reply" if tweet.is_reply else
                    "quote" if tweet.is_quote else "original")
                node["has_url"] = bool(tweet.url_count)
    controls, fixed = build_controls(source_nodes, selected,
                                     args.control_count, args.sampling_seed)
    uid = refined["account_id"]
    threshold = float(refined["threshold"])
    buffer = float(refined["margin_buffer"])
    original_bot = bool(case["predicted_bot"])
    full_margin = oracle.predict(uid)["margin"]
    empty_margin = oracle.predict(uid, full)["margin"]
    if abs(full_margin - case["baseline_margin"]) > 1e-4:
        raise ValueError("Frozen full decision does not replay")

    def evaluate(sources: frozenset[str]) -> dict:
        kept = oracle.predict(uid, full - sources)["margin"]
        removed = oracle.predict(uid, sources)["margin"]
        if original_bot:
            passed = kept >= threshold + buffer and removed < threshold - buffer
            result = {"retained_only_margin": kept,
                      "full_without_set_margin": removed,
                      "passes_bidirectional_class_flip": passed}
        else:
            gain = empty_margin - kept
            loss = removed - full_margin
            passed = gain >= buffer and loss >= buffer
            result = {"retained_only_margin": kept,
                      "full_without_set_margin": removed,
                      "human_gain_over_empty": gain,
                      "human_loss_when_removed_from_full": loss,
                      "passes_directional_support": passed}
        return result

    selected_result = evaluate(selected)
    if not selected_result.get("passes_bidirectional_class_flip",
                               selected_result.get("passes_directional_support")):
        raise ValueError("Selected coalition no longer passes its criterion")
    cases = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, control in enumerate(controls):
        row = {"control_index": index, "source_ids": sorted(control),
               "source_count": len(control),
               "overlap_with_selected": len(control & selected),
               **evaluate(control)}
        cases.append(row)
        args.output.write_text(json.dumps({
            "account_id": uid, "predicted_bot": original_bot,
            "checkpoint_sha256": report["checkpoint_sha256"],
            "source_sha256": report["source_sha256"],
            "coalition_report_sha256": sha256_file(args.coalition_report),
            "refinement_sha256": sha256_file(args.refinement),
            "sampling_seed": args.sampling_seed,
            "matching": args.matching,
            "control_count": args.control_count,
            "fixed_source_ids": fixed,
            "threshold": threshold, "margin_buffer": buffer,
            "full_margin": full_margin, "empty_margin": empty_margin,
            "selected_source_ids": sorted(selected),
            "selected_result": selected_result,
            "completed_controls": len(cases), "controls": cases,
        }, indent=2) + "\n", encoding="utf-8")
    key = ("passes_bidirectional_class_flip" if original_bot else
           "passes_directional_support")
    print(json.dumps({"account_id": uid, "controls": len(cases),
                      "controls_passing": sum(row[key] for row in cases),
                      "fixed_sources": len(fixed)}), flush=True)


if __name__ == "__main__":
    main()
