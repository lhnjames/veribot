"""Greedily refine Human score support with exact source interventions."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--margin-buffer", type=float, default=0.05)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.margin_buffer <= 0:
        raise ValueError("Human directional support needs a positive margin buffer")
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
        raise ValueError("Raw source contract differs from frozen model")
    report = json.loads(args.coalition_report.read_text(encoding="utf-8"))
    if (report["checkpoint_sha256"] != sha256_file(checkpoint_path) or
            report["source_sha256"] != sha256_file(raw) or
            report["split"] != "dev"):
        raise ValueError("Coalition report is not the frozen dev model")
    case = next((row for row in report["cases"]
                 if row["account_id"] == args.account_id), None)
    if case is None or case["predicted_bot"] or case["label"] != 0:
        raise ValueError("Expected a correctly classified Human dev account")
    audit = case["coalition_audit"]
    names = audit["group_names"]
    group_ids = audit["group_source_ids"]
    full_mask = (1 << len(names)) - 1
    queried = {row["mask"]: row for row in audit["coalitions"]}
    if full_mask not in queried or 0 not in queried:
        raise ValueError("Missing full or empty input query")
    full_margin = float(queried[full_mask]["margin"])
    empty_margin = float(queried[0]["margin"])
    threshold = float(audit["threshold"])
    candidates = []
    for mask, row in queried.items():
        complement = queried.get(full_mask ^ mask)
        if mask == 0 or complement is None or row["margin"] >= threshold:
            continue
        gain = empty_margin - row["margin"]
        loss = complement["margin"] - full_margin
        if gain >= args.margin_buffer and loss >= args.margin_buffer:
            candidates.append((row["source_count"], len(row["groups"]),
                               -min(gain, loss), mask))
    if not candidates:
        raise ValueError("Search report contains no buffered Human support set")
    _, _, _, initial_mask = min(candidates)
    selected_groups = queried[initial_mask]["groups"]
    full = frozenset().union(*(frozenset(values) for values in group_ids.values()))
    retained = frozenset().union(*(frozenset(group_ids[name])
                                   for name in selected_groups))
    source_nodes = {node["source_id"]: node
                    for node in case["source_graph"]["source_nodes"]}
    if (len(source_nodes) != audit["full_dependency_source_count"] or
            not retained.issubset(source_nodes)):
        raise ValueError("Source graph does not cover the chosen Human coalition")

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
    if full - frozenset(oracle.lookup):
        raise ValueError("Coalition uses unknown raw sources")

    def keep_margin(sources: frozenset[str]) -> float:
        return oracle.predict(args.account_id, full - sources)["margin"]

    if (abs(keep_margin(retained) - queried[initial_mask]["margin"]) > 1e-4 or
            abs(oracle.predict(args.account_id, retained)["margin"] -
                queried[full_mask ^ initial_mask]["margin"]) > 1e-4 or
            abs(keep_margin(frozenset()) - empty_margin) > 1e-4 or
            abs(oracle.predict(args.account_id)["margin"] - full_margin) > 1e-4):
        raise ValueError("Frozen Human support queries do not replay")

    trace = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(stage: str) -> None:
        kept = keep_margin(retained)
        removed = oracle.predict(args.account_id, retained)["margin"]
        payload = {
            "account_id": args.account_id,
            "criterion": "directional_human_margin_support",
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "source_sha256": sha256_file(raw),
            "coalition_report_sha256": sha256_file(args.coalition_report),
            "threshold": threshold,
            "margin_buffer": args.margin_buffer,
            "initial_groups": selected_groups,
            "initial_source_count": queried[initial_mask]["source_count"],
            "empty_margin": empty_margin,
            "full_margin": full_margin,
            "stage": stage,
            "retained_source_ids": sorted(retained),
            "retained_source_count": len(retained),
            "retained_only_margin": kept,
            "full_without_retained_margin": removed,
            "human_gain_over_empty": empty_margin - kept,
            "human_loss_when_removed_from_full": removed - full_margin,
            "trace": trace,
        }
        if stage == "single_deletion_stable":
            if (payload["human_gain_over_empty"] < args.margin_buffer or
                    payload["human_loss_when_removed_from_full"] <
                    args.margin_buffer):
                raise ValueError("Final Human support lost the margin buffer")
            payload["raw_source_locators"] = [source_nodes[sid]
                                              for sid in sorted(retained)]
        args.output.write_text(json.dumps(payload, ensure_ascii=False,
                                          indent=2) + "\n", encoding="utf-8")

    passes = 0
    while True:
        passes += 1
        changed = False
        for source_id in sorted(retained):
            candidate = retained - {source_id}
            kept = keep_margin(candidate)
            gain = empty_margin - kept
            removed = (oracle.predict(args.account_id, candidate)["margin"]
                       if gain >= args.margin_buffer else None)
            loss = (removed - full_margin if removed is not None else None)
            accepted = (loss is not None and loss >= args.margin_buffer)
            trace.append({"pass": passes, "source_id": source_id,
                          "candidate_source_count": len(candidate),
                          "candidate_retained_margin": kept,
                          "candidate_human_gain_over_empty": gain,
                          "candidate_full_without_margin": removed,
                          "candidate_human_loss_when_removed": loss,
                          "accepted": accepted})
            if accepted:
                retained = candidate
                changed = True
            save(f"atomic_pass_{passes}")
        if not changed:
            break
    save("single_deletion_stable")
    print(json.dumps({"account_id": args.account_id,
                      "initial_sources": queried[initial_mask]["source_count"],
                      "retained_sources": len(retained),
                      "human_gain_over_empty": empty_margin - keep_margin(retained),
                      "human_loss_when_removed_from_full":
                          oracle.predict(args.account_id, retained)["margin"] -
                          full_margin, "passes": passes,
                      "queries": len(trace)}), flush=True)


if __name__ == "__main__":
    main()
