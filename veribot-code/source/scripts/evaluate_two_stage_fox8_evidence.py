"""Audit dev calibration and selected-only replay of the two-stage Fox8 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
BASE = ROOT / "output/aaai_veribot_fox8/seed0/model.pt"
PILOT = ROOT / "output/two_stage_fox8_evidence/pilot_gate0/model.pt"
OUTPUT = ROOT / "output/two_stage_fox8_evidence/pilot_gate0/evaluation.json"


def classification_metrics(margin: np.ndarray, labels: np.ndarray,
                           indices: np.ndarray, threshold: float) -> dict:
    truth = labels[indices].astype(int)
    predicted = margin[indices] >= threshold
    return {"n": len(indices),
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, average="macro")),
            "auc": float(roc_auc_score(truth, margin[indices]))}


def dev_threshold(margin: np.ndarray, labels: np.ndarray,
                  dev: np.ndarray) -> float:
    values = sorted(set(float(v) for v in margin[dev]))
    candidates = [values[0] - 1, *[(a + b) / 2 for a, b in
                                    zip(values[:-1], values[1:])], values[-1] + 1]
    return max(candidates, key=lambda threshold: (
        classification_metrics(margin, labels, dev, threshold)["macro_f1"],
        classification_metrics(margin, labels, dev, threshold)["accuracy"],
        -abs(threshold)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PILOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    output_path = args.output or checkpoint_path.with_name("evaluation.json")
    torch.set_num_threads(4)
    x, y, split, ids, graph, pack = load_inputs()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (saved["source_sha256"] != sha256_file(RAW) or
            saved["event_index_sha256"] != sha256_file(EVENTS) or
            saved["hgt_context_checkpoint_sha256"] != sha256_file(BASE)):
        raise ValueError("two-stage checkpoint differs from source contract")
    base = torch.load(BASE, map_location="cpu", weights_only=False)
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3).eval()
    hgt.load_state_dict(base["model_state"]["hgt"])
    accounts = load_account_atoms(RAW, ids)
    relations = load_relation_atoms(EVENTS, RAW, ids)
    bank = Fox8TwoStageInputs.build(accounts, relations, ids)
    if saved["feature_names"] != list(ALL_FACTS):
        raise ValueError("two-stage fact order differs")
    relnet = GatedRelNet(len(ALL_FACTS), hidden_channels=128,
                         relation_names=tuple(bank.full_graph),
                         num_layers=2, dropout=0.3, out_channels=2)
    model = TwoStageFox8(HGTSourceSelector(context_dim=128), relnet,
                         torch.tensor(saved["scaler_mean"]),
                         torch.tensor(saved["scaler_scale"])).eval()
    model.load_state_dict(saved["model_state"])
    with torch.inference_mode():
        full_context = hgt({"user": x}, graph)["user_emb"]
        full = model(full_context, bank)
    logits = full["logits"].numpy()
    margin = logits[:, 1] - logits[:, 0]
    labels = y.numpy().astype(int)
    threshold = float(saved.get(
        "dev_logit_threshold", dev_threshold(margin, labels, split["dev"])))

    selected = full["hard_selection"]
    parity = {}
    for class_index, name in ((0, "human"), (1, "bot")):
        profile = selected["profile"][:, class_index]
        tweet = selected["tweet"][:, class_index]
        relation = selected["relation"][:, class_index]
        facts = bank.exact_view(profile, tweet)
        scaled = torch.from_numpy(np.clip(
            (facts - np.asarray(saved["scaler_mean"])) /
            np.asarray(saved["scaler_scale"]), -3, 3).astype(np.float32))
        relation_ids = [item.source_id for item, keep in zip(relations, relation.tolist())
                        if keep]
        rebuilt = graph_from_relation_atoms(relations, ids, relation_ids)
        with torch.inference_mode():
            independent = model.relnet(scaled, rebuilt)[:, class_index].numpy()
        parity[name] = float(np.max(np.abs(independent - logits[:, class_index])))
    if max(parity.values()) > 1e-5:
        raise ValueError("selected-source exact graph replay differs from gated model")

    union = {name: torch.maximum(gate[:, 0], gate[:, 1])
             for name, gate in selected.items()}
    union_facts = bank.exact_view(union["profile"], union["tweet"])
    union_base = torch.from_numpy(np.clip(
        (union_facts[:, :31] - np.asarray(base["input_contract"]["scaler_mean"])) /
        np.asarray(base["input_contract"]["scaler_scale"]), -3, 3).astype(np.float32))
    union_relation_ids = [item.source_id for item, keep in
                          zip(relations, union["relation"].tolist()) if keep]
    union_graph = graph_from_relation_atoms(relations, ids, union_relation_ids)
    with torch.inference_mode():
        union_context = hgt({"user": union_base}, {
            ("user", name, "user"): edge for name, edge in union_graph.items()
        })["user_emb"]
        replay = model(union_context, bank, availability=union)
    replay_margin = (replay["logits"][:, 1] - replay["logits"][:, 0]).numpy()
    all_test = split["test"]
    full_prediction = margin[all_test] >= threshold
    replay_prediction = replay_margin[all_test] >= threshold
    report = {
        "pilot_checkpoint_sha256": sha256_file(checkpoint_path),
        "threshold_selected_on_dev_only": threshold,
        "full_source_view_replay_max_diff": parity,
        "original_default_threshold": {
            "dev": classification_metrics(margin, labels, split["dev"], 0.0),
            "test": classification_metrics(margin, labels, all_test, 0.0)},
        "calibrated": {
            "train": classification_metrics(margin, labels, split["train"], threshold),
            "dev": classification_metrics(margin, labels, split["dev"], threshold),
            "test": classification_metrics(margin, labels, all_test, threshold)},
        "selected_only_full_pipeline_replay": {
            "scope": "all accounts restricted to the union of emitted Human/Bot sources",
            "test_prediction_agreement_with_full": float(np.mean(
                full_prediction == replay_prediction)),
            "test_accuracy": classification_metrics(replay_margin, labels, all_test,
                                                     threshold)["accuracy"],
            "test_changed_count": int(np.count_nonzero(
                full_prediction != replay_prediction)),
            "union_source_counts": {name: int(gate.sum()) for name, gate in union.items()},
        },
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
