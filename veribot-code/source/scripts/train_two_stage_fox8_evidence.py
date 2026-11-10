"""Train HGT source selection and evidence-view RelNet on fixed Fox8 splits."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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
from verifiable_bot.fox8_two_stage_model import (
    HGTSourceSelector, TwoStageFox8, initialize_relnet_from_binary,
)

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
BASE = ROOT / "output/aaai_veribot_fox8/seed0/model.pt"
DEFAULT_OUTPUT = ROOT / "output/two_stage_fox8_evidence/seed0"


def metrics(logits: np.ndarray, labels: np.ndarray, indices: np.ndarray,
            threshold: float = 0.0) -> dict:
    margin = logits[:, 1] - logits[:, 0]
    prediction = margin[indices] >= threshold
    target = labels[indices].astype(int)
    return {
        "n": len(indices),
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro")),
        "auc": float(roc_auc_score(target, margin[indices])),
    }


def dev_joint_policy(full_logits: np.ndarray, replay_logits: np.ndarray,
                     labels: np.ndarray, dev: np.ndarray) -> tuple[float, tuple]:
    full_margin = full_logits[:, 1] - full_logits[:, 0]
    replay_margin = replay_logits[:, 1] - replay_logits[:, 0]
    values = sorted(set(full_margin[dev].tolist() + replay_margin[dev].tolist()))
    candidates = [values[0] - 1, *[(a + b) / 2 for a, b in
                                    zip(values[:-1], values[1:])], values[-1] + 1]
    truth = labels[dev].astype(int)

    def criterion(threshold: float) -> tuple:
        full = full_margin[dev] >= threshold
        replay = replay_margin[dev] >= threshold
        full_f1 = float(f1_score(truth, full, average="macro"))
        replay_f1 = float(f1_score(truth, replay, average="macro"))
        return (min(full_f1, replay_f1), (full_f1 + replay_f1) / 2,
                -abs(threshold))

    threshold = max(candidates, key=criterion)
    return float(threshold), criterion(threshold)


def selected_only_context(
    hgt: VerifiableHGT, bank: Fox8TwoStageInputs,
    hard_selection: dict[str, torch.Tensor],
    base_mean: np.ndarray, base_scale: np.ndarray,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = next(hgt.parameters()).device
    availability = {name: torch.maximum(mask[:, 0], mask[:, 1]).detach()
                    for name, mask in hard_selection.items()}
    facts = bank.exact_view(availability["profile"], availability["tweet"])
    selected = torch.from_numpy(np.clip(
        (facts[:, :31] - base_mean) / base_scale, -3, 3).astype(np.float32)).to(device)
    relation_ids = [atom.source_id for atom, keep in
                    zip(bank.relations, availability["relation"].tolist()) if keep]
    graph = {name: edges.to(device) for name, edges in
             graph_from_relation_atoms(bank.relations, bank.ids,
                                       relation_ids).items()}
    with torch.inference_mode():
        context = hgt({"user": selected}, {
            ("user", name, "user"): edge for name, edge in graph.items()
        })["user_emb"].detach()
    return context, availability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--sparsity", type=float, default=0.02)
    parser.add_argument("--initial-gate-bias", type=float, default=0.0)
    parser.add_argument("--selected-only-weight", type=float, default=0.0)
    parser.add_argument("--counterfactual-weight", type=float, default=0.0)
    parser.add_argument("--counterfactual-margin", type=float, default=0.5)
    parser.add_argument("--dev-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but no GPU is available")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available())
                          else "cpu")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    x, y, split, ids, graph, pack = load_inputs()
    checkpoint = torch.load(BASE, map_location="cpu", weights_only=False)
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3)
    hgt.load_state_dict(checkpoint["model_state"]["hgt"])
    hgt.to(device)
    hgt.eval()
    with torch.inference_mode():
        context = hgt({"user": x.to(device)}, {
            key: edges.to(device) for key, edges in graph.items()
        })["user_emb"].detach()
    accounts = load_account_atoms(RAW, ids)
    relations = load_relation_atoms(EVENTS, RAW, ids)
    bank = Fox8TwoStageInputs.build(accounts, relations, ids)
    feature_file = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
    original = pd.read_csv(feature_file)[list(ALL_FACTS[:31])].to_numpy(dtype=np.float32)
    if not np.array_equal(bank.full_facts[:, :31], original):
        raise ValueError("source-preserving full view differs from original model input")
    for name, actual in graph.items():
        if not torch.equal(actual, bank.full_graph[name[1]]):
            raise ValueError("source-preserving graph differs from HGT input")
    scaler = StandardScaler().fit(bank.full_facts[split["train"]])
    if (not np.allclose(scaler.mean_[:31], checkpoint["input_contract"]["scaler_mean"], atol=1e-6)
            or not np.allclose(scaler.scale_[:31], checkpoint["input_contract"]["scaler_scale"], atol=1e-6)):
        raise ValueError("full-input standardization differs from RelNet initializer")
    relnet = GatedRelNet(len(ALL_FACTS), hidden_channels=128,
                         relation_names=tuple(bank.full_graph),
                         num_layers=2, dropout=0.3, out_channels=2)
    initialize_relnet_from_binary(relnet, checkpoint["model_state"]["relnet"])
    model = TwoStageFox8(HGTSourceSelector(context_dim=context.shape[1],
                                          initial_gate_bias=args.initial_gate_bias),
                         relnet, torch.from_numpy(scaler.mean_),
                         torch.from_numpy(scaler.scale_)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    train = torch.as_tensor(split["train"], dtype=torch.long, device=device)
    labels = y.long().to(device)
    labels_np = y.numpy()
    base_mean = np.asarray(checkpoint["input_contract"]["scaler_mean"])
    base_scale = np.asarray(checkpoint["input_contract"]["scaler_scale"])
    best = None
    best_epoch = 0
    best_dev = (-1.0, -1.0, -1.0)
    best_threshold = 0.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model(context, bank,
                       compute_complement=args.counterfactual_weight > 0)
        prediction_loss = nn.functional.cross_entropy(result["logits"][train], labels[train])
        if args.counterfactual_weight > 0:
            chosen_margin = result["logits"][:, 1] - result["logits"][:, 0]
            complement_margin = (result["complement_logits"][:, 1] -
                                 result["complement_logits"][:, 0])
            direction = 2 * labels[train].float() - 1
            counterfactual_loss = nn.functional.relu(
                args.counterfactual_margin - direction *
                (chosen_margin[train] - complement_margin[train])).mean()
        else:
            counterfactual_loss = torch.zeros_like(prediction_loss)
        if args.selected_only_weight > 0:
            replay_context, availability = selected_only_context(
                hgt, bank, result["hard_selection"], base_mean, base_scale)
            replay_result = model(replay_context, bank, availability=availability)
            replay_ce = nn.functional.cross_entropy(
                replay_result["logits"][train], labels[train])
        else:
            replay_ce = torch.zeros_like(prediction_loss)
        loss = (prediction_loss + args.selected_only_weight * replay_ce +
                args.counterfactual_weight * counterfactual_loss +
                args.sparsity * result["mean_source_selection_probability"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            evaluation = model(context, bank)
            logits = evaluation["logits"].detach().cpu().numpy()
            if args.selected_only_weight > 0:
                evaluation_context, evaluation_availability = selected_only_context(
                    hgt, bank, evaluation["hard_selection"], base_mean, base_scale)
                replay_evaluation = model(evaluation_context, bank,
                                          availability=evaluation_availability)
                replay_logits = replay_evaluation["logits"].detach().cpu().numpy()
            else:
                replay_logits = logits
        dev = metrics(logits, labels_np, split["dev"])
        replay_dev = metrics(replay_logits, labels_np, split["dev"])
        candidate_threshold, score = dev_joint_policy(
            logits, replay_logits, labels_np, split["dev"])
        if score > best_dev:
            best_dev = score
            best_epoch = epoch
            best_threshold = candidate_threshold
            best = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(json.dumps({"epoch": epoch, "train_loss": float(loss.detach()),
                          "train_ce": float(prediction_loss.detach()),
                          "replay_ce": float(replay_ce.detach()),
                          "counterfactual_loss": float(counterfactual_loss.detach()),
                          "dev": dev, "selected_only_dev": replay_dev,
                          "joint_dev_threshold": candidate_threshold,
                          "joint_dev_min_macro_f1": score[0],
                          "mean_selected_probability": float(result[
                              "mean_source_selection_probability"].detach()),
                          "best_epoch": best_epoch}), flush=True)
        if stale >= args.patience:
            break
    if best is None:
        raise ValueError("no trained two-stage model")
    model.load_state_dict(best)
    model.eval()
    with torch.inference_mode():
        final = model(context, bank)
        final_context, final_availability = selected_only_context(
            hgt, bank, final["hard_selection"], base_mean, base_scale)
        final_replay = model(final_context, bank, availability=final_availability)
    final_logits = final["logits"].detach().cpu().numpy()
    final_replay_logits = final_replay["logits"].detach().cpu().numpy()
    source_counts = {name: {
        "human": int(outcome[:, 0].sum()),
        "bot": int(outcome[:, 1].sum()),
        "available": len(outcome),
    } for name, outcome in final["hard_selection"].items()}
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model.pt"
    torch.save({"model_state": model.state_dict(),
                "hgt_context_checkpoint_sha256": sha256_file(BASE),
                "source_sha256": sha256_file(RAW),
                "event_index_sha256": sha256_file(EVENTS),
                "source_counts": source_counts,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "best_epoch": best_epoch, "seed": args.seed,
                "training_device": str(device),
                "counterfactual_weight": args.counterfactual_weight,
                "counterfactual_margin": args.counterfactual_margin,
                "sparsity": args.sparsity,
                "initial_gate_bias": args.initial_gate_bias,
                "dev_logit_threshold": best_threshold,
                "feature_names": list(ALL_FACTS)}, model_path)
    report = {
        "algorithm": "HGT_source_selector_then_evidence_view_RelNet",
        "training_device": str(device),
        "checkpoint_sha256": sha256_file(model_path),
        "source_counts": source_counts,
        "source_universe": {"profile_fields": len(bank.profile_account),
                            "tweets": len(bank.tweet_account),
                            "native_relation_events": len(bank.relations)},
        "best_epoch_dev_only": best_epoch,
        "dev_logit_threshold": best_threshold,
        "selected_only_weight": args.selected_only_weight,
        "counterfactual_weight": args.counterfactual_weight,
        "counterfactual_margin": args.counterfactual_margin,
        "sparsity": args.sparsity,
        "initial_gate_bias": args.initial_gate_bias,
        "train": metrics(final_logits, labels_np, split["train"]),
        "dev": metrics(final_logits, labels_np, split["dev"]),
        "selected_only_dev": metrics(final_replay_logits, labels_np, split["dev"]),
        "selected_only_calibrated_dev": metrics(
            final_replay_logits, labels_np, split["dev"], best_threshold),
        "calibrated_dev": metrics(final_logits, labels_np,
                                  split["dev"], best_threshold),
    }
    if not args.dev_only:
        report.update({
            "test": metrics(final_logits, labels_np, split["test"]),
            "calibrated_test": metrics(final_logits, labels_np,
                                       split["test"], best_threshold),
            "selected_only_test": metrics(final_replay_logits, labels_np,
                                          split["test"]),
            "selected_only_calibrated_test": metrics(
                final_replay_logits, labels_np, split["test"], best_threshold),
            "selected_only_test_prediction_agreement": float(np.mean(
                (final_logits[split["test"], 1] -
                 final_logits[split["test"], 0] >= 0) ==
                (final_replay_logits[split["test"], 1] -
                 final_replay_logits[split["test"], 0] >= 0))),
            "selected_only_calibrated_test_prediction_agreement": float(np.mean(
                (final_logits[split["test"], 1] -
                 final_logits[split["test"], 0] >= best_threshold) ==
                (final_replay_logits[split["test"], 1] -
                 final_replay_logits[split["test"], 0] >= best_threshold))),
            "test_predictions": [{"account_id": ids[int(i)],
                                  "label": int(labels_np[i]),
                                  "human_score": float(final_logits[i, 0]),
                                  "bot_score": float(final_logits[i, 1])}
                                 for i in split["test"]],
        })
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "test_predictions"}), flush=True)


if __name__ == "__main__":
    main()
