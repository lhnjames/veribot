"""Train target-conditioned HGT selection and one-hop RelNet on Fox8 dev."""

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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "source/scripts"),
                str(ROOT / "source/hgt_import/code")]

from hgt_model import VerifiableHGT
from relnet_reimplementation import GatedRelNet
from train_aaai_veribot_fox8 import load_inputs, sha256_file
from train_two_stage_fox8_evidence import dev_joint_policy, metrics
from verifiable_bot.fox8_atomic_sources import load_account_atoms, load_relation_atoms
from verifiable_bot.fox8_target_evidence_model import (
    HGTTargetFactSelector, HGTTargetRelationSelector, TargetEvidenceFox8,
    initialize_target_relnet,
)
from verifiable_bot.fox8_two_stage_inputs import ALL_FACTS, Fox8TwoStageInputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparsity", type=float, default=0.1)
    parser.add_argument("--counterfactual-weight", type=float, default=0.2)
    parser.add_argument("--counterfactual-margin", type=float, default=0.5)
    parser.add_argument("--own-class-support-weight", type=float, default=0.0)
    parser.add_argument("--own-class-support-margin", type=float, default=0.5)
    parser.add_argument("--intervention-impact-weight", type=float, default=0.0)
    parser.add_argument("--intervention-impact-margin", type=float, default=0.5)
    parser.add_argument("--intervention-sample-per-class", type=int, default=16)
    parser.add_argument("--initial-gate-bias", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--relation-gates", action="store_true")
    parser.add_argument("--centered-evidence", action="store_true")
    parser.add_argument("--bot-only-support", action="store_true")
    parser.add_argument("--selected-neighbor-sources", action="store_true")
    parser.add_argument("--neutral-support-weight", type=float, default=0.0)
    parser.add_argument("--neutral-support-margin", type=float, default=0.5)
    parser.add_argument("--deterministic-training-gates", action="store_true")
    parser.add_argument("--gate-polarization-weight", type=float, default=0.0)
    parser.add_argument("--bot-source-count-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.own_class_support_weight < 0 or args.own_class_support_margin < 0:
        raise ValueError("own-class support weight and margin must be nonnegative")
    if (args.neutral_support_weight < 0 or args.neutral_support_margin < 0):
        raise ValueError("neutral support weight and margin must be nonnegative")
    if args.gate_polarization_weight < 0:
        raise ValueError("gate polarization weight must be nonnegative")
    if args.bot_source_count_weight < 0:
        raise ValueError("Bot source count weight must be nonnegative")
    if args.bot_source_count_weight and not args.bot_only_support:
        raise ValueError("Bot source count penalty requires Bot-only support")
    if args.centered_evidence and not args.relation_gates:
        raise ValueError("centered evidence requires relation gates")
    if args.bot_only_support and not args.centered_evidence:
        raise ValueError("Bot-only support requires centered evidence")
    if args.selected_neighbor_sources and not args.bot_only_support:
        raise ValueError("Selected neighbor sources require Bot-only support")
    if args.bot_only_support and args.intervention_impact_weight:
        raise ValueError("Bot-only intervention impact needs a class-specific sampler")
    if args.neutral_support_weight and not args.centered_evidence:
        raise ValueError("neutral support loss requires centered evidence")
    if (args.intervention_impact_weight < 0 or
            args.intervention_impact_margin < 0 or
            args.intervention_sample_per_class < 1):
        raise ValueError("invalid intervention impact settings")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    raw = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    events = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
    base_path = ROOT / "output/aaai_veribot_fox8/seed0/model.pt"
    x, y, split, ids, graph, pack = load_inputs()
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3).to(device).eval()
    hgt.load_state_dict(base["model_state"]["hgt"])
    with torch.inference_mode():
        context = hgt({"user": x.to(device)}, {
            key: edges.to(device) for key, edges in graph.items()
        })["user_emb"].detach()
    hgt_graph = {key: edges.to(device) for key, edges in graph.items()}
    hgt_mean = np.asarray(base["input_contract"]["scaler_mean"], dtype=np.float32)
    hgt_scale = np.asarray(base["input_contract"]["scaler_scale"], dtype=np.float32)
    bank = Fox8TwoStageInputs.build(
        load_account_atoms(raw, ids), load_relation_atoms(events, raw, ids), ids)
    feature_file = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
    original = pd.read_csv(feature_file)[list(ALL_FACTS[:31])].to_numpy(
        dtype=np.float32)
    if not np.array_equal(bank.full_facts[:, :31], original):
        raise ValueError("source-preserving view differs from original model input")
    for name, edges in graph.items():
        if not torch.equal(edges, bank.full_graph[name[1]]):
            raise ValueError("native graph differs from HGT input")
    scaler = StandardScaler().fit(bank.full_facts[split["train"]])
    relnet = GatedRelNet(len(ALL_FACTS), hidden_channels=128,
                         relation_names=tuple(bank.full_graph),
                         num_layers=1, dropout=0.3, out_channels=2)
    initialize_target_relnet(relnet, base["model_state"]["relnet"])
    model = TargetEvidenceFox8(
        (HGTTargetRelationSelector if args.relation_gates else
         HGTTargetFactSelector)(context_dim=128,
                                initial_gate_bias=args.initial_gate_bias),
        relnet, torch.from_numpy(scaler.mean_),
        torch.from_numpy(scaler.scale_),
        centered_evidence=args.centered_evidence,
        sample_training_gates=not args.deterministic_training_gates,
        bot_only_support=args.bot_only_support,
        selected_neighbor_sources=args.selected_neighbor_sources).to(device)
    if args.warmup_epochs < 0:
        raise ValueError("warmup epochs must be nonnegative")
    if args.warmup_epochs:
        full_x = torch.from_numpy(np.clip(
            (bank.full_facts - scaler.mean_) / scaler.scale_,
            -3, 3).astype(np.float32)).to(device)
        full_graph = {name: edge.to(device)
                      for name, edge in bank.full_graph.items()}
        warmup_optimizer = torch.optim.AdamW(
            model.relnet.parameters(), lr=2e-4, weight_decay=1e-4)
        warmup_train = torch.as_tensor(
            split["train"], dtype=torch.long, device=device)
        warmup_labels = y.long().to(device)
        warmup_best = None
        warmup_score = (-1.0, -1.0, -1.0)
        warmup_best_epoch = 0
        for warmup_epoch in range(1, args.warmup_epochs + 1):
            model.relnet.train()
            warmup_optimizer.zero_grad(set_to_none=True)
            warmup_logits = model.relnet(full_x, full_graph)
            warmup_loss = nn.functional.cross_entropy(
                warmup_logits[warmup_train], warmup_labels[warmup_train])
            warmup_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.relnet.parameters(), 1.0)
            warmup_optimizer.step()
            model.relnet.eval()
            with torch.inference_mode():
                dev_logits = model.relnet(full_x, full_graph).cpu().numpy()
            _, current_score = dev_joint_policy(
                dev_logits, dev_logits, y.numpy(), split["dev"])
            if current_score > warmup_score:
                warmup_best = deepcopy(model.relnet.state_dict())
                warmup_score = current_score
                warmup_best_epoch = warmup_epoch
            print(json.dumps({"warmup_epoch": warmup_epoch,
                              "train_loss": float(warmup_loss.detach()),
                              "dev_macro_f1": current_score[0],
                              "best_warmup_epoch": warmup_best_epoch}), flush=True)
        model.relnet.load_state_dict(warmup_best)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4,
                                  weight_decay=1e-4)
    train = torch.as_tensor(split["train"], dtype=torch.long, device=device)
    labels = y.long().to(device)
    labels_np = y.numpy()
    train_by_class = {
        cls: np.asarray([int(i) for i in split["train"]
                         if int(labels_np[int(i)]) == cls], dtype=np.int64)
        for cls in (0, 1)
    }
    best = None
    best_score = (-1.0, -1.0, -1.0)
    best_epoch = 0
    best_threshold = 0.0
    stale = 0
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model(context, bank,
                       compute_complement=(args.counterfactual_weight > 0 or
                                           args.own_class_support_weight > 0))
        ce = nn.functional.cross_entropy(result["logits"][train], labels[train])
        support_train = train[labels[train] == 1] if args.bot_only_support else train
        if args.neutral_support_weight:
            true_class_support = result["logits"][
                support_train, labels[support_train]]
            neutral_support = nn.functional.relu(
                args.neutral_support_margin - true_class_support).mean()
        else:
            neutral_support = torch.zeros_like(ce)
        if args.counterfactual_weight:
            selected_margin = result["logits"][:, 1] - result["logits"][:, 0]
            complement_margin = (result["complement_logits"][:, 1] -
                                 result["complement_logits"][:, 0])
            direction = labels[support_train].float() * 2 - 1
            counterfactual = nn.functional.relu(
                args.counterfactual_margin - direction *
                (selected_margin[support_train] -
                 complement_margin[support_train])).mean()
        else:
            counterfactual = torch.zeros_like(ce)
        if args.own_class_support_weight:
            chosen_score = result["logits"][
                support_train, labels[support_train]]
            complement_score = result["complement_logits"][
                support_train, labels[support_train]]
            own_class_support = nn.functional.relu(
                args.own_class_support_margin -
                (chosen_score - complement_score)).mean()
        else:
            own_class_support = torch.zeros_like(ce)
        if args.intervention_impact_weight:
            # Optimize the change after withholding named raw sources and
            # recomputing the frozen HGT, rather than comparing proxy views.
            model.eval()
            full_eval = model(context, bank)
            sampled = np.concatenate([
                np.random.choice(train_by_class[cls],
                                 min(args.intervention_sample_per_class,
                                     len(train_by_class[cls])), replace=False)
                for cls in (0, 1)
            ])
            availability = {
                "profile": torch.ones(len(bank.profile_account), dtype=torch.bool),
                "tweet": torch.ones(len(bank.tweet_account), dtype=torch.bool),
                "relation": torch.ones(len(bank.relations), dtype=torch.bool),
            }
            for idx in sampled:
                cls = int(labels_np[idx])
                p0, p1 = bank.profile_offsets[idx:idx + 2]
                t0, t1 = bank.tweet_offsets[idx:idx + 2]
                availability["profile"][p0:p1] &= ~full_eval[
                    "hard_selection"]["profile"][p0:p1, cls].bool().cpu()
                availability["tweet"][t0:t1] &= ~full_eval[
                    "hard_selection"]["tweet"][t0:t1, cls].bool().cpu()
            masked_facts = bank.exact_view(availability["profile"],
                                           availability["tweet"])
            masked_x = torch.from_numpy(np.clip(
                (masked_facts[:, :31] - hgt_mean) / hgt_scale,
                -3, 3).astype(np.float32)).to(device)
            with torch.no_grad():
                masked_context = hgt({"user": masked_x}, hgt_graph)[
                    "user_emb"].detach()
            masked_eval = model(masked_context, bank,
                                availability=availability)
            sampled_t = torch.as_tensor(sampled, dtype=torch.long, device=device)
            full_margin = (full_eval["logits"][sampled_t, 1] -
                           full_eval["logits"][sampled_t, 0])
            masked_margin = (masked_eval["logits"][sampled_t, 1] -
                             masked_eval["logits"][sampled_t, 0])
            direction = labels[sampled_t].float() * 2 - 1
            intervention_impact = nn.functional.relu(
                args.intervention_impact_margin -
                direction * (full_margin - masked_margin)).mean()
            removed_source_count = (
                (~availability["profile"]).sum() +
                (~availability["tweet"]).sum()).item()
            model.train()
        else:
            intervention_impact = torch.zeros_like(ce)
            removed_source_count = 0
        loss = (ce + args.counterfactual_weight * counterfactual +
                args.neutral_support_weight * neutral_support +
                args.gate_polarization_weight *
                result["mean_gate_uncertainty"] +
                args.bot_source_count_weight *
                result["expected_bot_source_count_per_account"] +
                args.own_class_support_weight * own_class_support +
                args.intervention_impact_weight * intervention_impact +
                args.sparsity * result["mean_source_selection_probability"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            evaluation = model(context, bank)
        logits = evaluation["logits"].detach().cpu().numpy()
        threshold, score = dev_joint_policy(
            logits, logits, labels_np, split["dev"])
        if score > best_score:
            best = deepcopy(model.state_dict())
            best_score = score
            best_epoch = epoch
            best_threshold = threshold
            stale = 0
        else:
            stale += 1
        print(json.dumps({
            "epoch": epoch, "train_loss": float(loss.detach()),
            "train_ce": float(ce.detach()),
            "counterfactual_loss": float(counterfactual.detach()),
            "neutral_support_loss": float(neutral_support.detach()),
            "gate_uncertainty": float(result[
                "mean_gate_uncertainty"].detach()),
            "expected_bot_source_count_per_account": float(result[
                "expected_bot_source_count_per_account"].detach()),
            "own_class_support_loss": float(own_class_support.detach()),
            "intervention_impact_loss": float(intervention_impact.detach()),
            "intervention_removed_source_count": removed_source_count,
            "dev": metrics(logits, labels_np, split["dev"]),
            "dev_threshold": threshold,
            "calibrated_dev_macro_f1": score[0],
            "mean_selected_probability": float(result[
                "mean_source_selection_probability"].detach()),
            "best_epoch": best_epoch,
        }), flush=True)
        if stale >= args.patience:
            break
    if best is None:
        raise ValueError("no target evidence model saved")
    model.load_state_dict(best)
    model.eval()
    with torch.inference_mode():
        final = model(context, bank)
    final_logits = final["logits"].detach().cpu().numpy()
    source_counts = {name: {
        "human": int(mask[:, 0].sum()),
        "bot": int(mask[:, 1].sum()),
        "available": len(mask),
    } for name, mask in final["hard_selection"].items()}
    architecture = (
        "target_HGT_bot_source_relation_selector_selected_neighbor_centered_RelNet"
        if args.selected_neighbor_sources else
        "target_HGT_bot_source_relation_selector_centered_RelNet"
        if args.bot_only_support else
        "target_HGT_dual_class_source_relation_selector_centered_RelNet"
        if args.centered_evidence else
        "target_HGT_source_and_relation_selector_one_hop_evidence_RelNet"
        if args.relation_gates else
        "target_HGT_source_selector_one_hop_evidence_RelNet")
    model_path = args.output / "model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "hgt_context_checkpoint_sha256": sha256_file(base_path),
        "source_sha256": sha256_file(raw),
        "event_index_sha256": sha256_file(events),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "best_epoch": best_epoch,
        "seed": args.seed,
        "warmup_epochs": args.warmup_epochs,
        "relation_gates": args.relation_gates,
        "centered_evidence": args.centered_evidence,
        "bot_only_support": args.bot_only_support,
        "selected_neighbor_sources": args.selected_neighbor_sources,
        "neutral_support_weight": args.neutral_support_weight,
        "neutral_support_margin": args.neutral_support_margin,
        "sample_training_gates": not args.deterministic_training_gates,
        "gate_polarization_weight": args.gate_polarization_weight,
        "bot_source_count_weight": args.bot_source_count_weight,
        "dev_logit_threshold": best_threshold,
        "source_counts": source_counts,
        "own_class_support_weight": args.own_class_support_weight,
        "own_class_support_margin": args.own_class_support_margin,
        "intervention_impact_weight": args.intervention_impact_weight,
        "intervention_impact_margin": args.intervention_impact_margin,
        "intervention_sample_per_class": args.intervention_sample_per_class,
        "feature_names": list(ALL_FACTS),
        "architecture": architecture,
    }, model_path)
    report = {
        "architecture": architecture,
        "checkpoint_sha256": sha256_file(model_path),
        "training_device": str(device),
        "seed": args.seed,
        "warmup_epochs": args.warmup_epochs,
        "relation_gates": args.relation_gates,
        "centered_evidence": args.centered_evidence,
        "bot_only_support": args.bot_only_support,
        "selected_neighbor_sources": args.selected_neighbor_sources,
        "neutral_support_weight": args.neutral_support_weight,
        "neutral_support_margin": args.neutral_support_margin,
        "sample_training_gates": not args.deterministic_training_gates,
        "gate_polarization_weight": args.gate_polarization_weight,
        "bot_source_count_weight": args.bot_source_count_weight,
        "sparsity": args.sparsity,
        "counterfactual_weight": args.counterfactual_weight,
        "counterfactual_margin": args.counterfactual_margin,
        "own_class_support_weight": args.own_class_support_weight,
        "own_class_support_margin": args.own_class_support_margin,
        "intervention_impact_weight": args.intervention_impact_weight,
        "intervention_impact_margin": args.intervention_impact_margin,
        "intervention_sample_per_class": args.intervention_sample_per_class,
        "best_epoch_dev_only": best_epoch,
        "dev_logit_threshold": best_threshold,
        "source_counts": source_counts,
        "source_universe": {
            "profile_fields": len(bank.profile_account),
            "tweets": len(bank.tweet_account),
            "native_relation_events": len(bank.relations),
        },
        "train": metrics(final_logits, labels_np, split["train"], best_threshold),
        "dev": metrics(final_logits, labels_np, split["dev"], best_threshold),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
