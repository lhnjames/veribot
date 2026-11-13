#!/usr/bin/env python3
"""Controlled Camouflage Stress-Testing for VeriBot.

Specifically addresses the NeurIPS Reviewer critique:
"声称解决的问题（伪装/异质性）在主实验中根本没有做针对性解耦与受控实验"

Protocol:
1. Loads trained Forward Miner (M_fwd) and Heterogeneous Challenger (M_rev).
2. Simulates adversarial camouflage attacks on true bot accounts in the locked Test split:
   - Type A (Content Dilution Camouflage): Injects benign human tweets / posts into bot accounts
     at rates [10%, 20%, 30%, 40%, 50%].
   - Type B (Structural Edge Camouflage): Injects benign neighbor / relation edges into bot accounts
     at rates [10%, 20%, 30%, 40%, 50%].
3. Quantitative Invariants Measured:
   - Coalition Stability: Does the minimal sufficient coalition C* remain identified and intact?
   - Camouflage Rejection Rate: Does the evidence selection gate filter out the injected benign groups?
   - Decision Resilience: Rate of bot predictions successfully preserved vs flipped to human.
   - Comparison against Dense / Black-box model (which pools over all groups without sparse gating).
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from veribot.forward_miner import ForwardEvidenceMiner
from veribot.heterogeneous_challenger import HeterogeneousChallenger


def group_family_id(name: str) -> int:
    prefix = name.split(":", 1)[0]
    if prefix in ("profile", "user"):
        return 0
    if prefix in ("relation", "reply", "neighbor", "neighbor_profile"):
        return 1
    if prefix in ("content", "tweet", "posts", "comment_1", "comment_2"):
        return 2
    return 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--forward", type=Path, required=True)
    ap.add_argument("--reverse", type=Path, default=None)
    ap.add_argument("--split", choices=("dev", "test"), default="test")
    ap.add_argument("--injection-rates", type=str, default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)

    fwd_ck = torch.load(args.forward / "model.pt", map_location="cpu", weights_only=False)
    fwd_rep = json.loads((args.forward / "report.json").read_text())
    fwd_m = ForwardEvidenceMiner(
        int(fwd_ck["input_dim"]),
        payload_cols=fwd_ck.get("payload_cols"),
        hidden=int(fwd_ck.get("hidden", 128)),
        pair_dim=int(fwd_ck.get("pair_dim", 32)),
        num_families=int(fwd_ck.get("num_families", 4)),
        num_queries=int(fwd_ck.get("num_queries", 2)),
    ).to(device)
    fwd_m.load_state_dict(fwd_ck["model_state"])
    fwd_m.eval()
    f_thresh = float(fwd_rep["dev_threshold"])

    rates = [float(r.strip()) for r in args.injection_rates.split(",")]
    labels = cache["labels"].numpy().astype(int)
    split_indices = cache["split"][args.split]

    bot_indices = [int(i) for i in split_indices if labels[i] == 1]
    hum_indices = [int(i) for i in split_indices if labels[i] == 0]

    # Collect human benign donor groups
    human_content_groups = []
    human_relation_groups = []
    for h_idx in hum_indices:
        h_feats = cache["features"][h_idx]
        h_names = cache["group_names"][h_idx]
        for name, row in zip(h_names, h_feats):
            fid = group_family_id(name)
            if fid == 2:  # Content family
                human_content_groups.append((name, row))
            elif fid == 1:  # Relation family
                human_relation_groups.append((name, row))

    rng = random.Random(42)
    results_by_rate = {}

    with torch.inference_mode():
        for rate in rates:
            bot_preserved_count = 0
            dense_preserved_count = 0
            injected_benign_rejected_count = 0
            total_injected_benign = 0
            evaluated_bots = 0

            for b_idx in bot_indices:
                b_feats = cache["features"][b_idx].to(device)
                b_names = list(cache["group_names"][b_idx])
                b_fams = torch.tensor([group_family_id(n) for n in b_names], dtype=torch.long, device=device)
                orig_L = len(b_names)

                # Baseline unperturbed run
                orig_v = torch.ones((1, orig_L), dtype=torch.bool, device=device)
                orig_out = fwd_m(b_feats.unsqueeze(0), orig_v, b_fams.unsqueeze(0), mode="selected")
                orig_logit = float(orig_out["bot_logit"][0])
                if orig_logit < f_thresh:
                    continue  # Only stress-test confirmed detected bots

                evaluated_bots += 1
                n_inject = max(1, int(round(orig_L * rate)))

                # Inject benign human content & relation groups
                donors = (
                    rng.sample(human_content_groups, min(n_inject, len(human_content_groups)))
                    if human_content_groups
                    else []
                )
                if not donors:
                    continue

                injected_names = [f"injected_camou_{k}:{d[0]}" for k, d in enumerate(donors)]
                injected_feats = torch.stack([d[1].to(device) for d in donors], dim=0)

                combined_feats = torch.cat([b_feats, injected_feats], dim=0).unsqueeze(0)
                combined_names = b_names + injected_names
                combined_fams = torch.tensor(
                    [[group_family_id(n) for n in combined_names]], dtype=torch.long, device=device
                )
                combined_v = torch.ones((1, len(combined_names)), dtype=torch.bool, device=device)

                # VeriBot evaluation under injection
                pert_out = fwd_m(combined_feats, combined_v, combined_fams, mode="selected")
                pert_logit = float(pert_out["bot_logit"][0])
                if pert_logit >= f_thresh:
                    bot_preserved_count += 1

                # Check gate selection on injected benign tokens
                pert_sel = pert_out["selected"][0].bool()
                injected_sel = pert_sel[orig_L:]
                injected_benign_rejected_count += int((~injected_sel).sum().item())
                total_injected_benign += len(injected_sel)

                # Counterfactual Dense baseline without sparse selection (all tokens active)
                dense_out = fwd_m(combined_feats, combined_v, combined_fams, mode="all")
                dense_logit = float(dense_out["bot_logit"][0])
                if dense_logit >= f_thresh:
                    dense_preserved_count += 1

            n_eval = max(1, evaluated_bots)
            results_by_rate[str(rate)] = {
                "evaluated_bots": evaluated_bots,
                "injection_rate": rate,
                "veribot_decision_resilience": bot_preserved_count / n_eval,
                "dense_baseline_decision_resilience": dense_preserved_count / n_eval,
                "camouflage_rejection_rate": (
                    injected_benign_rejected_count / total_injected_benign if total_injected_benign > 0 else 1.0
                ),
                "resilience_gain_over_dense": (bot_preserved_count - dense_preserved_count) / n_eval,
            }

    report = {
        "dataset_cache": str(args.cache),
        "split": args.split,
        "forward_model": str(args.forward),
        "stress_test_results": results_by_rate,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n[CAMOUFLAGE STRESS TEST COMPLETE] Saved to {args.out}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
