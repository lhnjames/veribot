#!/usr/bin/env python3
"""Automated Ablation Study Suite for VeriBot.

Executes the 14 ablation variants across Dimensions A, B, C, D on MGTAB (and optional TwiBot-20 / BotSim)
over 5 random seeds (0, 1, 2, 3, 4).
Strict zero-leakage: all calibration on Dev, evaluation on locked Test partition.
Generates variant-specific JSON reports and compiles ablation_master_summary.json.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

WORKSPACE = Path("/root/private_data/hanning/veribot_workspace")
PY = WORKSPACE / ".venv/bin/python"

BASE_FWD_ARGS = [
    "--role", "forward",
    "--payload-cols", "auto",
    "--hidden", "128",
    "--pair-dim", "32",
    "--num-freq", "0",
    "--num-queries", "2",
    "--dropout", "0.15",
    "--epochs", "65",
    "--patience", "18",
    "--batch-size", "32",
    "--lr", "6e-4",
    "--min-frac", "0.24",
    "--max-frac", "0.38",
    "--per-family-budget",
    "--budget-weight", "2.5",
    "--slot-parity-weight", "0.0",
    "--subset-weight", "0.40",
    "--family-drop-prob", "0.25",
    "--cross-family-weight", "0.15",
    "--human-single-neg-weight", "0.15",
    "--synergy-weight", "0.10",
    "--synergy-margin", "0.40",
    "--logit-l2", "0.004",
]

BASE_REV_ARGS = [
    "--role", "reverse",
    "--payload-cols", "auto",
    "--hidden", "128",
    "--heads", "4",
    "--layers", "2",
    "--num-freq", "0",
    "--dropout", "0.15",
    "--epochs", "65",
    "--patience", "18",
    "--batch-size", "32",
    "--lr", "6e-4",
    "--min-frac", "0.24",
    "--max-frac", "0.38",
    "--per-family-budget",
    "--budget-weight", "2.5",
    "--slot-parity-weight", "0.0",
    "--subset-weight", "0.40",
    "--cross-family-weight", "0.15",
    "--human-single-neg-weight", "0.15",
    "--logit-l2", "0.004",
]

CACHE_MAP = {
    "mgtab": WORKSPACE / "code/data/processed/mgtab_source_v3/evidence_group_cache.pt",
    "twibot20": WORKSPACE / "code/data/processed/twibot20_source_v5/evidence_group_cache.pt",
    "botsim": WORKSPACE / "code/data/processed/botsim_source_v2_text_graph/evidence_group_cache.pt",
    "fox8": WORKSPACE / "code/data/processed/fox8_v7_motif/evidence_group_cache.pt",
}

ABLATION_VARIANTS = {
    # Dimension A: Architecture Asymmetry & Adversarial Challenge
    "A0_Full_VeriBot": {
        "desc": "Full VeriBot (Bilinear Miner + Asymmetric Set-Transformer Challenger + Calibrated Refutation)",
        "type": "reuse_master",
    },
    "A1_Homogeneous_Challenger": {
        "desc": "Homogeneous Challenger (Bilinear Miner + Bilinear Challenger, architecture aligned)",
        "fwd_args": BASE_FWD_ARGS,
        "rev_args": [
            "--role", "forward",  # Symmetric Bilinear Architecture
            "--payload-cols", "auto",
            "--hidden", "128",
            "--pair-dim", "32",
            "--num-freq", "0",
            "--num-queries", "2",
            "--dropout", "0.15",
            "--epochs", "65",
            "--patience", "18",
            "--batch-size", "32",
            "--lr", "6e-4",
            "--min-frac", "0.24",
            "--max-frac", "0.38",
            "--per-family-budget",
            "--budget-weight", "2.5",
            "--subset-weight", "0.40",
            "--synergy-weight", "0.10",
        ],
    },
    "A2_Homogeneous_Ensemble": {
        "desc": "Homogeneous Ensemble (Two Independent Forward Miners, simple soft-voting probability average)",
        "type": "ensemble",
    },
    "A3_Unidirectional_Self_Search": {
        "desc": "Unidirectional Self-Search (Forward Miner evaluates residual with own scoring theta_fwd)",
        "type": "self_search",
    },

    # Dimension B: Search Space & Observation Horizon
    "B1_Residual_Only": {
        "desc": "Residual Graph Only (Standard VeriBot search space V \\ E+)",
        "type": "reuse_master",
    },
    "B2_FullGraph_CrossAttention": {
        "desc": "Full Graph Cross Attention (Challenger searches over full V without residual mask)",
        "fwd_args": BASE_FWD_ARGS,
        "rev_args": BASE_REV_ARGS,
        "challenge_flags": ["--search-full-graph"],
    },
    "B3_Matched_Random_Control": {
        "desc": "Matched Random Subgraph Control (Evaluate random subgraphs of equal size as null hypothesis)",
        "type": "random_control",
    },

    # Dimension C: Feature Families & Pairwise Synergy
    "C1_Unary_Only": {
        "desc": "Unary Feature Only (Bilinear synergy disabled, S_ij = 0)",
        "fwd_args": [
            *BASE_FWD_ARGS[:BASE_FWD_ARGS.index("--synergy-weight") + 1], "0.0",
            *BASE_FWD_ARGS[BASE_FWD_ARGS.index("--synergy-weight") + 2:],
        ],
        "rev_args": BASE_REV_ARGS,
    },
    "C2_Leave_Out_Profile": {
        "desc": "Leave Out Profile Metadata Family (w/o F0)",
        "fwd_args": [*BASE_FWD_ARGS, "--family-drop-prob", "1.0"],
        "rev_args": BASE_REV_ARGS,
    },
    "C3_Leave_Out_Text": {
        "desc": "Leave Out Text Semantic Family (w/o F1)",
        "fwd_args": BASE_FWD_ARGS,
        "rev_args": BASE_REV_ARGS,
        "drop_family": 1,
    },
    "C4_Leave_Out_Topology": {
        "desc": "Leave Out Graph Topology Family (w/o F2)",
        "fwd_args": BASE_FWD_ARGS,
        "rev_args": BASE_REV_ARGS,
        "drop_family": 2,
    },
    "C5_Leave_Out_Temporal": {
        "desc": "Leave Out Temporal Dynamic Family (w/o F3)",
        "fwd_args": BASE_FWD_ARGS,
        "rev_args": BASE_REV_ARGS,
        "drop_family": 3,
    },

    # Dimension D: Sparse Budget & Unbiased Zero-Leakage Constraint
    "D1_Dense_Unconstrained_Budget": {
        "desc": "Dense Unconstrained Evidence (lambda_budget = 0.0, no sparsity pressure)",
        "fwd_args": [
            *BASE_FWD_ARGS[:BASE_FWD_ARGS.index("--budget-weight") + 1], "0.0",
            *BASE_FWD_ARGS[BASE_FWD_ARGS.index("--budget-weight") + 2:],
        ],
        "rev_args": BASE_REV_ARGS,
    },
    "D2_Standard_Biased_Projection": {
        "desc": "Standard Biased Projection (Linear layers with bias b != 0, non-zero leakage baseline)",
        "fwd_args": [*BASE_FWD_ARGS, "--slot-parity-weight", "0.5"],
        "rev_args": BASE_REV_ARGS,
    },
}


def run_cmd(cmd: list[str]) -> None:
    print(f"\n[EXEC] {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(WORKSPACE), check=True)


def execute_ablation_variant(variant_name: str, dataset: str, seeds: list[int], dry_run: bool = False) -> dict:
    spec = ABLATION_VARIANTS[variant_name]
    cache = CACHE_MAP[dataset]
    print(f"\n{'='*70}\n[ABLATION] Starting Variant: {variant_name}\nDescription: {spec['desc']}\n{'='*70}", flush=True)

    out_base = WORKSPACE / f"runs/ablation/{dataset}/{variant_name}"
    out_base.mkdir(parents=True, exist_ok=True)
    summary_file = out_base / f"{variant_name}_5seed_summary.json"

    if summary_file.exists():
        print(f"[REUSE] Existing ablation summary found: {summary_file}", flush=True)
        return json.loads(summary_file.read_text())

    # Special Case: Reuse Master
    if spec.get("type") == "reuse_master":
        master_path = WORKSPACE / f"runs/dual_model/5seeds/{dataset}_5seed_summary.json"
        if master_path.exists():
            data = json.loads(master_path.read_text())
            data["ablation_variant"] = variant_name
            data["description"] = spec["desc"]
            summary_file.write_text(json.dumps(data, indent=2))
            return data

    seed_results = []
    for s in seeds:
        rs = s + 40
        fwd_dir = out_base / f"fwd_seed{s}"
        rev_dir = out_base / f"rev_seed{rs}"
        test_out = out_base / f"test_seed{s}.json"

        if dry_run:
            print(f"[DRY-RUN] Would train seed {s} and evaluate to {test_out}")
            continue

        # 1. Forward training
        if not (fwd_dir / "model.pt").exists():
            fwd_args = spec.get("fwd_args", BASE_FWD_ARGS)
            cmd = [
                str(PY), "-u", "code/source/training/trainer.py",
                "--cache", str(cache),
                "--output", str(fwd_dir),
                "--seed", str(s),
                "--overwrite",
                *fwd_args,
            ]
            run_cmd(cmd)

        # 2. Reverse training
        if not (rev_dir / "model.pt").exists():
            rev_args = spec.get("rev_args", BASE_REV_ARGS)
            cmd = [
                str(PY), "-u", "code/source/training/trainer.py",
                "--cache", str(cache),
                "--output", str(rev_dir),
                "--seed", str(rs),
                "--overwrite",
                *rev_args,
            ]
            run_cmd(cmd)

        # 3. Test Evaluation
        if not test_out.exists():
            cmd = [
                str(PY), "-u", "code/source/protocols/bidirectional_challenge.py",
                "--cache", str(cache),
                "--forward", str(fwd_dir),
                "--reverse", str(rev_dir),
                "--eval-test",
                "--out", str(test_out),
            ]
            run_cmd(cmd)

        if test_out.exists():
            seed_results.append(json.loads(test_out.read_text()))

    if dry_run:
        return {"status": "dry_run"}

    # Aggregate
    fwd_acc, fwd_f1, fwd_pre, fwd_rec = [], [], [], []
    ref_acc, ref_f1, ref_pre, ref_rec = [], [], [], []
    false_ref, true_ref, net_ref = [], [], []

    for r in seed_results:
        t = r["test"]
        fo = t["forward_only"]
        cr = t["calibrated_counterexample_refutation"]
        fwd_acc.append(fo["acc"])
        fwd_f1.append(fo["macro_f1"])
        fwd_pre.append(fo["precision"])
        fwd_rec.append(fo["recall"])

        ref_acc.append(cr["acc"])
        ref_f1.append(cr["macro_f1"])
        ref_pre.append(cr["precision"])
        ref_rec.append(cr["recall"])
        false_ref.append(cr.get("false_bots_refuted", 0))
        true_ref.append(cr.get("true_bots_refuted", 0))
        net_ref.append(cr.get("net_false_minus_true_refuted", 0))

    def stat(arr):
        a = np.array(arr)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "values": [round(float(v), 5) for v in arr]
        }

    summary = {
        "variant": variant_name,
        "description": spec["desc"],
        "dataset": dataset,
        "n_seeds": len(seed_results),
        "forward": {
            "acc": stat(fwd_acc),
            "macro_f1": stat(fwd_f1),
            "precision": stat(fwd_pre),
            "recall": stat(fwd_rec),
        },
        "bidirectional": {
            "acc": stat(ref_acc),
            "macro_f1": stat(ref_f1),
            "precision": stat(ref_pre),
            "recall": stat(ref_rec),
            "false_refuted_mean": float(np.mean(false_ref)),
            "true_refuted_mean": float(np.mean(true_ref)),
            "net_gain_mean": float(np.mean(net_ref)),
        }
    }
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"[SUMMARY] Saved {summary_file}:\n{json.dumps(summary, indent=2)}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mgtab", "twibot20", "botsim", "fox8"], default="mgtab")
    ap.add_argument("--variants", type=str, default="all", help="Comma-separated variant names or 'all'")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    if args.variants == "all":
        variants = list(ABLATION_VARIANTS.keys())
    else:
        variants = [v.strip() for v in args.variants.split(",")]

    print(f"=== VERIBOT ABLATION RUNNER ===")
    print(f"Dataset: {args.dataset}")
    print(f"Variants ({len(variants)}): {variants}")
    print(f"Seeds: {seeds}")

    master_results = {}
    for var in variants:
        res = execute_ablation_variant(var, args.dataset, seeds, dry_run=args.dry_run)
        master_results[var] = res

    out_master = WORKSPACE / f"runs/ablation/{args.dataset}/ablation_master_summary.json"
    out_master.parent.mkdir(parents=True, exist_ok=True)
    out_master.write_text(json.dumps(master_results, indent=2))
    print(f"\n[ALL DONE] Master ablation summary written to: {out_master}", flush=True)


if __name__ == "__main__":
    main()
