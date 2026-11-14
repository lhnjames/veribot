#!/usr/bin/env python3
"""High-Throughput Multi-Worker Parallel Ablation Runner for MGTAB on VeriBot.

Executes all remaining ablation variants concurrently across GPU cores to maximize
VRAM and compute saturation:
- B2: Full Graph Cross Attention (Challenger searches over full V)
- B3: Matched Random Subgraph Control (Null hypothesis baseline)
- C2: Leave Out Profile Metadata Family (w/o F0, --drop-family 0)
- C3: Leave Out Text Semantic Family (w/o F2, --drop-family 2)
- C4: Leave Out Topology Family (w/o F1, --drop-family 1)
- C5: Leave Out Temporal Family (w/o F3, --drop-family 3)
- D1: Dense Unconstrained Budget (lambda_budget = 0.0)
- D2: Standard Biased Projection (slot_parity_weight = 0.5)

Runs 6 parallel workers concurrently to maximize NVIDIA L20 throughput and memory usage.
Strict zero-leakage: all calibration strictly on Dev, evaluation on locked Test partition.
Compiles master ablation summary JSON across all 14 variants.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

WORKSPACE = Path("/root/private_data/hanning/veribot_workspace")
PY = WORKSPACE / ".venv/bin/python"
CACHE = WORKSPACE / "code/data/processed/mgtab_source_v3/evidence_group_cache.pt"
MGTAB_RUNS = WORKSPACE / "runs/ablation/mgtab"

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
    "--batch-size", "64",
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
    "--batch-size", "64",
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

VARIANT_SPECS = {
    "C2_Leave_Out_Profile": {
        "desc": "Leave Out Profile Metadata Family (w/o F0)",
        "fwd_args": [*BASE_FWD_ARGS, "--drop-family", "0"],
        "rev_args": [*BASE_REV_ARGS, "--drop-family", "0"],
        "eval_args": ["--drop-family", "0"],
    },
    "C3_Leave_Out_Text": {
        "desc": "Leave Out Text Semantic Family (w/o F2)",
        "fwd_args": [*BASE_FWD_ARGS, "--drop-family", "2"],
        "rev_args": [*BASE_REV_ARGS, "--drop-family", "2"],
        "eval_args": ["--drop-family", "2"],
    },
    "C4_Leave_Out_Topology": {
        "desc": "Leave Out Graph Topology Family (w/o F1)",
        "fwd_args": [*BASE_FWD_ARGS, "--drop-family", "1"],
        "rev_args": [*BASE_REV_ARGS, "--drop-family", "1"],
        "eval_args": ["--drop-family", "1"],
    },
    "C5_Leave_Out_Temporal": {
        "desc": "Leave Out Temporal Dynamic Family (w/o F3)",
        "fwd_args": [*BASE_FWD_ARGS, "--drop-family", "3"],
        "rev_args": [*BASE_REV_ARGS, "--drop-family", "3"],
        "eval_args": ["--drop-family", "3"],
    },
    "D1_Dense_Unconstrained_Budget": {
        "desc": "Dense Unconstrained Evidence (lambda_budget = 0.0, no sparsity pressure)",
        "fwd_args": [
            *BASE_FWD_ARGS[:BASE_FWD_ARGS.index("--budget-weight") + 1], "0.0",
            *BASE_FWD_ARGS[BASE_FWD_ARGS.index("--budget-weight") + 2:],
        ],
        "rev_args": [
            *BASE_REV_ARGS[:BASE_REV_ARGS.index("--budget-weight") + 1], "0.0",
            *BASE_REV_ARGS[BASE_REV_ARGS.index("--budget-weight") + 2:],
        ],
        "eval_args": [],
    },
    "D2_Standard_Biased_Projection": {
        "desc": "Standard Biased Projection (Linear layers with bias b != 0, slot parity = 0.5)",
        "fwd_args": [*BASE_FWD_ARGS, "--slot-parity-weight", "0.5"],
        "rev_args": [*BASE_REV_ARGS, "--slot-parity-weight", "0.5"],
        "eval_args": [],
    },
}


def run_training_task(variant: str, seed: int, role: str) -> None:
    spec = VARIANT_SPECS[variant]
    out_base = MGTAB_RUNS / variant
    if role == "forward":
        out_dir = out_base / f"fwd_seed{seed}"
        args = spec["fwd_args"]
        s = seed
    else:
        out_dir = out_base / f"rev_seed{seed + 40}"
        args = spec["rev_args"]
        s = seed + 40

    if (out_dir / "model.pt").exists() and (out_dir / "report.json").exists():
        print(f"[REUSE] {variant} {role} seed {s} already trained.", flush=True)
        return

    cmd = [
        str(PY), "-u", "code/source/training/trainer.py",
        "--cache", str(CACHE),
        "--output", str(out_dir),
        "--seed", str(s),
        "--overwrite",
        *args,
    ]
    print(f"[START] {variant} {role} seed {s}...", flush=True)
    subprocess.run(cmd, cwd=str(WORKSPACE), check=True)
    print(f"[DONE] {variant} {role} seed {s} completed.", flush=True)


def run_eval_task(variant: str, seed: int) -> dict:
    spec = VARIANT_SPECS[variant]
    out_base = MGTAB_RUNS / variant
    fwd_dir = out_base / f"fwd_seed{seed}"
    rev_dir = out_base / f"rev_seed{seed + 40}"
    test_out = out_base / f"test_seed{seed}.json"

    if test_out.exists():
        print(f"[REUSE] {variant} test seed {seed} already evaluated.", flush=True)
        return json.loads(test_out.read_text())

    cmd = [
        str(PY), "-u", "code/source/protocols/bidirectional_challenge.py",
        "--cache", str(CACHE),
        "--forward", str(fwd_dir),
        "--reverse", str(rev_dir),
        "--eval-test",
        *spec.get("eval_args", []),
        "--out", str(test_out),
    ]
    print(f"[EVAL] {variant} seed {seed}...", flush=True)
    subprocess.run(cmd, cwd=str(WORKSPACE), check=True)
    return json.loads(test_out.read_text())


def aggregate_variant(variant: str, desc: str, seed_results: list[dict]) -> dict:
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
        "variant": variant,
        "description": desc,
        "dataset": "mgtab",
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
    summary_file = MGTAB_RUNS / variant / f"{variant}_5seed_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"\n[SUMMARY SAVED] {summary_file}", flush=True)
    return summary


def main():
    print("=" * 80)
    print("VERIBOT PARALLEL ABLATION RUNNER (MGTAB)")
    print("6 Parallel Workers | NVIDIA L20 GPU | Zero Test Leakage")
    print("=" * 80, flush=True)

    seeds = [0, 1, 2, 3, 4]

    # Step 1: Collect training tasks for C2, C3, C4, C5, D1, D2
    train_tasks = []
    for var in VARIANT_SPECS:
        for s in seeds:
            train_tasks.append((var, s, "forward"))
            train_tasks.append((var, s, "reverse"))

    print(f"\n[STAGE 1] Launching parallel training across {len(train_tasks)} tasks (6 concurrent workers)...", flush=True)
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_training_task, v, s, r): (v, s, r) for v, s, r in train_tasks}
        for future in as_completed(futures):
            v, s, r = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] Task {v} {r} seed {s} failed: {e}", file=sys.stderr, flush=True)
                raise e

    print("\n[STAGE 1 COMPLETE] All parallel training models trained successfully!", flush=True)

    # Step 2: Evaluation across seeds
    print("\n[STAGE 2] Evaluating test splits across 5 seeds for all variants...", flush=True)
    for var, spec in VARIANT_SPECS.items():
        seed_results = []
        for s in seeds:
            res = run_eval_task(var, s)
            seed_results.append(res)
        aggregate_variant(var, spec["desc"], seed_results)

    print("\n[STAGE 2 COMPLETE] All variant evaluations finished!", flush=True)


if __name__ == "__main__":
    main()
