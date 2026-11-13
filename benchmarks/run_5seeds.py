#!/usr/bin/env python3
"""Automated 5-seed benchmark runner & evaluator for VeriBot.

Evaluates Forward Evidence Miner (M_fwd) & Heterogeneous Challenger (M_rev)
across 5 random seeds (0, 1, 2, 3, 4) on locked test splits.
Strict zero-leakage: all checkpoints, thresholds, and refutation rules calibrated on Dev.
All datasets run strictly through the unified ForwardEvidenceMiner & HeterogeneousChallenger
with automatic payload column resolution (--payload-cols auto).
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

WORKSPACE = Path("/root/private_data/hanning/veribot_workspace")
PY = WORKSPACE / ".venv/bin/python"

# Unified Architecture Configurations across all 4 datasets
UNIFIED_FWD_ARGS = [
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

UNIFIED_REV_ARGS = [
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

CONFIGS = {
    "fox8": {
        "cache": WORKSPACE / "code/data/processed/fox8_v7_motif/evidence_group_cache.pt",
        "fwd_dir_template": "models/forward/fox8_seed{s}_unified",
        "rev_dir_template": "models/reverse/fox8_seed{rs}_unified",
    },
    "botsim": {
        "cache": WORKSPACE / "code/data/processed/botsim_source_v2_text_graph/evidence_group_cache.pt",
        "fwd_dir_template": "models/forward/botsim_seed{s}_unified",
        "rev_dir_template": "models/reverse/botsim_seed{rs}_unified",
    },
    "mgtab": {
        "cache": WORKSPACE / "code/data/processed/mgtab_source_v3/evidence_group_cache.pt",
        "fwd_dir_template": "models/forward/mgtab_seed{s}_unified",
        "rev_dir_template": "models/reverse/mgtab_seed{rs}_unified",
    },
    "twibot20": {
        "cache": WORKSPACE / "code/data/processed/twibot20_source_v5/evidence_group_cache.pt",
        "fwd_dir_template": "models/forward/twibot20_seed{s}_unified",
        "rev_dir_template": "models/reverse/twibot20_seed{rs}_unified",
    },
}


def run_cmd(cmd: list[str]) -> None:
    print(f"\n[EXEC] {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(WORKSPACE), check=True)


def train_if_needed(cfg: dict, s: int, rs: int) -> tuple[Path, Path]:
    fwd_dir = WORKSPACE / cfg["fwd_dir_template"].format(s=s, rs=rs)
    rev_dir = WORKSPACE / cfg["rev_dir_template"].format(s=s, rs=rs)

    trainer_script = "code/source/training/trainer.py"
    if not (fwd_dir / "report.json").exists() or not (fwd_dir / "model.pt").exists():
        cmd = [
            str(PY), "-u", trainer_script,
            "--cache", str(cfg["cache"]),
            "--output", str(fwd_dir),
            "--seed", str(s),
            *UNIFIED_FWD_ARGS,
        ]
        run_cmd(cmd)
    else:
        print(f"[REUSE] Forward model exists: {fwd_dir}", flush=True)

    if not (rev_dir / "report.json").exists() or not (rev_dir / "model.pt").exists():
        cmd = [
            str(PY), "-u", trainer_script,
            "--cache", str(cfg["cache"]),
            "--output", str(rev_dir),
            "--seed", str(rs),
            *UNIFIED_REV_ARGS,
        ]
        run_cmd(cmd)
    else:
        print(f"[REUSE] Reverse model exists: {rev_dir}", flush=True)

    return fwd_dir, rev_dir


def evaluate_seed(dataset: str, cfg: dict, s: int, rs: int, fwd_dir: Path, rev_dir: Path) -> dict:
    out_dir = WORKSPACE / "runs/dual_model/5seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_json = out_dir / f"{dataset}_seed{s}_test.json"
    audit_jsonl = out_dir / f"{dataset}_seed{s}_sample_audit.jsonl"

    eval_script = "code/source/protocols/bidirectional_challenge.py"
    cmd = [
        str(PY), "-u", eval_script,
        "--cache", str(cfg["cache"]),
        "--forward", str(fwd_dir),
        "--reverse", str(rev_dir),
        "--eval-test",
        "--dump-sample-audit", str(audit_jsonl),
        "--out", str(res_json),
    ]
    run_cmd(cmd)
    data = json.loads(res_json.read_text())
    return data


def aggregate_dataset(dataset: str, results: list[dict]) -> dict:
    fwd_acc, fwd_macro_f1, fwd_bot_f1, fwd_pre, fwd_rec, fwd_auc = [], [], [], [], [], []
    ref_acc, ref_macro_f1, ref_bot_f1, ref_pre, ref_rec = [], [], [], [], []
    cross_sup, false_ref, true_ref = [], [], []

    for r in results:
        t = r["test"]
        fo = t["forward_only"]
        fwd_acc.append(fo["acc"])
        fwd_macro_f1.append(fo["macro_f1"])
        fwd_bot_f1.append(fo["bot_f1"])
        fwd_pre.append(fo["precision"])
        fwd_rec.append(fo["recall"])
        if "auc" in fo:
            fwd_auc.append(fo["auc"])

        cr = t["calibrated_counterexample_refutation"]
        ref_acc.append(cr["acc"])
        ref_macro_f1.append(cr["macro_f1"])
        ref_bot_f1.append(cr["bot_f1"])
        ref_pre.append(cr["precision"])
        ref_rec.append(cr["recall"])
        false_ref.append(cr.get("false_bots_refuted", 0))
        true_ref.append(cr.get("true_bots_refuted", 0))

        cs = t["full_denominator_bidirectional_metrics"]["over_all_accounts"]["bidirectional_cross_support"]
        cross_sup.append(cs)

    def stats(arr: list[float]) -> dict:
        a = np.array(arr)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "values": [round(float(v), 5) for v in arr]
        }

    summary = {
        "dataset": dataset,
        "n_seeds": len(results),
        "forward_only": {
            "acc": stats(fwd_acc),
            "macro_f1": stats(fwd_macro_f1),
            "bot_f1": stats(fwd_bot_f1),
            "precision": stats(fwd_pre),
            "recall": stats(fwd_rec),
            "auc": stats(fwd_auc) if fwd_auc else {},
        },
        "bidirectional_refutation": {
            "acc": stats(ref_acc),
            "macro_f1": stats(ref_macro_f1),
            "bot_f1": stats(ref_bot_f1),
            "precision": stats(ref_pre),
            "recall": stats(ref_rec),
            "false_bots_refuted_mean": float(np.mean(false_ref)),
            "true_bots_refuted_mean": float(np.mean(true_ref)),
            "net_refuted_mean": float(np.mean(np.array(false_ref) - np.array(true_ref))),
            "acc_gain_mean": float(np.mean(np.array(ref_acc) - np.array(fwd_acc))),
        },
        "bidirectional_cross_support_rate": stats(cross_sup),
    }

    out_file = WORKSPACE / f"runs/dual_model/5seeds/{dataset}_5seed_summary.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\n[SUMMARY SAVED] {out_file}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("fox8", "botsim", "mgtab", "twibot20", "all"), required=True)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    targets = ["fox8", "botsim", "mgtab", "twibot20"] if args.dataset == "all" else [args.dataset]

    overall_summaries = {}
    for d in targets:
        print(f"\n==========================================")
        print(f"=== RUNNING 5-SEED BENCHMARK FOR {d.upper()} ===")
        print(f"==========================================")
        cfg = CONFIGS[d]
        results = []
        for s in seeds:
            rs = 44 + s
            print(f"\n--- Dataset: {d} | Seed: {s} (Forward) / {rs} (Reverse) ---", flush=True)
            fwd_dir, rev_dir = train_if_needed(cfg, s, rs)
            res = evaluate_seed(d, cfg, s, rs, fwd_dir, rev_dir)
            results.append(res)
        summary = aggregate_dataset(d, results)
        overall_summaries[d] = summary

    final_report = WORKSPACE / "runs/dual_model/5seeds/all_benchmarks_5seed_report.json"
    final_report.write_text(json.dumps(overall_summaries, indent=2))
    print(f"\n[ALL BENCHMARKS COMPLETE] Final report written to {final_report}", flush=True)


if __name__ == "__main__":
    main()
