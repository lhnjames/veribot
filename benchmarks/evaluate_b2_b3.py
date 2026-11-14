#!/usr/bin/env python3
"""Evaluate Dimension B Ablations (B2: Full Graph Search & B3: Matched Random Control) for MGTAB.

Dimension B tests the search space and observation horizon of the Heterogeneous Challenger:
- B1: Residual Graph Only (Standard VeriBot, V \\ E+) -> already in A0 summary
- B2: Full Graph Cross Attention (Challenger searches over full V without residual mask)
- B3: Matched Random Control (Null hypothesis: evaluates random subgraphs of equal size)

Both variants are calibrated strictly on Dev (with 5-fold cross-validation guard) and
evaluated on locked Test across 5 random seeds (0, 1, 2, 3, 4).
Outputs variant-specific JSON reports and 5-seed summaries.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

WORKSPACE = Path("/root/private_data/hanning/veribot_workspace")
sys.path.insert(0, str(WORKSPACE / "code/source"))
from protocols.bidirectional_challenge import (
    load_any_model,
    collect_split_records,
)

CACHE = WORKSPACE / "code/data/processed/mgtab_source_v3/evidence_group_cache.pt"
MGTAB_RUNS = WORKSPACE / "runs/ablation/mgtab"


def calibrate_and_eval_variant(dev_records: list[dict], test_records: list[dict], ft: float, rt: float, mode: str) -> dict:
    """mode in ('full_graph', 'random_control')"""
    y_dev = np.array([r["label"] for r in dev_records], dtype=int)
    f_score_dev = np.array([r["f_score"] for r in dev_records], dtype=float)
    r_score_dev = np.array([r["r_score"] for r in dev_records], dtype=float)
    r_on_f_dev = np.array([r["r_on_f"] for r in dev_records], dtype=float)
    h_ce_dev = np.array([r["h_ce_own"] for r in dev_records], dtype=float)

    if mode == "full_graph":
        ce_signal_dev = np.array([r["rev_search_max_a"] for r in dev_records], dtype=float)
    elif mode == "random_control":
        ce_signal_dev = np.array([r["rand_search_max_a"] for r in dev_records], dtype=float)
    else:
        raise ValueError(f"Unknown mode {mode}")

    base_pred_dev = f_score_dev >= ft
    base_f1_dev = f1_score(y_dev, base_pred_dev, pos_label=1)
    base_acc_dev = accuracy_score(y_dev, base_pred_dev)

    best_rule = {
        "fwd_cap": 0.0,
        "chal_thresh": -1e9,
        "w_own": 0.4,
        "w_onf": 0.4,
        "w_ce": 0.2,
        "w_hce": 0.1,
    }
    best_key = (0, round(base_f1_dev, 6), round(base_acc_dev, 6), 0)

    n_folds = 5
    bot_idx = np.where(y_dev == 1)[0]
    hum_idx = np.where(y_dev == 0)[0]
    fold_splits = [np.concatenate([bot_idx[k::n_folds], hum_idx[k::n_folds]]) for k in range(n_folds)]

    for w_own, w_onf, w_ce, w_hce in (
        (0.4, 0.4, 0.2, 0.15),
        (0.5, 0.3, 0.2, 0.20),
        (0.3, 0.5, 0.2, 0.10),
        (0.6, 0.2, 0.2, 0.25),
        (0.0, 1.0, 0.0, 0.0),
    ):
        chal_dev = (
            w_own * (r_score_dev - rt)
            + w_onf * (r_on_f_dev - rt)
            + w_ce * (ce_signal_dev - ft)
            - w_hce * np.maximum(0.0, h_ce_dev)
        )
        for fwd_cap in (0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 10.0):
            for c_th in np.linspace(-2.5, 0.5, 61):
                refute_dev = base_pred_dev & ((f_score_dev - ft) < fwd_cap) & (chal_dev < c_th)
                if not refute_dev.any():
                    continue

                fold_net_gains = []
                for val_idx in fold_splits:
                    v_ref = refute_dev[val_idx]
                    v_y = y_dev[val_idx]
                    v_fp = int(np.sum(v_ref & (v_y == 0)))
                    v_tp = int(np.sum(v_ref & (v_y == 1)))
                    fold_net_gains.append(v_fp - v_tp)

                if any(gain < 0 for gain in fold_net_gains):
                    continue

                pred_dev = base_pred_dev & (~refute_dev)
                b_f1 = float(f1_score(y_dev, pred_dev, pos_label=1))
                acc = float(accuracy_score(y_dev, pred_dev))
                fp_ref = int(np.sum(refute_dev & (y_dev == 0)))
                tp_ref = int(np.sum(refute_dev & (y_dev == 1)))
                min_fold_gain = min(fold_net_gains)
                key = (min_fold_gain, round(b_f1, 6), round(acc, 6), fp_ref - tp_ref)
                if key > best_key:
                    best_key = key
                    best_rule = {
                        "fwd_cap": float(fwd_cap),
                        "chal_thresh": float(c_th),
                        "w_own": float(w_own),
                        "w_onf": float(w_onf),
                        "w_ce": float(w_ce),
                        "w_hce": float(w_hce),
                    }

    # Evaluate on Locked Test
    y_test = np.array([r["label"] for r in test_records], dtype=int)
    f_score_test = np.array([r["f_score"] for r in test_records], dtype=float)
    r_score_test = np.array([r["r_score"] for r in test_records], dtype=float)
    r_on_f_test = np.array([r["r_on_f"] for r in test_records], dtype=float)
    h_ce_test = np.array([r["h_ce_own"] for r in test_records], dtype=float)

    if mode == "full_graph":
        ce_signal_test = np.array([r["rev_search_max_a"] for r in test_records], dtype=float)
    else:
        ce_signal_test = np.array([r["rand_search_max_a"] for r in test_records], dtype=float)

    base_pred_test = f_score_test >= ft
    chal_test = (
        best_rule["w_own"] * (r_score_test - rt)
        + best_rule["w_onf"] * (r_on_f_test - rt)
        + best_rule["w_ce"] * (ce_signal_test - ft)
        - best_rule["w_hce"] * np.maximum(0.0, h_ce_test)
    )
    refute_test = base_pred_test & ((f_score_test - ft) < best_rule["fwd_cap"]) & (chal_test < best_rule["chal_thresh"])
    pred_test = base_pred_test & (~refute_test)

    fp_refuted = int(np.sum(refute_test & (y_test == 0)))
    tp_refuted = int(np.sum(refute_test & (y_test == 1)))
    net_gain = fp_refuted - tp_refuted

    return {
        "dev_locked_rule": best_rule,
        "forward_only": {
            "acc": float(accuracy_score(y_test, base_pred_test)),
            "macro_f1": float(f1_score(y_test, base_pred_test, average="macro")),
            "precision": float(precision_score(y_test, base_pred_test, zero_division=0)),
            "recall": float(recall_score(y_test, base_pred_test, zero_division=0)),
        },
        "calibrated_counterexample_refutation": {
            "acc": float(accuracy_score(y_test, pred_test)),
            "macro_f1": float(f1_score(y_test, pred_test, average="macro")),
            "precision": float(precision_score(y_test, pred_test, zero_division=0)),
            "recall": float(recall_score(y_test, pred_test, zero_division=0)),
            "false_bots_refuted": fp_refuted,
            "true_bots_refuted": tp_refuted,
            "net_false_minus_true_refuted": net_gain,
        }
    }


def aggregate_and_save(variant: str, desc: str, seed_results: list[dict]) -> dict:
    fwd_acc, fwd_f1, fwd_pre, fwd_rec = [], [], [], []
    ref_acc, ref_f1, ref_pre, ref_rec = [], [], [], []
    false_ref, true_ref, net_ref = [], [], []

    for r in seed_results:
        fo = r["forward_only"]
        cr = r["calibrated_counterexample_refutation"]
        fwd_acc.append(fo["acc"])
        fwd_f1.append(fo["macro_f1"])
        fwd_pre.append(fo["precision"])
        fwd_rec.append(fo["recall"])

        ref_acc.append(cr["acc"])
        ref_f1.append(cr["macro_f1"])
        ref_pre.append(cr["precision"])
        ref_rec.append(cr["recall"])
        false_ref.append(cr["false_bots_refuted"])
        true_ref.append(cr["true_bots_refuted"])
        net_ref.append(cr["net_false_minus_true_refuted"])

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
    out_dir = MGTAB_RUNS / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{variant}_5seed_summary.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\n[SUMMARY SAVED] {out_file}", flush=True)
    return summary


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL B2 & B3] Using device: {device}")

    cache = torch.load(CACHE, map_location="cpu", weights_only=False)

    b2_results = []
    b3_results = []

    for s in range(5):
        fwd_dir = WORKSPACE / f"models/forward/mgtab_seed{s}_unified"
        rev_dir = WORKSPACE / f"models/reverse/mgtab_seed{s+44}_unified"

        f_model, f_kind, ft = load_any_model(fwd_dir, device)
        r_model, r_kind, rt = load_any_model(rev_dir, device)

        print(f"Seed {s}: Collecting Dev & Test records (search_budget=4)...", flush=True)
        dev_records = collect_split_records(cache, "dev", f_model, f_kind, r_model, r_kind, device, search_budget=4)
        test_records = collect_split_records(cache, "test", f_model, f_kind, r_model, r_kind, device, search_budget=4)

        b2_res = calibrate_and_eval_variant(dev_records, test_records, ft, rt, mode="full_graph")
        b2_results.append(b2_res)
        (MGTAB_RUNS / "B2_FullGraph_CrossAttention" / f"test_seed{s}.json").write_text(json.dumps({"test": b2_res}, indent=2))

        b3_res = calibrate_and_eval_variant(dev_records, test_records, ft, rt, mode="random_control")
        b3_results.append(b3_res)
        (MGTAB_RUNS / "B3_Matched_Random_Control" / f"test_seed{s}.json").write_text(json.dumps({"test": b3_res}, indent=2))

    aggregate_and_save(
        "B2_FullGraph_CrossAttention",
        "Full Graph Cross Attention (Challenger searches over full V without residual mask)",
        b2_results,
    )
    aggregate_and_save(
        "B3_Matched_Random_Control",
        "Matched Random Subgraph Control (Evaluate random subgraphs of equal size as null hypothesis)",
        b3_results,
    )
    print("\n[COMPLETE] B2 and B3 evaluated and locked across 5 seeds!")


if __name__ == "__main__":
    main()
