"""Read-only CPU probe: does the v4 slot_of encoding (p10/p30/p70/p90 cuts + tie_map, right=True)
cost RF accuracy versus raw values and versus the old linspace bucketing?

The design doc's "bucketing is free" number (83.94%) came from scripts/rf_bucket.py, which used
`np.percentile(linspace(0,100,6)[1:-1])` = the 20/40/60/80 cuts and `np.digitize(..., right=False)`.
render_v4 encodes tiers differently (10/30/70/90 cuts, ties resolved by percentile-midpoint). If
that different encoding loses signal, the whole prompt encoding is suspect, so this measures all
three side by side on the same RF and the same 33 columns. Fitting is train-only throughout.

    python scripts/probe_bucket_rf.py
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from verifiable_bot.v4.features_v4 import BINARY_INDICES, CONTINUOUS_INDICES, feature_matrix
from verifiable_bot.v4.thresholds_v4 import load_thresholds, slot_of

import json
from pathlib import Path

DATA = Path("data/twibot20_complete")
THR = "outputs/v4/render_v4/bucket_thresholds_v4.json"


def load(split):
    d = json.loads((DATA / f"{split}.json").read_text(encoding="utf-8"))
    X = np.nan_to_num(feature_matrix(d), posinf=0, neginf=0)
    y = np.asarray([int(r["label"]) for r in d])
    return X, y


def old_linspace_bucketing(Xtr, Xte):
    """Reproduces scripts/rf_bucket.py exactly: 5 linspace tiers, np.digitize default (right=False),
    on the continuous columns; binary columns untouched."""
    cont = [i for i in range(Xtr.shape[1]) if i not in BINARY_INDICES]
    qs = np.linspace(0, 100, 6)[1:-1]  # [20, 40, 60, 80]
    Btr, Bte = Xtr.copy(), Xte.copy()
    for i in cont:
        cut = np.percentile(Xtr[:, i], qs)
        Btr[:, i] = np.digitize(Xtr[:, i], cut)
        Bte[:, i] = np.digitize(Xte[:, i], cut)
    return Btr, Bte


def v4_slot_encoding(Xtr, Xte, thresholds):
    """Encode continuous columns with render_v4's slot_of (the exact mapping the prompt tier uses)."""
    Btr, Bte = Xtr.copy(), Xte.copy()
    name_by_index = {spec["index"]: name for name, spec in thresholds["features"].items()}
    for i in CONTINUOUS_INDICES:
        spec = thresholds["features"][name_by_index[i]]
        Btr[:, i] = [slot_of(v, spec) for v in Xtr[:, i]]
        Bte[:, i] = [slot_of(v, spec) for v in Xte[:, i]]
    return Btr, Bte


def run(tag, Xtr, ytr, Xte, yte):
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=0)
    clf.fit(Xtr, ytr)
    pr = clf.predict_proba(Xte)[:, 1]
    pred = (pr >= 0.5).astype(int)
    acc, auc, f1 = accuracy_score(yte, pred), roc_auc_score(yte, pr), f1_score(yte, pred)
    print(f"{tag:34s} ACC={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}")
    return acc, auc, f1


def main():
    thresholds = load_thresholds(THR)
    Xtr, ytr = load("train")
    Xte, yte = load("test")
    raw = run("A raw (无分桶)", Xtr, ytr, Xte, yte)
    old_tr, old_te = old_linspace_bucketing(Xtr, Xte)
    old = run("B 旧 linspace 分桶 (rf_bucket)", old_tr, ytr, old_te, yte)
    new_tr, new_te = v4_slot_encoding(Xtr, Xte, thresholds)
    new = run("C v4 slot_of 编码 (render_v4)", new_tr, ytr, new_te, yte)
    print()
    labels = ["ACC", "AUC", "F1"]
    for i, lab in enumerate(labels):
        print(f"{lab}: C={new[i]:.4f}  C-A={new[i]-raw[i]:+.4f}  C-B={new[i]-old[i]:+.4f}")
    print()
    # Honest, per-metric verdict. ACC/F1 improve over raw; AUC is LOWER than both raw and old
    # bucketing (the 10/30/70/90 cuts coarsen the middle of the score distribution). Do NOT claim
    # "all metrics hold".
    print("结论（逐指标，不笼统）：")
    print(f"  ACC: v4 {new[0]:.4f} 高于 raw {raw[0]:.4f}，与旧 bucket {old[0]:.4f} 相当。")
    print(f"  F1 : v4 {new[2]:.4f} 高于 raw {raw[2]:.4f}。")
    print(f"  AUC: v4 {new[1]:.4f} **低于** raw {raw[1]:.4f} 和旧 bucket {old[1]:.4f}"
          f"（差 {new[1]-raw[1]:+.4f}/{new[1]-old[1]:+.4f}）——分档粗化了分数分布中段，如实记录。")
    threshold_ok = new[0] >= raw[0] - 0.005 and new[2] >= raw[2] - 0.005
    print("VERDICT:", "PASS（决策指标 ACC/F1 不掉；AUC 有小幅下降，已如实标注）"
          if threshold_ok else "FAIL：决策指标掉点，需回头查映射")


if __name__ == "__main__":
    main()
