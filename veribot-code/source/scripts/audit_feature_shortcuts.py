"""Check whether simple non-graph feature blocks explain dataset performance.

Exploratory logistic baselines use train labels and dev-only threshold selection.
They do not change the already frozen HGT/RelNet checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def metrics(probs, labels, threshold):
    pred = probs >= threshold
    return {"n": len(labels), "accuracy": float(accuracy_score(labels, pred)),
            "auc": float(roc_auc_score(labels, probs)),
            "macro_f1": float(f1_score(labels, pred, average="macro")),
            "threshold": threshold}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = torch.load(args.graph, map_location="cpu", weights_only=False)
    x = data["user"].x.numpy()
    y = data["user"].y.numpy()
    masks = {split: data["user"][f"{split}_mask"].numpy()
             for split in ("train", "dev", "test")}
    if x.shape[1] != 2053:
        raise RuntimeError("expected 1024 + 1024 + 5 input blocks")
    blocks = {"numeric_only": x[:, 2048:], "description_only": x[:, :1024],
              "content_only": x[:, 1024:2048], "all_features_no_graph": x}
    report = {"scope": "exploratory non-graph logistic baselines",
              "graph": str(args.graph), "blocks": {}}
    for name, features in blocks.items():
        model = LogisticRegression(solver="liblinear", class_weight="balanced",
                                   C=1.0, max_iter=500, random_state=42)
        model.fit(features[masks["train"]], y[masks["train"]])
        dev_prob = model.predict_proba(features[masks["dev"]])[:, 1]
        test_prob = model.predict_proba(features[masks["test"]])[:, 1]
        thresholds = np.linspace(0.05, 0.95, 100)
        threshold = float(max(thresholds, key=lambda t: accuracy_score(
            y[masks["dev"]], dev_prob >= t)))
        report["blocks"][name] = {
            "dev": metrics(dev_prob, y[masks["dev"]], threshold),
            "test": metrics(test_prob, y[masks["test"]], threshold),
            "iterations": int(model.n_iter_[0]),
        }
        print(name, report["blocks"][name], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
