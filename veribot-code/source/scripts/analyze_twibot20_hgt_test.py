"""Audit frozen HGT official-test predictions by train-graph connectivity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def metrics(rows: list[dict]) -> dict:
    labels = [int(row["label"]) for row in rows]
    preds = [int(row["prediction"]) for row in rows]
    probs = [float(row["bot_probability"]) for row in rows]
    return {
        "n": len(rows), "bots": sum(labels), "humans": len(labels) - sum(labels),
        "accuracy": float(accuracy_score(labels, preds)),
        "auc": float(roc_auc_score(labels, probs)) if len(set(labels)) == 2 else None,
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "confusion_matrix_human_bot": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = json.loads(args.train_json.read_text(encoding="utf-8"))
    test = json.loads(args.test_json.read_text(encoding="utf-8"))
    train_ids = {str(row["ID"]) for row in train}
    test_ids = {str(row["ID"]) for row in test}
    touched = set()
    for row in train:
        neighbor = row.get("neighbor") or {}
        touched.update(str(uid) for uid in neighbor.get("following") or [] if str(uid) in test_ids)
        touched.update(str(uid) for uid in neighbor.get("follower") or [] if str(uid) in test_ids)
    for row in test:
        uid = str(row["ID"])
        neighbor = row.get("neighbor") or {}
        if any(str(other) in train_ids for other in
               (neighbor.get("following") or []) + (neighbor.get("follower") or [])):
            touched.add(uid)
    predictions = [json.loads(line) for line in args.predictions.open(encoding="utf-8")]
    assert len(predictions) == len(test)
    assert {row["account_id"] for row in predictions} == test_ids
    by_id = {row["account_id"]: row for row in predictions}
    for row in test:
        assert int(row["label"]) == int(by_id[str(row["ID"])]["label"])
    connected = [row for row in predictions if row["account_id"] in touched]
    disconnected = [row for row in predictions if row["account_id"] not in touched]
    report = {
        "scope": "frozen HGT official TwiBot-20 test under train-test edge attachment protocol",
        "all": metrics(predictions),
        "with_train_graph_edge": metrics(connected),
        "without_train_graph_edge": metrics(disconnected),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
