"""Evaluate one dev-selected native-follow HGT on the official test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/hgt_import/code"))
from hgt_model import VerifiableHGT

from train_twibot20_native_hgt_dev import load_text_features, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--text-features", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    checkpoint_path = args.run / "model.pt"
    report_path = args.run / "report.json"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dev_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (state["bundle_sha256"] != sha256(args.bundle)
            or dev_report["checkpoint_sha256"] != sha256(checkpoint_path)):
        raise ValueError("frozen model or bundle differs from dev selection")
    with np.load(args.bundle, allow_pickle=False) as bundle:
        raw = bundle["values"].astype(np.float64)
        labels = bundle["labels"].astype(np.int64)
        split = bundle["split"].astype(str)
        ids = bundle["ids"].astype(str)
        edges = torch.from_numpy(bundle["edge_index"].astype(np.int64)).to(device)
        names = bundle["names"].astype(str).tolist()
    text_sha = state.get("text_features_sha256")
    if text_sha is not None:
        if args.text_features is None or sha256(args.text_features) != text_sha:
            raise ValueError("frozen text features differ")
        text_values, text_names = load_text_features(
            args.text_features, ids, sha256(args.bundle))
        raw = np.column_stack((raw, text_values))
        names.extend(text_names)
    elif args.text_features is not None:
        raise ValueError("numeric-only checkpoint cannot use text features")
    if names != state["feature_names"]:
        raise ValueError("feature order differs")
    medians = np.asarray(state["train_medians"], dtype=np.float64)
    mean = np.asarray(state["scaler_mean"], dtype=np.float64)
    scale = np.asarray(state["scaler_scale"], dtype=np.float64)
    x = np.clip((np.where(np.isfinite(raw), raw, medians) - mean) / scale,
                -3, 3).astype(np.float32)
    graph = {
        ("user", "follows", "user"): edges,
        ("user", "rev_follows", "user"): edges.flip(0),
    }
    model = VerifiableHGT((['user'], list(graph)), {"user": len(names)},
                          hidden_channels=state["hidden"], out_channels=1,
                          num_layers=2, heads=4, dropout=0.3).to(device).eval()
    model.load_state_dict(state["model_state"])
    with torch.inference_mode():
        logits = model({"user": torch.from_numpy(x).to(device)}, graph)[
            "bot_logit"].detach().cpu().numpy()
    test = np.flatnonzero(split == "test")
    truth = labels[test]
    score = logits[test]
    threshold = state["dev_logit_threshold"]
    predicted = (score >= threshold).astype(np.int64)
    result = {
        "checkpoint_sha256": sha256(checkpoint_path),
        "bundle_sha256": sha256(args.bundle),
        "text_features_sha256": text_sha,
        "threshold_from_dev": threshold,
        "test": {
            "n": int(len(test)),
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, average="macro")),
            "bot_f1": float(f1_score(truth, predicted, pos_label=1)),
            "auc": float(roc_auc_score(truth, score)),
        },
        "predictions": [
            {"account_id": ids[i], "label": int(labels[i]),
             "bot_logit": float(logits[i]), "predicted_bot": int(p)}
            for i, p in zip(test, predicted)
        ],
    }
    (args.run / "frozen_test_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "predictions"}), flush=True)


if __name__ == "__main__":
    main()
