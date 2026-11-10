"""Train a typed native-relation GNN on the same 31 named Fox8 facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/scripts"))
sys.path.insert(0, str(ROOT / "source/hgt_import/code"))
from relnet_reimplementation import GatedRelNet
from train_v2_detector import file_sha256, load_data
from verifiable_bot.fox8_adapter import FEATURE_FIELDS
from verifiable_bot.fox8_relational_graph import RELATIONS, typed_native_edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/fox8_rel_evidence_gnn/seed0")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    raw = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    events = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
    loader_args = argparse.Namespace(
        dataset="fox8", data_dir=Path("D:/Code/SAHG/SAHG/code/data/fox8_23"),
        account_file=None, raw_source=[raw], knn_k=10)
    x, labels, split, ids, _, _, hashes, scaler = load_data(loader_args, 0)
    if x.shape[1] != len(FEATURE_FIELDS):
        raise ValueError("named fact count differs from GNN input width")
    typed, graph_hash = typed_native_edges(events, ids)
    model = GatedRelNet(x.shape[1], hidden_channels=64,
                        relation_names=RELATIONS, num_layers=2, dropout=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    y = torch.as_tensor(labels, dtype=torch.float32)
    train = torch.as_tensor(split["train"])
    val = torch.as_tensor(split["val"])
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stalled = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, typed)
        loss = F.binary_cross_entropy_with_logits(logits[train], y[train])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(x, typed)
            val_loss = float(F.binary_cross_entropy_with_logits(logits[val], y[val]))
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: value.detach().cpu().clone()
                          for k, value in model.state_dict().items()}
            best_epoch = epoch
            stalled = 0
        else:
            stalled += 1
        if epoch % 10 == 0 or epoch == 1:
            print(json.dumps({"epoch": epoch, "train_loss": float(loss),
                              "val_loss": val_loss, "best_epoch": best_epoch}), flush=True)
        if stalled >= args.patience:
            break
    if best_state is None:
        raise ValueError("training produced no valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        logits = model(x, typed)
        probability = torch.sigmoid(logits).numpy()
    test = split["test"]
    predicted = probability[test] >= 0.5
    acc = float(np.mean(predicted == labels[test]))
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "model.pt"
    torch.save({"model_state": best_state, "input_dim": x.shape[1],
                "hidden_channels": 64, "num_layers": 2, "dropout": 0.2,
                "relation_names": RELATIONS, "best_epoch": best_epoch,
                "best_val_loss": best_loss, "feature_names": FEATURE_FIELDS,
                "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                "feature_sha256": hashes["feature_sha256"],
                "split_sha256": hashes["split_sha256"],
                "native_graph_sha256": graph_hash,
                "native_event_index_sha256": file_sha256(events),
                "raw_source_sha256": file_sha256(raw)}, checkpoint_path)
    report = {"model": "GatedRelNet-native-relations", "seed": 0,
              "checkpoint_sha256": file_sha256(checkpoint_path),
              "node_feature_sha256": hashes["feature_sha256"],
              "split_sha256": hashes["split_sha256"],
              "native_graph_sha256": graph_hash,
              "native_event_index_sha256": file_sha256(events),
              "best_epoch": best_epoch, "best_val_loss": best_loss,
              "test_count": len(test), "test_accuracy_at_fixed_0_5": acc,
              "predictions": [{"account_id": ids[int(i)],
                               "probability": float(probability[i]),
                               "label": int(labels[i])} for i in test]}
    (args.output / "test_predictions.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}), flush=True)


if __name__ == "__main__":
    main()
