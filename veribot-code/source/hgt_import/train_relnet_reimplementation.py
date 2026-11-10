"""Train a new, explicitly non-historical RelNet branch on the frozen HGT graph.

Uses exactly the HGT account features, graph, and train/validation/holdout IDs.
The HGT/RelNet conflict experiment can then compare two separately trained
architectures without reusing test labels for selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from relnet_reimplementation import GatedRelNet  # noqa: E402


def val_uid(uid: str) -> bool:
    return int(hashlib.sha256(f"hgt_val_v1:{uid}".encode()).hexdigest()[:8], 16) % 20 == 0


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha_pos: float) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    alpha = alpha_pos * targets + (1 - alpha_pos) * (1 - targets)
    return (alpha * (1 - pt).square() * ce).mean()


def threshold_on_val(probs: np.ndarray, labels: np.ndarray) -> float:
    best_acc, best_t = -1.0, 0.5
    for value in np.linspace(0.05, 0.95, 100):
        acc = accuracy_score(labels, probs >= value)
        if acc > best_acc:
            best_acc, best_t = acc, float(value)
    return best_t


def metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = probs >= threshold
    return {
        "n": len(labels), "accuracy": float(accuracy_score(labels, pred)),
        "auc": float(roc_auc_score(labels, probs)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("train the new RelNet only on the remote CUDA server")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data = torch.load(args.graph, map_location="cpu", weights_only=False)
    ids = data["user"].account_id
    labels = data["user"].y.long()
    holdout = data["user"].holdout_mask.bool()
    val = torch.tensor([val_uid(uid) for uid in ids], dtype=torch.bool) & ~holdout
    train = ~holdout & ~val
    assert int(train.sum() + val.sum() + holdout.sum()) == len(ids)
    assert (int(train.sum()), int(val.sum()), int(holdout.sum())) == (7444, 410, 424)
    alpha_pos = float((labels[train] == 0).float().mean())

    x = data["user"].x.cuda()
    edges = {
        "follows": data["user", "follows", "user"].edge_index.cuda(),
        "rev_follows": data["user", "rev_follows", "user"].edge_index.cuda(),
    }
    y = labels.float().cuda()
    train_gpu = train.cuda()
    model = GatedRelNet(input_channels=x.shape[1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)
    best_auc, best_epoch, best_state, patience = -1.0, 0, None, 0
    labels_np = labels.numpy()
    val_np = val.numpy()
    print(f"New RelNet reimplementation: nodes={len(ids)} features={x.shape[1]} "
          f"train={int(train.sum())} val={int(val.sum())} holdout={int(holdout.sum())}", flush=True)

    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edges)
        loss = focal_loss(logits[train_gpu], y[train_gpu], alpha_pos)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            probs = torch.sigmoid(model(x, edges)).cpu().numpy()
        auc = roc_auc_score(labels_np[val_np], probs[val_np])
        if auc > best_auc:
            best_auc, best_epoch = float(auc), epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} loss={float(loss):.5f} val_auc={auc:.4f} "
                  f"best={best_auc:.4f} patience={patience}/40", flush=True)
        if patience >= 40:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        logits = model(x, edges).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    threshold = threshold_on_val(probs[val_np], labels_np[val_np])
    val_metrics = metrics(probs[val_np], labels_np[val_np], threshold)
    holdout_np = holdout.numpy()
    holdout_metrics = metrics(probs[holdout_np], labels_np[holdout_np], threshold)
    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output / "relnet_reimplementation_best.pt"
    torch.save({
        "model_state": best_state, "input_channels": int(x.shape[1]),
        "hidden_channels": 128, "num_layers": 2, "dropout": 0.3,
        "relation_names": ("follows", "rev_follows"),
        "seed": args.seed, "best_epoch": best_epoch, "best_val_auc": best_auc,
        "frozen_threshold": threshold, "graph": str(args.graph),
        "architecture_note": "new gated relation-wise mean convolution; not the missing historical RelNet",
    }, ckpt_path)
    report = {
        "status": "new RelNet reimplementation, not historical checkpoint recovery",
        "graph": str(args.graph), "seed": args.seed,
        "split_sizes": {"train": int(train.sum()), "val": int(val.sum()),
                        "holdout": int(holdout.sum())},
        "alpha_pos": alpha_pos, "best_epoch": best_epoch,
        "best_val_auc": best_auc, "frozen_threshold": threshold,
        "val_metrics": val_metrics, "holdout_metrics": holdout_metrics,
        "checkpoint": str(ckpt_path),
    }
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output / "train_graph_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for i, uid in enumerate(ids):
            split = "holdout" if holdout[i] else ("val" if val[i] else "train")
            stream.write(json.dumps({"account_id": uid, "split": split,
                                     "label": int(labels[i]), "logit": float(logits[i]),
                                     "bot_probability": float(probs[i])}) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
