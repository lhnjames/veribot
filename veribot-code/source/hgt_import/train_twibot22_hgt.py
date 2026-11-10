"""Train the train-only TwiBot-22 HGT with a strict three-way split.

The project holdout never influences gradients, checkpoint selection, or threshold selection.
An independent salted validation slice is carved from the remaining official-train accounts for
early stopping and threshold selection.  The project holdout is evaluated exactly once after both
the checkpoint and threshold are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from hgt_model import VerifiableHGT  # noqa: E402

HGT_VAL_SALT = "twibot22_hgt_val_v1"


def hgt_val(uid: str) -> bool:
    key = f"{HGT_VAL_SALT}:{uid}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 20 == 0


def focal_loss(logits, targets, alpha_pos: float, gamma: float = 2.0):
    probabilities = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_t = alpha_pos * targets + (1 - alpha_pos) * (1 - targets)
    return (alpha_t * (1 - p_t).pow(gamma) * ce).mean()


def threshold_on(probs: np.ndarray, labels: np.ndarray) -> float:
    best = (-1.0, 0.5)
    for threshold in np.linspace(0.05, 0.95, 181):
        macro = f1_score(labels, probs >= threshold, average="macro")
        if macro > best[0]:
            best = (macro, float(threshold))
    return best[1]


def metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = (probs >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "bot_f1": float(f1_score(labels, pred, pos_label=1)),
        "human_f1": float(f1_score(labels, pred, pos_label=0)),
        "threshold": threshold,
    }


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("HGT training is permitted only on the remote CUDA server")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data = torch.load(args.graph, map_location="cpu", weights_only=False)
    labels = data["user"].y.float()
    account_ids = data["user"].account_id
    project_holdout = data["user"].holdout_mask.bool()
    val_flags = torch.tensor([hgt_val(uid) for uid in account_ids])
    hgt_val_mask = val_flags & ~project_holdout
    hgt_train_mask = ~val_flags & ~project_holdout
    assert not (hgt_train_mask & hgt_val_mask).any()
    assert not (hgt_train_mask & project_holdout).any()
    assert int(hgt_train_mask.sum() + hgt_val_mask.sum() + project_holdout.sum()) == len(labels)

    train_y = labels[hgt_train_mask]
    n_bot = int((train_y == 1).sum())
    n_human = int((train_y == 0).sum())
    alpha_pos = n_human / (n_bot + n_human)
    print(
        f"split hgt_train={int(hgt_train_mask.sum())} hgt_val={int(hgt_val_mask.sum())} "
        f"project_holdout={int(project_holdout.sum())}; train bot={n_bot} human={n_human}; "
        f"alpha_pos={alpha_pos:.6f}", flush=True,
    )

    device = torch.device("cuda")
    x_dict = {key: value.to(device=device, dtype=torch.float32) for key, value in data.x_dict.items()}
    edge_dict = {key: value.to(device) for key, value in data.edge_index_dict.items()}
    labels_gpu = labels.to(device)
    train_gpu = hgt_train_mask.to(device)
    model = VerifiableHGT(
        metadata=data.metadata(),
        in_channels={key: value.shape[1] for key, value in x_dict.items()},
        hidden_channels=args.hidden,
        out_channels=1,
        num_layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.02
    )

    labels_np = labels.numpy().astype(int)
    val_np = hgt_val_mask.numpy()
    best_auc = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(x_dict, edge_dict)["bot_logit"]
        loss = focal_loss(output[train_gpu], labels_gpu[train_gpu], alpha_pos)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(x_dict, edge_dict)["bot_logit"]).float().cpu().numpy()
        val_auc = float(roc_auc_score(labels_np[val_np], probabilities[val_np]))
        if val_auc > best_auc + 1e-6:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch} loss={loss.item():.6f} grad_norm={float(grad_norm):.4f} "
            f"val_auc={val_auc:.6f} best_auc={best_auc:.6f} stale={stale} "
            f"gpu_peak_mib={torch.cuda.max_memory_allocated()/2**20:.0f}", flush=True,
        )
        if stale >= args.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        final_probs = torch.sigmoid(model(x_dict, edge_dict)["bot_logit"]).float().cpu().numpy()
    frozen_threshold = threshold_on(final_probs[val_np], labels_np[val_np])
    val_metrics = metrics(final_probs[val_np], labels_np[val_np], frozen_threshold)

    # First and only holdout metric access is below, after model and threshold are frozen.
    holdout_np = project_holdout.numpy()
    holdout_metrics = metrics(final_probs[holdout_np], labels_np[holdout_np], frozen_threshold)
    metadata = (list(data.metadata()[0]), [tuple(v) for v in data.metadata()[1]])
    checkpoint = {
        "model_state": best_state,
        "best_hgt_val_auc": best_auc,
        "best_epoch": best_epoch,
        "frozen_threshold": frozen_threshold,
        "in_channels": {key: value.shape[1] for key, value in x_dict.items()},
        "metadata": metadata,
        "config": {
            "hidden_channels": args.hidden, "heads": args.heads, "num_layers": args.layers,
            "dropout": args.dropout, "lr": args.lr, "epochs": args.epochs,
            "patience": args.patience, "seed": args.seed,
        },
        "alpha_pos": alpha_pos,
        "split_sizes": {
            "hgt_train": int(hgt_train_mask.sum()), "hgt_val": int(hgt_val_mask.sum()),
            "project_holdout": int(project_holdout.sum()),
        },
    }
    torch.save(checkpoint, args.output / "clean_hgt_best.pt")
    np.save(args.output / "frozen_probabilities.float32.npy", final_probs.astype(np.float32))
    report = {
        "graph": str(args.graph),
        "split_rule": {
            "project_holdout": "sha256(account_id) prefix modulo twenty",
            "hgt_val": f"sha256('{HGT_VAL_SALT}:'+account_id) prefix modulo twenty among non-holdout",
        },
        "best_epoch": best_epoch,
        "best_hgt_val_auc": best_auc,
        "frozen_threshold": frozen_threshold,
        "hgt_val_metrics": val_metrics,
        "project_holdout_metrics": holdout_metrics,
        "class_balance": {"train_bot": n_bot, "train_human": n_human, "alpha_pos": alpha_pos},
    }
    atomic_json(args.output / "clean_hgt_training_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
