"""Train fresh HGT and gated relation-mean branches on one fixed WWW dataset graph.

Train labels drive loss; dev labels select checkpoint and thresholds. Test labels
are evaluated after both branches and their thresholds have been frozen.
Historical Fox8/BotSim checkpoints and SAHG features are not reproduced here.
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

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "source/hgt_import/code"))
from hgt_model import VerifiableHGT  # noqa: E402
from relnet_reimplementation import GatedRelNet  # noqa: E402


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def focal_loss(logits, targets, alpha_pos):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    alpha = alpha_pos * targets + (1 - alpha_pos) * (1 - targets)
    return (alpha * (1 - pt).square() * ce).mean()


def choose_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    best_acc, best_threshold = -1.0, 0.5
    for threshold in np.linspace(0.05, 0.95, 100):
        score = accuracy_score(labels, probabilities >= threshold)
        if score > best_acc:
            best_acc, best_threshold = score, float(threshold)
    return best_threshold


def metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predictions = probabilities >= threshold
    return {
        "n": len(labels), "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "threshold": threshold,
    }


def train_branch(name, model, forward, labels, masks, alpha_pos, output, seed, config):
    train_mask, dev_mask = masks["train"], masks["dev"]
    train_gpu = train_mask.cuda()
    y = labels.float().cuda()
    y_np = labels.numpy()
    dev_np = dev_mask.numpy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)
    best_auc, best_epoch, best_state, patience = -1.0, 0, None, 0
    print(f"[{name}] training seed={seed} train={int(train_mask.sum())} "
          f"dev={int(dev_mask.sum())}", flush=True)
    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        logits = forward(model)
        loss = focal_loss(logits[train_gpu], y[train_gpu], alpha_pos)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            probs = torch.sigmoid(forward(model)).cpu().numpy()
        dev_auc = roc_auc_score(y_np[dev_np], probs[dev_np])
        if dev_auc > best_auc:
            best_auc, best_epoch = float(dev_auc), epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"[{name}] epoch={epoch} loss={float(loss):.5f} "
                  f"dev_auc={dev_auc:.4f} best={best_auc:.4f} patience={patience}/40", flush=True)
        if patience >= 40:
            print(f"[{name}] early stopping at epoch {epoch}", flush=True)
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        frozen_logits = forward(model).cpu()
    frozen_probabilities = torch.sigmoid(frozen_logits).numpy()
    threshold = choose_threshold(frozen_probabilities[dev_np], y_np[dev_np])
    branch_dir = output / name
    branch_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best_state, "seed": seed, "best_epoch": best_epoch,
        "best_dev_auc": best_auc, "frozen_threshold": threshold, **config,
    }
    torch.save(checkpoint, branch_dir / "best.pt")
    return {
        "name": name, "probabilities": frozen_probabilities,
        "logits": frozen_logits.numpy(), "threshold": threshold,
        "best_epoch": best_epoch, "best_dev_auc": best_auc,
        "dev_metrics": metrics(frozen_probabilities[dev_np], y_np[dev_np], threshold),
        "checkpoint": str(branch_dir / "best.pt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("train on the remote CUDA server")
    args.output.mkdir(parents=True, exist_ok=True)
    graph = torch.load(args.graph, map_location="cpu", weights_only=False)
    x_dict = {name: tensor.cuda() for name, tensor in graph.x_dict.items()}
    edge_dict = {name: tensor.cuda() for name, tensor in graph.edge_index_dict.items()}
    masks = {name: graph["user"][f"{name}_mask"].bool() for name in ("train", "dev", "test")}
    assert sum(int(mask.sum()) for mask in masks.values()) == graph["user"].num_nodes
    assert not (masks["train"] & masks["dev"]).any()
    assert not (masks["train"] & masks["test"]).any()
    assert not (masks["dev"] & masks["test"]).any()
    labels = graph["user"].y.long()
    alpha_pos = float((labels[masks["train"]] == 0).float().mean())
    seed = args.seed
    print(f"Graph: {args.graph} metadata={graph.metadata()} alpha_pos={alpha_pos:.4f}", flush=True)

    set_seed(seed)
    hgt = VerifiableHGT(
        metadata=graph.metadata(),
        in_channels={name: tensor.shape[1] for name, tensor in x_dict.items()},
        hidden_channels=128, out_channels=1, num_layers=2, heads=4, dropout=0.3,
    ).cuda()
    hgt_result = train_branch(
        "hgt", hgt, lambda model: model(x_dict, edge_dict)["bot_logit"],
        labels, masks, alpha_pos, args.output, seed,
        {"model_kind": "new_hgt", "metadata": graph.metadata(),
         "in_channels": {name: tensor.shape[1] for name, tensor in x_dict.items()},
         "hidden_channels": 128, "num_layers": 2, "heads": 4, "dropout": 0.3},
    )
    del hgt
    torch.cuda.empty_cache()

    relation_keys = [key for key in edge_dict if key[0] == key[2] == "user"]
    if not relation_keys:
        raise RuntimeError("new RelNet requires at least one user-user relation")
    relation_names = tuple(key[1] for key in relation_keys)
    user_edges = {key[1]: edge_dict[key] for key in relation_keys}
    set_seed(seed)
    relnet = GatedRelNet(input_channels=x_dict["user"].shape[1],
                         relation_names=relation_names).cuda()
    relnet_result = train_branch(
        "relnet_reimplementation", relnet,
        lambda model: model(x_dict["user"], user_edges),
        labels, masks, alpha_pos, args.output, seed,
        {"model_kind": "new gated relation-wise mean model, not historical RelNet",
         "input_channels": x_dict["user"].shape[1], "hidden_channels": 128,
         "relation_names": relation_names, "num_layers": 2, "dropout": 0.3},
    )
    del relnet
    torch.cuda.empty_cache()

    # Both branch checkpoints and validation thresholds are frozen above.
    y_np = labels.numpy()
    test_np = masks["test"].numpy()
    ids = graph["user"].account_id
    report = {
        "status": "new Qwen feature pipeline and fresh branch models; not historical manuscript replication",
        "graph": str(args.graph), "graph_sha256": hashlib.sha256(args.graph.read_bytes()).hexdigest(),
        "seed": seed, "split_sizes": {name: int(mask.sum()) for name, mask in masks.items()},
        "alpha_pos": alpha_pos,
        "branches": {},
    }
    for result in (hgt_result, relnet_result):
        result["test_metrics"] = metrics(
            result["probabilities"][test_np], y_np[test_np], result["threshold"])
        report["branches"][result["name"]] = {
            key: result[key] for key in ("checkpoint", "best_epoch", "best_dev_auc",
                                    "threshold", "dev_metrics", "test_metrics")
        }
        branch_dir = args.output / result["name"]
        with (branch_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for index, uid in enumerate(ids):
                split = next(name for name, mask in masks.items() if bool(mask[index]))
                stream.write(json.dumps({
                    "account_id": uid, "split": split, "label": int(labels[index]),
                    "logit": float(result["logits"][index]),
                    "bot_probability": float(result["probabilities"][index]),
                    "prediction": int(result["probabilities"][index] >= result["threshold"]),
                }) + "\n")
    (args.output / "training_and_test_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
