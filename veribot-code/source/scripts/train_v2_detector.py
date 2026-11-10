"""Train the VeriBot v2 detector under a fixed HypBot-compatible split.

Examples (paths can point to a shared data directory; no data are copied):
  python source/scripts/train_v2_detector.py --dataset fox8 \
    --data-dir D:/Code/SAHG/SAHG/code/data/fox8_23 \
    --raw-source data/raw/fox8_23/fox8_23_dataset.ndjson.gz --output output/v2/fox8

  python source/scripts/train_v2_detector.py --dataset mgtab \
    --data-dir D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB \
    --raw-source D:/Code/SAHG/SAHG/code/data/raw/MGTAB_download.zip \
    --output output/v2/mgtab
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.v2_detector import EvidenceGatedDetector, focal_loss, knn_edges, relation_means


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_bytes(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(len(chunk).to_bytes(8, "little"))
        h.update(chunk)
    return h.hexdigest()


def source_hash(paths: list[Path]) -> str:
    """Hash ordered source files without loading large releases into RAM."""
    h = hashlib.sha256()
    for path in paths:
        length = path.stat().st_size
        h.update(length.to_bytes(8, "little"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def stratified_tabular_split(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """The classwise 70/15/15 indices in the current HypBot release."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for label in (1, 0):
        idx = np.where(labels == label)[0].copy()
        rng.shuffle(idx)
        n_train, n_val = int(0.70 * len(idx)), int(0.15 * len(idx))
        parts["train"].append(idx[:n_train])
        parts["val"].append(idx[n_train:n_train + n_val])
        parts["test"].append(idx[n_train + n_val:])
    return {name: np.concatenate(groups) for name, groups in parts.items()}


def load_data(args, seed: int):
    data_dir = args.data_dir
    if args.dataset == "mgtab":
        load = lambda filename: torch.load(data_dir / filename, map_location="cpu", weights_only=True)
        raw_x = load("features.pt").float().numpy()
        labels = load("labels_bot.pt").long().numpy()
        edge_index = load("edge_index.pt").long()
        edge_type = load("edge_type.pt").long()
        train, rest = train_test_split(np.arange(len(labels)), test_size=0.30,
                                       stratify=labels, random_state=seed)
        val, test = train_test_split(rest, test_size=1 / 3, stratify=labels[rest],
                                     random_state=seed)
        split = {"train": train, "val": val, "test": test}
        account_ids = [str(i) for i in range(len(labels))]
    else:
        frame = pd.read_csv(data_dir / "features.csv")
        labels = pd.read_csv(data_dir / "labels.csv")["label"].to_numpy(dtype=np.int64)
        if len(frame) != len(labels):
            raise ValueError("features and labels contain different account counts")
        if args.dataset == "fox8":
            if "user_id" not in frame:
                raise ValueError("Fox8 requires user_id in features.csv")
            account_ids = frame["user_id"].astype(str).tolist()
            if len(set(account_ids)) != len(account_ids):
                raise ValueError("Fox8 duplicate account ID: preprocess before splitting")
        else:
            if args.account_file is None:
                raise ValueError("BotSim needs --account-file Users.csv for source-aligned IDs")
            account_ids = pd.read_csv(args.account_file)["user_id"].astype(str).tolist()
            if len(account_ids) != len(frame) or len(set(account_ids)) != len(account_ids):
                raise ValueError("BotSim account IDs and feature rows do not align")
        raw_x = frame.drop(columns=["user_id"], errors="ignore").fillna(0.0).to_numpy(dtype=np.float32)
        split = stratified_tabular_split(labels, seed)
        edge_index = edge_type = None
    scaler = StandardScaler().fit(raw_x[split["train"]])
    x = torch.from_numpy(np.clip(scaler.transform(raw_x), -3, 3).astype(np.float32))
    if args.dataset != "mgtab":
        edge_index, edge_type = knn_edges(x, args.knn_k)
    if len(set(account_ids)) != len(account_ids):
        raise ValueError("each evaluation unit must have a unique account ID")
    all_ids = [set(map(int, split[name])) for name in ("train", "val", "test")]
    if any(all_ids[i] & all_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("train/validation/test splits overlap")
    if sum(map(len, all_ids)) != len(labels):
        raise ValueError("split does not cover all accounts")
    hashes = {
        "raw_sha256": source_hash(args.raw_source),
        "split_sha256": hash_bytes(json.dumps({name: [account_ids[i] for i in split[name]]
                                                for name in ("train", "val", "test")},
                                               sort_keys=True).encode()),
        "feature_sha256": hash_bytes(x.numpy().tobytes()),
        "graph_sha256": hash_bytes(edge_index.numpy().tobytes(), edge_type.numpy().tobytes()),
    }
    return x, labels, split, account_ids, edge_index, edge_type, hashes, scaler


def select_threshold(prob: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.5, prob])
    values = np.array([f1_score(labels, prob >= threshold, average="macro", zero_division=0)
                       for threshold in candidates])
    tied = np.flatnonzero(np.isclose(values, values.max()))
    return float(candidates[tied[np.argmin(np.abs(candidates[tied] - 0.5))]])


def train_one(args, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, y, split, ids, edge, relation, hashes, scaler = load_data(args, seed)
    device = torch.device(args.device)
    x = x.to(device)
    edge = edge.to(device)
    relation = relation.to(device)
    graph_dim = min(args.graph_dim, x.shape[1])
    if graph_dim < x.shape[1]:
        generator = torch.Generator(device="cpu").manual_seed(2027)
        projection = torch.randn(x.shape[1], graph_dim, generator=generator) / x.shape[1] ** 0.5
        graph_x = x @ projection.to(device)
    else:
        projection = None
        graph_x = x
    means, degree = relation_means(graph_x, edge, relation, int(relation.max()) + 1)
    model = EvidenceGatedDetector(x.shape[1], graph_dim, means.shape[1],
                                  args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    labels = torch.from_numpy(y).to(device)
    train = torch.as_tensor(split["train"], device=device)
    val = torch.as_tensor(split["val"], device=device)
    alpha = float((labels[train] == 0).float().mean())
    best_auc, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = train[torch.randperm(len(train), device=device)]
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start:start + args.batch_size]
            logit = model(x[idx], means[idx], degree[idx])
            loss = focal_loss(logit, labels[idx], alpha)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prob_val = torch.sigmoid(model(x[val], means[val], degree[val])).cpu().numpy()
        score = roc_auc_score(y[split["val"]], prob_val)
        if score > best_auc:
            best_auc, best_epoch, stale = float(score), epoch, 0
            best_state = {key: tensor.detach().cpu().clone()
                          for key, tensor in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        val_prob = torch.sigmoid(model(x[val], means[val], degree[val])).cpu().numpy()
    threshold = select_threshold(val_prob, y[split["val"]])
    with torch.inference_mode():
        test = torch.as_tensor(split["test"], device=device)
        test_prob = torch.sigmoid(model(x[test], means[test], degree[test])).cpu().numpy()
    checkpoint_dir = args.output / f"seed{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pt"
    checkpoint = {
        "state_dict": best_state, "projection": projection,
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
        "feature_dim": x.shape[1], "graph_dim": graph_dim,
        "num_relations": means.shape[1], "hidden": args.hidden, "dropout": args.dropout,
        "seed": seed, "best_epoch": best_epoch, "best_val_auc": best_auc,
        "threshold": threshold, "hashes": hashes,
    }
    torch.save(checkpoint, checkpoint_path)
    manifest = {"dataset": args.dataset, "seed": seed, **hashes,
                "threshold": threshold, "model_sha256": file_sha256(checkpoint_path),
                "best_epoch": best_epoch, "best_val_auc": best_auc}
    rows = [{"account_id": ids[int(i)], "label": int(y[i]),
             "probability": float(p), "split": "test"}
            for i, p in zip(split["test"], test_prob)]
    result = {"manifest": manifest, "predictions": rows}
    (checkpoint_dir / "test_predictions.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    val_rows = [{"account_id": ids[int(i)], "label": int(y[i]),
                 "probability": float(p), "split": "val"}
                for i, p in zip(split["val"], val_prob)]
    (checkpoint_dir / "val_predictions.json").write_text(
        json.dumps({"manifest": manifest, "predictions": val_rows}, indent=2), encoding="utf-8")
    acc = float(np.mean((test_prob >= threshold) == y[split["test"]]))
    return {"dataset": args.dataset, "seed": seed, "test_acc": acc,
            "checkpoint": str(checkpoint_path), "predictions": str(checkpoint_dir / "test_predictions.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mgtab", "fox8", "botsim"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--raw-source", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--graph-dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    reports = [train_one(args, seed) for seed in args.seeds]
    print(json.dumps({"runs": reports, "mean_test_acc": float(np.mean([r["test_acc"] for r in reports]))},
                     indent=2))


if __name__ == "__main__":
    main()
