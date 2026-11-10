"""Train a native-follow HGT using only official TwiBot-20 train labels."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/hgt_import/code"))
from hgt_model import VerifiableHGT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_text_features(path: Path, ids: np.ndarray,
                       graph_bundle_sha256: str) -> tuple[np.ndarray, list[str]]:
    """Load source-aligned fixed text features without labels."""
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as text_bundle:
            if not np.array_equal(text_bundle["ids"].astype(str), ids):
                raise ValueError("text rows differ from graph account order")
            values = text_bundle["values"].astype(np.float64)
        width = values.shape[1] // 2
        names = ([f"profile_hash_{i}" for i in range(width)] +
                 [f"tweet_hash_{i}" for i in range(values.shape[1] - width)])
    elif path.suffix == ".npy":
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (manifest["native_graph_bundle_sha256"] != graph_bundle_sha256
                or manifest["nodes"] != len(ids)
                or manifest["feature_family"] !=
                "frozen_Qwen3_0.6B_description_plus_five_tweets"):
            raise ValueError("frozen text manifest differs from native graph")
        values = np.load(path, allow_pickle=False, mmap_mode="r").astype(np.float64)
        width = values.shape[1] // 2
        if width != manifest["description_width"] or width != manifest["tweet_width"]:
            raise ValueError("Qwen text width differs from manifest")
        names = ([f"description_qwen_{i}" for i in range(width)] +
                 [f"tweet_qwen_{i}" for i in range(width)])
    else:
        raise ValueError("fixed text features must be .npz or .npy")
    if values.shape[0] != len(ids) or not np.all(np.isfinite(values)):
        raise ValueError("text feature shape or values differ")
    return values, names


def threshold_on_dev(logits: np.ndarray, labels: np.ndarray) -> float:
    values = sorted(set(logits.tolist()))
    candidates = [values[0] - 1, *[(a + b) / 2 for a, b in
                                    zip(values[:-1], values[1:])], values[-1] + 1]
    scores = [f1_score(labels, logits >= threshold, average="macro")
              for threshold in candidates]
    best = max(scores)
    ties = [threshold for threshold, score in zip(candidates, scores)
            if abs(score - best) <= 1e-12]
    return float(min(ties, key=abs))


def metric(logits: np.ndarray, labels: np.ndarray, indices: np.ndarray,
           threshold: float) -> dict:
    truth = labels[indices]
    scores = logits[indices]
    prediction = scores >= threshold
    return {"n": len(indices),
            "accuracy": float(accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, average="macro")),
            "bot_f1": float(f1_score(truth, prediction, pos_label=1)),
            "auc": float(roc_auc_score(truth, scores))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--text-features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    with np.load(args.bundle, allow_pickle=False) as bundle:
        raw = bundle["values"].astype(np.float64)
        labels = bundle["labels"].astype(np.int64)
        split = bundle["split"].astype(str)
        ids = bundle["ids"].astype(str)
        edges = torch.from_numpy(bundle["edge_index"].astype(np.int64))
        names = bundle["names"].astype(str).tolist()
    bundle_sha = sha256(args.bundle)
    text_sha = None
    if args.text_features is not None:
        text_sha = sha256(args.text_features)
        text_values, text_names = load_text_features(
            args.text_features, ids, bundle_sha)
        raw = np.column_stack((raw, text_values))
        names.extend(text_names)
    train = np.flatnonzero(split == "train")
    dev = np.flatnonzero(split == "dev")
    test = np.flatnonzero(split == "test")
    if ((len(train), len(dev), len(test)) != (8278, 2365, 1183)
            or np.any(labels[split == "support"] != -1)):
        raise ValueError("official split or support-label contract differs")
    medians = np.nanmedian(raw[train], axis=0)
    if not np.all(np.isfinite(medians)):
        raise ValueError("train-only median is non-finite")
    filled = np.where(np.isfinite(raw), raw, medians)
    scaler = StandardScaler().fit(filled[train])
    normalized = np.clip(scaler.transform(filled), -3, 3).astype(np.float32)
    x = torch.from_numpy(normalized).to(device)
    graph = {
        ("user", "follows", "user"): edges.to(device),
        ("user", "rev_follows", "user"): edges.flip(0).to(device),
    }
    metadata = (["user"], list(graph))
    model = VerifiableHGT(metadata, {"user": len(names)},
                          hidden_channels=args.hidden, out_channels=1,
                          num_layers=2, heads=4, dropout=0.3).to(device)
    initial_equivalence = None
    initial_sha = None
    if args.initialize_from is not None:
        initial_sha = sha256(args.initialize_from)
        previous = torch.load(args.initialize_from, map_location="cpu",
                              weights_only=False)
        if (previous["bundle_sha256"] != bundle_sha
                or previous.get("text_features_sha256") is not None
                or previous["feature_names"] != names[:22]
                or previous["hidden"] != args.hidden):
            raise ValueError("numeric initializer differs from native graph contract")
        state = model.state_dict()
        old = previous["model_state"]
        for name in state:
            if name == "lin_dict.user.weight":
                state[name] = torch.zeros_like(state[name])
                state[name][:, :22] = old[name]
            else:
                if name not in old or state[name].shape != old[name].shape:
                    raise ValueError(f"numeric initializer differs at {name}")
                state[name] = old[name]
        model.load_state_dict(state)
        numeric_model = VerifiableHGT(metadata, {"user": 22},
                                      hidden_channels=args.hidden, out_channels=1,
                                      num_layers=2, heads=4, dropout=0.3).to(device).eval()
        numeric_model.load_state_dict(old)
        model.eval()
        with torch.inference_mode():
            expected = numeric_model({"user": x[:, :22]}, graph)["bot_logit"]
            observed = model({"user": x}, graph)["bot_logit"]
        initial_equivalence = float(torch.max(torch.abs(expected - observed)))
        if initial_equivalence > 1e-5:
            raise ValueError("text-extended initializer changes numeric prediction")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4,
                                  weight_decay=1e-4)
    train_index = torch.as_tensor(train, dtype=torch.long, device=device)
    dev_index = torch.as_tensor(dev, dtype=torch.long, device=device)
    labels_gpu = torch.as_tensor(labels, dtype=torch.float32, device=device)
    best_auc = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    args.output.mkdir(parents=True, exist_ok=True)
    if initial_sha is not None:
        model.eval()
        with torch.inference_mode():
            dev_logits = model({"user": x}, graph)["bot_logit"][dev_index]
        best_auc = float(roc_auc_score(labels[dev],
                                       dev_logits.detach().cpu().numpy()))
        best_state = deepcopy(model.state_dict())
        print(json.dumps({"epoch": 0, "dev_auc": best_auc,
                          "initializer_sha256": initial_sha,
                          "max_numeric_equivalence_difference": initial_equivalence}),
              flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model({"user": x}, graph)["bot_logit"]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[train_index], labels_gpu[train_index])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            dev_logits = model({"user": x}, graph)["bot_logit"][dev_index]
        dev_np = dev_logits.detach().cpu().numpy()
        auc = float(roc_auc_score(labels[dev], dev_np))
        if auc > best_auc + 1e-9:
            best_auc = auc
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        print(json.dumps({"epoch": epoch, "train_loss": float(loss.detach()),
                          "dev_auc": auc, "best_epoch": best_epoch}), flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise ValueError("no HGT checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        logits = model({"user": x}, graph)["bot_logit"].detach().cpu().numpy()
    threshold = threshold_on_dev(logits[dev], labels[dev])
    model_path = args.output / "model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "architecture": ("native_follow_HGT_22_raw_facts_plus_" +
                         ("frozen_qwen" if args.text_features.suffix == ".npy"
                          else "text_hash") if text_sha else
                         "native_follow_HGT_22_raw_facts"),
        "bundle_sha256": bundle_sha,
        "text_features_sha256": text_sha,
        "feature_names": names,
        "train_medians": medians.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "dev_logit_threshold": threshold,
        "best_epoch_dev_auc": best_epoch,
        "seed": args.seed,
        "hidden": args.hidden,
        "initialized_from_checkpoint_sha256": initial_sha,
        "epoch0_numeric_equivalence_max_abs": initial_equivalence,
    }, model_path)
    report = {
        "architecture": ("native_follow_HGT_22_raw_facts_plus_" +
                         ("frozen_qwen" if args.text_features.suffix == ".npy"
                          else "text_hash") if text_sha else
                         "native_follow_HGT_22_raw_facts"),
        "checkpoint_sha256": sha256(model_path),
        "bundle_sha256": bundle_sha,
        "text_features_sha256": text_sha,
        "seed": args.seed,
        "initialized_from_checkpoint_sha256": initial_sha,
        "epoch0_numeric_equivalence_max_abs": initial_equivalence,
        "best_epoch_dev_auc": best_epoch,
        "dev_logit_threshold": threshold,
        "source_nodes": len(ids),
        "native_directed_edges": edges.shape[1],
        "train": metric(logits, labels, train, threshold),
        "dev": metric(logits, labels, dev, threshold),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
