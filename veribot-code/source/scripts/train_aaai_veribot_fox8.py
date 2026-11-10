"""Train the AAAI VeriBot HGT/RelNet branches on one fixed Fox8 split.

The two branches receive identical 31 named source features and the same
native reply/retweet graph. This keeps the manuscript's HGT/RelNet/fusion path;
it is a new executable run, not a surrogate for the submitted aggregate table.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/hgt_import/code"))
from hgt_model import VerifiableHGT
from relnet_reimplementation import GatedRelNet
from verifiable_bot.aaai_fox8_native_graph import native_relation_edges
from verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS


def sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    h = sha256()
    h.update(str(tuple(tensor.shape)).encode())
    h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()


def load_inputs() -> tuple[torch.Tensor, torch.Tensor, dict, list[str], dict, dict]:
    feature_file = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
    label_file = feature_file.with_name("labels.csv")
    split_file = ROOT / "data/splits/fox8_23_seed0.json"
    frame = pd.read_csv(feature_file)
    if list(frame.columns) != ["user_id", *FEATURE_FIELDS]:
        raise ValueError("Fox8 model feature schema differs from the 31 named facts")
    ids = frame["user_id"].astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("model rows need unique account identities")
    labels = pd.read_csv(label_file)["label"].to_numpy(dtype=np.int64)
    if len(labels) != len(ids):
        raise ValueError("feature and label rows differ")
    split = json.loads(split_file.read_text(encoding="utf-8"))
    id_to_index = {uid: i for i, uid in enumerate(ids)}
    indices = {}
    for name in ("train", "dev", "test"):
        selected = list(dict.fromkeys(split["splits"][name]["account_ids"]))
        if any(uid not in id_to_index for uid in selected):
            raise ValueError("split account absent from model inputs")
        indices[name] = np.asarray([id_to_index[uid] for uid in selected], dtype=np.int64)
    if len(set().union(*(set(index.tolist()) for index in indices.values()))) != len(ids):
        raise ValueError("fixed split does not cover all unique feature accounts")
    if any(set(indices[a]) & set(indices[b]) for a, b in
           (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("split accounts overlap")
    raw = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(raw[indices["train"]])
    x = torch.from_numpy(np.clip(scaler.transform(raw), -3, 3).astype(np.float32))
    raw_source = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    event_index = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
    edges, native_meta = native_relation_edges(
        raw_path=raw_source, event_index_path=event_index, account_ids=ids)
    metadata = (["user"], [("user", name, "user") for name in edges])
    graph = {("user", name, "user"): value for name, value in edges.items()}
    contract = {
        "feature_file_sha256": sha256_file(feature_file),
        "label_file_sha256": sha256_file(label_file),
        "split_file_sha256": sha256_file(split_file),
        "feature_sha256": tensor_hash(x),
        "graph_sha256": {name: tensor_hash(value) for name, value in edges.items()},
        "account_count": len(ids),
        "split_unique_sizes": {name: len(value) for name, value in indices.items()},
        "split_source_rows": {name: len(split["splits"][name]["account_ids"])
                              for name in indices},
        "feature_names": list(FEATURE_FIELDS),
        "graph_rule": native_meta["operation"],
        "native_graph": native_meta,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return x, torch.as_tensor(labels, dtype=torch.float32), indices, ids, graph, {
        "metadata": metadata, "relnet_edges": edges, "contract": contract}


def focal_loss(logits, targets, alpha_pos: float):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    alpha = alpha_pos * targets + (1 - alpha_pos) * (1 - targets)
    return (alpha * (1 - pt).square() * ce).mean()


def branch_logits(model, branch, x, graph, edges):
    if branch == "hgt":
        return model({"user": x}, graph)["bot_logit"]
    return model(x, edges)


def fit_branch(branch, x, y, split, graph, edges, metadata, seed,
               max_epochs, patience):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = (VerifiableHGT(metadata, {"user": x.shape[1]}, hidden_channels=128,
                           out_channels=1, num_layers=2, heads=4, dropout=0.3)
             if branch == "hgt" else
             GatedRelNet(x.shape[1], hidden_channels=128,
                         relation_names=tuple(edges),
                         num_layers=2, dropout=0.3))
    train = torch.as_tensor(split["train"])
    dev = torch.as_tensor(split["dev"])
    alpha_pos = float((y[train] == 0).float().mean())
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-5)
    best_auc = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = branch_logits(model, branch, x, graph, edges)
        loss = focal_loss(logits[train], y[train], alpha_pos)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            probability = torch.sigmoid(branch_logits(
                model, branch, x, graph, edges)).numpy()
        auc = float(roc_auc_score(y[dev].numpy(), probability[split["dev"]]))
        if auc > best_auc + 1e-8:
            best_auc = auc
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"branch": branch, "epoch": epoch,
                              "train_loss": float(loss.detach()),
                              "dev_auc": auc, "best_epoch": best_epoch}), flush=True)
        if stale >= patience:
            break
    if best_state is None:
        raise ValueError(f"no checkpoint for {branch}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        final = branch_logits(model, branch, x, graph, edges).numpy()
    return model, best_state, final, {"best_epoch": best_epoch,
                                     "best_dev_auc": best_auc,
                                     "alpha_pos": alpha_pos}


def metrics(logits, y, indices):
    probability = 1 / (1 + np.exp(-np.asarray(logits)))
    labels = y.numpy()[indices].astype(int)
    predicted = probability[indices] >= 0.5
    return {"n": len(indices),
            "accuracy": float(accuracy_score(labels, predicted)),
            "macro_f1": float(f1_score(labels, predicted, average="macro")),
            "auc": float(roc_auc_score(labels, probability[indices]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/aaai_veribot_fox8/seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()
    torch.set_num_threads(4)
    x, y, split, ids, graph, pack = load_inputs()
    model_states = {}
    outputs = {}
    selection = {}
    for branch in ("hgt", "relnet"):
        _, state, logits, fit = fit_branch(
            branch, x, y, split, graph, pack["relnet_edges"],
            pack["metadata"], args.seed, args.epochs, args.patience)
        model_states[branch] = state
        outputs[branch] = logits
        selection[branch] = fit
    fused = 0.5 * (outputs["hgt"] + outputs["relnet"])
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "model.pt"
    torch.save({"model_state": model_states, "selection": selection,
                "metadata": pack["metadata"],
                "input_contract": pack["contract"],
                "seed": args.seed}, checkpoint_path)
    report = {"algorithm": "AAAI_VeriBot_HGT_RelNet_mean_logit",
              "seed": args.seed,
              "checkpoint_sha256": sha256_file(checkpoint_path),
              "input_contract": {k: value for k, value in pack["contract"].items()
                                 if not k.startswith("scaler_")},
              "selection": selection,
              "dev": metrics(fused, y, split["dev"]),
              "test": metrics(fused, y, split["test"]),
              "predictions": [
                  {"account_id": ids[int(i)], "label": int(y[i]),
                   "hgt_logit": float(outputs["hgt"][i]),
                   "relnet_logit": float(outputs["relnet"][i]),
                   "fused_logit": float(fused[i])}
                  for i in split["test"]],
              "note": "new fixed-split run of the submitted architecture; 2279 unique Fox8 accounts"}
    (args.output / "test_predictions.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "predictions"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
