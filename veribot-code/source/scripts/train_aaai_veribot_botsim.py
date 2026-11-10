"""Train the VeriBot HGT and gated RelNet branches on BotSim's native graph.

HGT reads the user-subreddit bipartite graph. RelNet reads the exact binary
co-subreddit user projection through a mean-normalized matrix; its layer
weights and equations match the edge-wise gated relation convolution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/scripts"))
sys.path.insert(0, str(ROOT / "source/hgt_import/code"))

from hgt_model import VerifiableHGT
from train_aaai_veribot_fox8 import focal_loss, sha256_file, tensor_hash
from verifiable_bot.aaai_botsim_native_graph import build_native_graph, read_memberships
from verifiable_bot.aaai_dense_relnet import DenseGatedRelNet

USERS = ROOT / "data/raw/botsim_24/Users.csv"
ACTIVITY = ROOT / "data/raw/botsim_24/user_post_comment.json"
FEATURES = ROOT / "data/processed/aaai_veribot/botsim_24/features.csv"
LABELS = FEATURES.with_name("labels.csv")
SPLIT = ROOT / "data/splits/botsim_24_seed0.json"
OUTPUT = ROOT / "output/aaai_veribot_botsim/seed0"


def load_inputs():
    ids, subreddits, _ = read_memberships(USERS)
    frame = pd.read_csv(FEATURES)
    names = frame.columns.tolist()
    if len(frame) != len(ids) or len(names) != 17 or not all(names):
        raise ValueError("BotSim named user features differ from 17-column schema")
    labels = pd.read_csv(LABELS)["label"].to_numpy(dtype=np.int64)
    if len(labels) != len(ids) or set(labels.tolist()) != {0, 1}:
        raise ValueError("BotSim labels or user inventory differ")
    saved_split = json.loads(SPLIT.read_text(encoding="utf-8"))["splits"]
    split = {}
    for key in ("train", "dev", "test"):
        indices = np.asarray(saved_split[key]["indices"], dtype=np.int64)
        if [ids[int(i)] for i in indices] != saved_split[key]["account_ids"]:
            raise ValueError(f"BotSim {key} indices and account IDs differ")
        split[key] = indices
    if (len(set().union(*(set(values.tolist()) for values in split.values()))) != len(ids)
            or any(set(split[a]) & set(split[b]) for a, b in
                   (("train", "dev"), ("train", "test"), ("dev", "test")))):
        raise ValueError("fixed BotSim split is incomplete or overlapping")
    raw = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("BotSim model features contain non-finite values")
    scaler = StandardScaler().fit(raw[split["train"]])
    x = torch.from_numpy(np.clip(scaler.transform(raw), -3, 3).astype(np.float32))
    graph, relnet, graph_meta = build_native_graph(USERS, ids)
    metadata = (["user", "subreddit"], list(graph))
    x_dict = {"user": x, "subreddit": torch.ones((len(subreddits), 1))}
    contract = {
        "raw_users_sha256": sha256_file(USERS),
        "raw_activity_sha256": sha256_file(ACTIVITY),
        "feature_file_sha256": sha256_file(FEATURES),
        "label_file_sha256": sha256_file(LABELS),
        "split_file_sha256": sha256_file(SPLIT),
        "feature_names": names,
        "user_input_sha256": tensor_hash(x),
        "subreddit_input_sha256": tensor_hash(x_dict["subreddit"]),
        "hgt_graph_sha256": {str(key): tensor_hash(edge)
                             for key, edge in graph.items()},
        "relnet_operator_sha256": {name: tensor_hash(value)
                                    for name, value in relnet.items()},
        "graph_meta": graph_meta,
        "split_sizes": {key: len(value) for key, value in split.items()},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return x_dict, torch.as_tensor(labels, dtype=torch.float32), split, ids, graph, relnet, metadata, contract


def branch_logits(model, branch, x_dict, graph, relnet):
    if branch == "hgt":
        return model(x_dict, graph)["bot_logit"]
    return model(x_dict["user"], relnet)


def fit_branch(branch, x_dict, y, split, graph, relnet, metadata,
               seed, max_epochs, patience):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = (VerifiableHGT(metadata, {name: values.shape[1]
                                     for name, values in x_dict.items()},
                           hidden_channels=128, out_channels=1,
                           num_layers=2, heads=4, dropout=0.3)
             if branch == "hgt" else
             DenseGatedRelNet(x_dict["user"].shape[1], hidden_channels=128,
                              relation_names=tuple(relnet),
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
        output = branch_logits(model, branch, x_dict, graph, relnet)
        loss = focal_loss(output[train], y[train], alpha_pos)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            probability = torch.sigmoid(branch_logits(
                model, branch, x_dict, graph, relnet)).numpy()
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
        raise ValueError(f"no {branch} checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        final = branch_logits(model, branch, x_dict, graph, relnet).numpy()
    return best_state, final, {"best_epoch": best_epoch,
                               "best_dev_auc": best_auc,
                               "alpha_pos": alpha_pos}


def metrics(logits, labels, indices):
    probability = 1 / (1 + np.exp(-np.asarray(logits)))
    truth = labels.numpy()[indices].astype(int)
    guessed = probability[indices] >= 0.5
    return {"n": len(indices),
            "accuracy": float(accuracy_score(truth, guessed)),
            "macro_f1": float(f1_score(truth, guessed, average="macro")),
            "auc": float(roc_auc_score(truth, probability[indices]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()
    torch.set_num_threads(4)
    x_dict, y, split, ids, graph, relnet, metadata, contract = load_inputs()
    weights = {}
    logits = {}
    selection = {}
    for branch in ("hgt", "relnet"):
        weights[branch], logits[branch], selection[branch] = fit_branch(
            branch, x_dict, y, split, graph, relnet, metadata,
            args.seed, args.epochs, args.patience)
    fused = 0.5 * (logits["hgt"] + logits["relnet"])
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "model.pt"
    torch.save({"model_state": weights, "selection": selection,
                "metadata": metadata, "input_contract": contract,
                "seed": args.seed}, checkpoint)
    report = {
        "algorithm": "AAAI_VeriBot_HGT_RelNet_mean_logit",
        "graph": "native_user_subreddit_and_exact_shared_subreddit_projection",
        "seed": args.seed,
        "checkpoint_sha256": sha256_file(checkpoint),
        "input_contract": {key: value for key, value in contract.items()
                           if not key.startswith("scaler_")},
        "selection": selection,
        "dev": metrics(fused, y, split["dev"]),
        "test": metrics(fused, y, split["test"]),
        "predictions": [
            {"account_id": ids[int(i)], "label": int(y[i]),
             "hgt_logit": float(logits["hgt"][i]),
             "relnet_logit": float(logits["relnet"][i]),
             "fused_logit": float(fused[i])}
            for i in split["test"]],
    }
    (args.output / "test_predictions.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("predictions", "input_contract")}), flush=True)


if __name__ == "__main__":
    main()
