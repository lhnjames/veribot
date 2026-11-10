"""One fixed-protocol evaluation of a frozen train-only HGT on TwiBot-20 test.

Test accounts use the same frozen Qwen encoder and train-fitted numeric scaling.
Only observed train-test follow edges are attached to the saved train graph;
test-test and dev edges are excluded. No test label enters feature construction,
graph construction, calibration, or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from hgt_model import VerifiableHGT  # noqa: E402
from relnet_reimplementation import GatedRelNet  # noqa: E402
from build_features_and_graph import (  # noqa: E402
    EMBED_BATCH_SIZE, MAX_TEXT_CHARS, N_TWEETS_PER_ACCOUNT,
    as_number, embed_texts, sha256_file,
)

FOLLOWS = ("user", "follows", "user")
REVERSE = ("user", "rev_follows", "user")


def numeric(rows: list[dict]) -> torch.Tensor:
    followers = torch.tensor([as_number(r["profile"].get("followers_count")) for r in rows])
    following = torch.tensor([as_number(r["profile"].get("friends_count")) for r in rows])
    tweets = torch.tensor([as_number(r["profile"].get("statuses_count")) for r in rows])
    listed = torch.tensor([as_number(r["profile"].get("listed_count")) for r in rows])
    verified = torch.tensor([
        1.0 if str(r["profile"].get("verified")).strip() == "True" else 0.0
        for r in rows
    ])
    return torch.stack([followers.log1p(), following.log1p(), tweets.log1p(),
                        listed.log1p(), verified], dim=1)


def text_inputs(rows: list[dict]) -> tuple[list[str], list[str]]:
    descriptions = [(r["profile"].get("description") or "").strip()[:MAX_TEXT_CHARS]
                    for r in rows]
    tweet_texts = []
    for row in rows:
        tweets = row.get("tweet") or []
        original = [t for t in tweets if not t.strip().startswith("RT @")]
        chosen = (original[:N_TWEETS_PER_ACCOUNT] if len(original) >= N_TWEETS_PER_ACCOUNT
                  else (original + tweets)[:N_TWEETS_PER_ACCOUNT])
        tweet_texts.append(" ".join(chosen)[:MAX_TEXT_CHARS])
    return descriptions, tweet_texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--relnet-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("run the official evaluation on the remote CUDA server")

    train_rows = json.loads(args.train_json.read_text(encoding="utf-8"))
    test_rows = json.loads(args.test_json.read_text(encoding="utf-8"))
    train_graph = torch.load(args.graph, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_ids = train_graph["user"].account_id
    test_ids = [str(row["ID"]) for row in test_rows]
    if train_ids != [str(row["ID"]) for row in train_rows]:
        raise RuntimeError("saved train graph account order differs from train.json")
    if set(train_ids) & set(test_ids):
        raise RuntimeError("test IDs overlap training graph")
    if train_graph["user"].x.shape[1] != checkpoint["in_channels"]["user"]:
        raise RuntimeError("checkpoint and train graph feature dimensions differ")

    print(f"Frozen test evaluation: train={len(train_ids)} test={len(test_ids)}", flush=True)
    train_raw = numeric(train_rows)
    non_holdout = ~train_graph["user"].holdout_mask
    mean = train_raw[non_holdout].mean(dim=0, keepdim=True)
    std = train_raw[non_holdout].std(dim=0, keepdim=True).clamp(min=1e-6)
    test_numeric = (numeric(test_rows) - mean) / std

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder = AutoModel.from_pretrained(args.qwen_model, torch_dtype=torch.float32).cuda().eval()
    descriptions, tweets = text_inputs(test_rows)
    print("Encoding test descriptions...", flush=True)
    desc_embed = embed_texts(tokenizer, encoder, descriptions)
    print("Encoding test tweets...", flush=True)
    tweet_embed = embed_texts(tokenizer, encoder, tweets)
    del encoder
    torch.cuda.empty_cache()
    test_x = torch.cat([desc_embed, tweet_embed, test_numeric], dim=1)
    if test_x.shape[1] != checkpoint["in_channels"]["user"]:
        raise RuntimeError("new test features do not match frozen HGT")

    train_idx = {uid: i for i, uid in enumerate(train_ids)}
    test_idx = {uid: len(train_ids) + i for i, uid in enumerate(test_ids)}
    cross_edges: set[tuple[int, int]] = set()
    for row in train_rows:
        uid = str(row["ID"])
        neighbor = row.get("neighbor") or {}
        cross_edges.update((train_idx[uid], test_idx[str(dst)])
                           for dst in neighbor.get("following") or [] if str(dst) in test_idx)
        cross_edges.update((test_idx[str(src)], train_idx[uid])
                           for src in neighbor.get("follower") or [] if str(src) in test_idx)
    for row in test_rows:
        uid = str(row["ID"])
        neighbor = row.get("neighbor") or {}
        cross_edges.update((test_idx[uid], train_idx[str(dst)])
                           for dst in neighbor.get("following") or [] if str(dst) in train_idx)
        cross_edges.update((train_idx[str(src)], test_idx[uid])
                           for src in neighbor.get("follower") or [] if str(src) in train_idx)

    original_edges = train_graph[FOLLOWS].edge_index
    new_edges = torch.tensor(sorted(cross_edges), dtype=torch.long).t().contiguous() \
        if cross_edges else torch.empty((2, 0), dtype=torch.long)
    follows = torch.cat([original_edges, new_edges], dim=1).cuda()
    edge_dict = {FOLLOWS: follows, REVERSE: follows.flip(0)}
    x_dict = {"user": torch.cat([train_graph["user"].x, test_x], dim=0).cuda()}

    model = VerifiableHGT(
        metadata=checkpoint["metadata"], in_channels=checkpoint["in_channels"],
        hidden_channels=checkpoint["config"]["hidden_channels"], out_channels=1,
        num_layers=checkpoint["config"]["num_layers"], heads=checkpoint["config"]["heads"],
        dropout=checkpoint["config"]["dropout"],
    ).cuda().eval()
    model.load_state_dict(checkpoint["model_state"])
    with torch.inference_mode():
        logits = model(x_dict, edge_dict)["bot_logit"][len(train_ids):].cpu()
    probabilities = torch.sigmoid(logits).numpy()
    threshold = float(checkpoint["frozen_threshold"])
    predictions = (probabilities >= threshold).astype(int)

    relnet_probs = None
    relnet_ckpt = None
    if args.relnet_checkpoint:
        relnet_ckpt = torch.load(args.relnet_checkpoint, map_location="cpu", weights_only=False)
        if relnet_ckpt["input_channels"] != x_dict["user"].shape[1]:
            raise RuntimeError("new RelNet checkpoint and test features differ")
        relnet = GatedRelNet(
            input_channels=relnet_ckpt["input_channels"],
            hidden_channels=relnet_ckpt["hidden_channels"],
            relation_names=tuple(relnet_ckpt["relation_names"]),
            num_layers=relnet_ckpt["num_layers"], dropout=relnet_ckpt["dropout"],
        ).cuda().eval()
        relnet.load_state_dict(relnet_ckpt["model_state"])
        with torch.inference_mode():
            relnet_logits = relnet(
                x_dict["user"], {"follows": follows, "rev_follows": follows.flip(0)}
            )[len(train_ids):].cpu()
        relnet_probs = torch.sigmoid(relnet_logits).numpy()

    # The evaluation labels are read only after all features, edges, and scores are frozen.
    labels = [int(row["label"]) for row in test_rows]
    touched = {a for src, dst in cross_edges for a in (src, dst) if a >= len(train_ids)}
    report = {
        "protocol": "frozen train-only HGT; append unlabeled test features and observed train-test follow edges only; exclude test-test/dev edges; no retraining or test calibration",
        "train_json_sha256": sha256_file(args.train_json),
        "test_json_sha256": sha256_file(args.test_json),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "n_train": len(train_ids), "n_test": len(test_ids),
        "n_test_with_train_edge": len(touched),
        "n_cross_edges": len(cross_edges),
        "threshold_from_hgt_val": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "bot_f1": float(f1_score(labels, predictions, pos_label=1)),
        "human_f1": float(f1_score(labels, predictions, pos_label=0)),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    old_report_path = args.output / "official_test_report.json"
    if old_report_path.exists():
        old = json.loads(old_report_path.read_text(encoding="utf-8"))
        for key in ("train_json_sha256", "test_json_sha256", "checkpoint_sha256",
                    "n_train", "n_test", "n_cross_edges", "threshold_from_hgt_val"):
            if old[key] != report[key]:
                raise RuntimeError(f"frozen HGT evaluation protocol changed at {key}")
        for key in ("accuracy", "auc", "macro_f1"):
            if abs(old[key] - report[key]) > 1e-6:
                raise RuntimeError(f"frozen HGT test result changed at {key}")
    old_report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    with (args.output / "official_test_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for uid, label, prob, pred in zip(test_ids, labels, probabilities, predictions):
            stream.write(json.dumps({"account_id": uid, "label": label,
                                     "bot_probability": float(prob), "prediction": int(pred)},
                                    ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if relnet_probs is not None and relnet_ckpt is not None:
        relnet_threshold = float(relnet_ckpt["frozen_threshold"])
        relnet_preds = (relnet_probs >= relnet_threshold).astype(int)
        relnet_report = {
            "status": "new RelNet reimplementation, not historical checkpoint recovery",
            "protocol": report["protocol"],
            "checkpoint_sha256": sha256_file(args.relnet_checkpoint),
            "n_test": len(test_ids), "n_cross_edges": len(cross_edges),
            "threshold_from_relnet_val": relnet_threshold,
            "accuracy": float(accuracy_score(labels, relnet_preds)),
            "auc": float(roc_auc_score(labels, relnet_probs)),
            "macro_f1": float(f1_score(labels, relnet_preds, average="macro")),
        }
        (args.output / "relnet_test_report.json").write_text(
            json.dumps(relnet_report, indent=2), encoding="utf-8")
        with (args.output / "relnet_test_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for uid, label, prob, pred in zip(test_ids, labels, relnet_probs, relnet_preds):
                stream.write(json.dumps({"account_id": uid, "label": label,
                                         "bot_probability": float(prob), "prediction": int(pred)},
                                        ensure_ascii=False) + "\n")
        print(json.dumps(relnet_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
