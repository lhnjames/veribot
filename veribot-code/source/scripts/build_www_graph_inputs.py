"""Build leak-checked Fox8/BotSim graph inputs on the remote CUDA server.

This is a NEW feature pipeline. It uses a frozen Qwen encoder, train-fitted
numeric scaling, and label-free transductive graph construction. It does not
reproduce missing SAHG/RoBERTa features from the historical manuscript.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.data import HeteroData
from transformers import AutoModel, AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value) -> float:
    try:
        return float(str(value).strip() or 0)
    except (TypeError, ValueError):
        return 0.0


def subreddits(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value or "[]")
    except (SyntaxError, ValueError):
        parsed = []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def load_fox(path: Path):
    ids, labels, descriptions, contents, numerics = [], [], [], [], []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            tweets = row.get("user_tweets") or []
            profile = (tweets[0].get("user") or {}) if tweets else {}
            texts = [str(tweet.get("text") or "") for tweet in tweets]
            original = [text for text in texts if text and not text.lstrip().startswith("RT @")]
            chosen = (original[:5] if len(original) >= 5 else (original + texts)[:5])
            ids.append(str(row["user_id"]))
            labels.append(int(row["label"] == "bot"))
            descriptions.append(str(profile.get("description") or "")[:500])
            contents.append(" ".join(chosen)[:500])
            numerics.append([
                number(profile.get("followers_count")), number(profile.get("friends_count")),
                number(profile.get("statuses_count")), number(profile.get("listed_count")),
                1.0 if str(profile.get("verified")).strip() == "True" else 0.0,
            ])
    return ids, labels, descriptions, contents, numerics, None


def load_botsim(users_csv: Path, posts_json: Path):
    with users_csv.open(encoding="utf-8-sig", newline="") as stream:
        users = list(csv.DictReader(stream))
    if len(users) != 2907:
        raise RuntimeError("BotSim Users.csv must have 2907 rows")
    records = json.loads(posts_json.read_text(encoding="utf-8"))
    ids = [row["user_id"] for row in users]
    labels = [0] * 1907 + [1] * 1000
    descriptions, contents, numerics, subreddit_lists = [], [], [], []
    for row in users:
        record = records.get(row["user_id"]) or {}
        post_texts = [str(post.get("posts") or "") for post in record.get("posts") or []]
        comment_texts = [str(comment.get("comment_body") or "")
                         for level in ("comment_1", "comment_2")
                         for comment in record.get(level) or []]
        subs = subreddits(row.get("subreddit") or "")
        descriptions.append(str(row.get("description") or "")[:500])
        contents.append(" ".join((post_texts[:3] + comment_texts[:2]))[:500])
        numerics.append([
            number(row.get("submission_num")), number(row.get("comment_num")),
            number(row.get("comment_num_1")), number(row.get("comment_num_2")),
            float(len(subs)),
        ])
        subreddit_lists.append(subs)
    return ids, labels, descriptions, contents, numerics, (subreddit_lists, records)


@torch.inference_mode()
def embed(tokenizer, encoder, texts: list[str]) -> torch.Tensor:
    chunks = []
    for start in range(0, len(texts), 64):
        batch = [(text.strip() or "(empty)") for text in texts[start:start + 64]]
        tokens = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=128).to("cuda")
        hidden = encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        chunks.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)).cpu())
        if (start // 64) % 10 == 0:
            print(f"encoded {start + len(batch)}/{len(texts)}", flush=True)
    return torch.cat(chunks)


def split_masks(split: dict, ids: list[str]) -> dict[str, torch.Tensor]:
    assigned = set()
    masks = {}
    id_sets = {}
    for name in ("train", "dev", "test"):
        part = split["splits"][name]
        indices = part["indices"]
        assert [ids[index] for index in indices] == part["account_ids"]
        assert not assigned.intersection(indices)
        assigned.update(indices)
        id_sets[name] = set(part["account_ids"])
        mask = torch.zeros(len(ids), dtype=torch.bool)
        mask[indices] = True
        masks[name] = mask
    assert len(assigned) == len(ids)
    assert not (id_sets["train"] & id_sets["dev"] or
                id_sets["train"] & id_sets["test"] or id_sets["dev"] & id_sets["test"])
    return masks


def fox_graph(data: HeteroData) -> dict:
    x = data["user"].x.cuda()
    desc, content, num = x[:, :1024], x[:, 1024:2048], x[:, 2048:]
    blocks = [part / part.norm(dim=-1, keepdim=True).clamp(min=1e-8)
              for part in (desc, content, num)]
    normalized = torch.cat(blocks, dim=-1)
    normalized = normalized / normalized.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-2.0)
    neighbors = similarity.topk(10, dim=1).indices
    source = torch.arange(len(x), device="cuda").unsqueeze(1).expand(-1, 10).reshape(-1)
    directed = torch.stack([source, neighbors.reshape(-1)])
    edges = torch.unique(torch.cat([directed, directed.flip(0)], dim=1), dim=1).cpu()
    data["user", "knn_sim", "user"].edge_index = edges
    return {"knn_edges": int(edges.shape[1]), "knn_k": 10,
            "graph_note": "label-free transductive kNN over three normalized feature blocks; similarity, not observed social ties"}


def botsim_graph(data: HeteroData, ids: list[str], extra) -> dict:
    subreddit_lists, records = extra
    names = sorted({name for items in subreddit_lists for name in items})
    sub_idx = {name: index for index, name in enumerate(names)}
    user_sub = [(u, sub_idx[name]) for u, items in enumerate(subreddit_lists) for name in items]
    if not user_sub:
        raise RuntimeError("BotSim has no subreddit memberships")
    user_sub_edges = torch.tensor(user_sub, dtype=torch.long).t().contiguous()
    data["subreddit"].x = torch.ones((len(names), 1), dtype=torch.float32)
    data["user", "in_subreddit", "subreddit"].edge_index = user_sub_edges
    data["subreddit", "has_user", "user"].edge_index = user_sub_edges.flip(0)

    id_to_idx = {uid: index for index, uid in enumerate(ids)}
    object_author = {}
    for uid, record in records.items():
        for post in record.get("posts") or []:
            if post.get("submission_id"):
                object_author[str(post["submission_id"]).split("_", 1)[-1]] = uid
        for level in ("comment_1", "comment_2"):
            for comment in record.get(level) or []:
                if comment.get("comment_id"):
                    object_author[str(comment["comment_id"]).split("_", 1)[-1]] = uid
    replies = set()
    for uid, record in records.items():
        if uid not in id_to_idx:
            continue
        for level in ("comment_1", "comment_2"):
            for comment in record.get(level) or []:
                parent = str(comment.get("parent_id") or comment.get("link_id") or "")
                other = object_author.get(parent.split("_", 1)[-1])
                if other in id_to_idx and other != uid:
                    replies.add((id_to_idx[uid], id_to_idx[other]))
    reply_edges = (torch.tensor(sorted(replies), dtype=torch.long).t().contiguous()
                   if replies else torch.empty((2, 0), dtype=torch.long))
    data["user", "reply_to", "user"].edge_index = reply_edges
    data["user", "replied_by", "user"].edge_index = reply_edges.flip(0)

    members = defaultdict(list)
    for user, items in enumerate(subreddit_lists):
        for name in items:
            members[name].append(user)
    cosub = set()
    for users in members.values():
        if len(users) > 500:
            continue  # predefined cap in the prior project script; omit broad mega-subreddits
        for a in users:
            for b in users:
                if a != b:
                    cosub.add((a, b))
    cosub_edges = (torch.tensor(sorted(cosub), dtype=torch.long).t().contiguous()
                   if cosub else torch.empty((2, 0), dtype=torch.long))
    data["user", "cosubreddit", "user"].edge_index = cosub_edges
    return {"subreddit_nodes": len(names), "user_subreddit_edges": len(user_sub),
            "reply_edges": len(replies), "cosubreddit_projection_edges": len(cosub),
            "graph_note": "subreddit membership and reply_to are observed/derived from source IDs; cosubreddit is a projection, not a direct social tie; subreddits with >500 users excluded from projection"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("fox8_23", "botsim_24"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("build embeddings and graphs on the remote CUDA server")
    root = args.root
    source = (root / "data/raw/fox8_23/fox8_23_dataset.ndjson" if args.dataset == "fox8_23"
              else root / "data/raw/botsim_24/Users.csv")
    split = json.loads((root / "data/splits" / f"{args.dataset}_seed0.json").read_text())
    if sha256_file(source) != split["source_sha256"]:
        raise RuntimeError("raw data checksum does not match fixed split")
    if args.dataset == "fox8_23":
        ids, labels, descriptions, contents, numerics, extra = load_fox(source)
    else:
        ids, labels, descriptions, contents, numerics, extra = load_botsim(
            source, root / "data/raw/botsim_24/user_post_comment.json")
    masks = split_masks(split, ids)
    raw_num = torch.tensor(numerics, dtype=torch.float32)
    raw_num[:, :-1] = raw_num[:, :-1].clamp(min=0).log1p()
    mean = raw_num[masks["train"]].mean(0, keepdim=True)
    std = raw_num[masks["train"]].std(0, keepdim=True).clamp(min=1e-6)
    norm_num = (raw_num - mean) / std
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32).cuda().eval()
    print(f"{args.dataset}: encoding {len(ids)} descriptions", flush=True)
    desc = embed(tokenizer, encoder, descriptions)
    print(f"{args.dataset}: encoding {len(ids)} selected content fields", flush=True)
    content = embed(tokenizer, encoder, contents)
    del encoder
    torch.cuda.empty_cache()
    if desc.shape[1] != 1024 or content.shape[1] != 1024:
        raise RuntimeError("expected Qwen3-0.6B hidden size 1024")
    data = HeteroData()
    data["user"].x = torch.cat([desc, content, norm_num], dim=1)
    data["user"].y = torch.tensor(labels, dtype=torch.long)
    data["user"].account_id = ids
    for name, mask in masks.items():
        data["user"][f"{name}_mask"] = mask
    graph_report = fox_graph(data) if args.dataset == "fox8_23" else botsim_graph(data, ids, extra)
    args.output.mkdir(parents=True, exist_ok=True)
    graph_path = args.output / "graph.pt"
    torch.save(data, graph_path)
    manifest = {
        "dataset": args.dataset, "source": str(source),
        "source_sha256": split["source_sha256"],
        "split_source": str(root / "data/splits" / f"{args.dataset}_seed0.json"),
        "model": str(args.model), "model_weights_sha256": sha256_file(args.model / "model.safetensors"),
        "feature_dim": int(data["user"].x.shape[1]),
        "feature_blocks": {"description": 1024, "selected_content": 1024, "numeric": 5},
        "numeric_fit": "train split only", "graph_population": "all train/dev/test features, no labels used for edge construction",
        "split_sizes": {name: int(mask.sum()) for name, mask in masks.items()},
        "n_user_nodes": len(ids), "n_unique_account_ids": len(set(ids)),
        "graph": graph_report, "graph_path": str(graph_path),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
