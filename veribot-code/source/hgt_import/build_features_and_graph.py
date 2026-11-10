"""Build node features (text embeddings via frozen Qwen3-0.6B + profile numerics) and the
follows/rev_follows graph directly from data/twibot20_complete/train.json's real account IDs and
real neighbor lists -- NOT from data.py's AccountRecord conversion, which anonymizes neighbor
targets to positional labels ("n0", "n1", ...) and is unusable for real neighbor identity.

Reads ONLY train.json. Never globs the data directory, never opens test.json. sha256 of the input
file is recorded in the output manifest for audit.

neighbor direction (per train.json's own field names, confirmed by inspection):
  row["neighbor"]["following"] = list of account IDs this account follows  -> edge (self, target)
  row["neighbor"]["follower"]  = list of account IDs that follow this account -> edge (target, self)
Edges are kept only when BOTH endpoints are in this file's own 8278-account ID set (the graph has no
visibility into accounts outside this pool at all -- consistent with never reading test.json).
754/8278 accounts have neighbor=None (no graph data at all); these become isolated nodes, handled by
the model's residual path (see hgt_model.py's VerifiableHGT changes), not dropped or zero-featured.

Text features: frozen Qwen3-0.6B (no gradient, no fine-tuning -- pure embedding extraction), mean-
pooled last-hidden-state over non-pad tokens, run on the remote GPU. Two embeddings per account:
description (bio) and a pooled representative-tweet embedding (first N original/non-retweet tweets
if available, else first N tweets of any kind, else zero vector if the account has none).

Numeric profile features (followers_count, following_count [=friends_count, matching the project's
existing name unification], tweet_count [=statuses_count], listed_count, verified) are standardized
using mean/std fit ONLY on this project's own internal_split()
non-holdout 95% (verifiable_bot/data.py's exact rule) -- the holdout 5% is transformed with those
same stats, never included in fitting them, so no holdout information leaks into feature scaling.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from transformers import AutoModel, AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

TRAIN_JSON = Path(os.environ.get(
    "VERIBOT_TWIBOT20_TRAIN", Path(__file__).resolve().parents[2] / "data/raw/twibot20/train.json"))
MODEL_HUB_ROOT = Path(os.environ.get(
    "VERIBOT_QWEN_MODEL", Path(__file__).resolve().parents[2] / "data/models/Qwen3-0.6B"))


def resolve_model_snapshot(hub_root: Path) -> tuple[str, str]:
    """HF hub cache layout is <hub_root>/snapshots/<commit_hash>/config.json, ... -- resolve the
    actual loadable directory instead of hardcoding a commit hash, but still fail loudly (not
    silently pick an arbitrary one) if there isn't exactly one candidate, and return the commit
    hash for the audit manifest."""
    if (hub_root / "config.json").exists():
        return str(hub_root), os.environ.get("VERIBOT_QWEN_REVISION", "unrecorded-direct-directory")
    snapshots_dir = hub_root / "snapshots"
    candidates = [d for d in snapshots_dir.iterdir() if d.is_dir() and (d / "config.json").exists()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly 1 snapshot with a config.json under {snapshots_dir}, "
            f"found {len(candidates)}: {candidates}")
    return str(candidates[0]), candidates[0].name
OUT_DIR = Path(os.environ.get("VERIBOT_HGT_DATA_DIR", Path(__file__).resolve().parent / "data"))
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TWEETS_PER_ACCOUNT = 5
MAX_TEXT_CHARS = 500
EMBED_BATCH_SIZE = 64


def internal_split(uid: str) -> str:
    """Verbatim from verifiable_bot/data.py -- must stay byte-identical."""
    return "validation" if int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) % 20 == 0 else "train"


def as_number(v) -> float:
    try:
        return float(str(v).strip() or 0)
    except (TypeError, ValueError):
        return 0.0


def load_rows() -> list[dict]:
    with open(TRAIN_JSON, encoding="utf-8") as f:
        rows = json.load(f)
    return rows


@torch.no_grad()
def embed_texts(tokenizer, model, texts: list[str]) -> torch.Tensor:
    """Frozen mean-pooled embeddings, batched. Empty string -> embed the empty string itself (a
    real, consistent embedding for "no text"), not a hand-picked zero vector, so the model can
    still relate it to other sparse-text accounts rather than treating it as a special out-of-
    distribution value."""
    out_chunks = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        batch = [t if t.strip() else "(empty)" for t in batch]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                         max_length=128).to(DEVICE)
        out = model(**enc)
        hidden = out.last_hidden_state  # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        out_chunks.append(pooled.cpu())
        if (i // EMBED_BATCH_SIZE) % 10 == 0:
            print(f"  embedded {i + len(batch)}/{len(texts)}", flush=True)
    return torch.cat(out_chunks, dim=0)


def main():
    if not TRAIN_JSON.is_file():
        raise FileNotFoundError(f"TwiBot-20 training data missing: {TRAIN_JSON}")
    input_sha256 = sha256_file(TRAIN_JSON)
    expected_sha256 = os.environ.get("VERIBOT_EXPECT_TWIBOT20_SHA256")
    if expected_sha256 and input_sha256 != expected_sha256:
        raise ValueError(f"train.json sha256 {input_sha256} differs from expected {expected_sha256}")
    print(f"train.json sha256: {input_sha256}")

    rows = load_rows()
    n = len(rows)
    print(f"loaded {n} accounts from train.json")

    ids = [r["ID"] for r in rows]
    id_set = set(ids)
    id_to_idx = {uid: i for i, uid in enumerate(ids)}
    labels = torch.tensor([int(r["label"]) for r in rows], dtype=torch.float)

    # ── edges: real IDs, both endpoints must be in this file's own pool ────────────────────────
    edge_src, edge_dst = [], []
    n_isolated = 0
    for i, r in enumerate(rows):
        nb = r.get("neighbor")
        if nb is None:
            n_isolated += 1
            continue
        for target in (nb.get("following") or []):
            if target in id_set:
                edge_src.append(i)
                edge_dst.append(id_to_idx[target])  # self -> target ("self follows target")
        for target in (nb.get("follower") or []):
            if target in id_set:
                edge_src.append(id_to_idx[target])  # target -> self ("target follows self")
                edge_dst.append(i)

    follows_edge = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    follows_edge = torch.unique(follows_edge, dim=1)
    print(f"follows edges (both endpoints in train.json pool, deduped): {follows_edge.shape[1]}")
    print(f"accounts with neighbor=None (isolated, no graph edges): {n_isolated}/{n}")

    # ── text embeddings (frozen Qwen3-0.6B, remote GPU) ─────────────────────────────────────────
    model_path, commit_hash = resolve_model_snapshot(MODEL_HUB_ROOT)
    print(f"loading Qwen3-0.6B for frozen embedding extraction from {model_path} "
          f"(commit {commit_hash})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).to(DEVICE).eval()

    desc_texts = [(r["profile"].get("description") or "").strip()[:MAX_TEXT_CHARS] for r in rows]
    tweet_texts = []
    for r in rows:
        tweets = r.get("tweet") or []
        original = [t for t in tweets if not t.strip().startswith("RT @")]
        chosen = (original[:N_TWEETS_PER_ACCOUNT] if len(original) >= N_TWEETS_PER_ACCOUNT
                  else (original + tweets)[:N_TWEETS_PER_ACCOUNT])
        tweet_texts.append(" ".join(chosen)[:MAX_TEXT_CHARS])

    print(f"embedding {n} descriptions...")
    desc_emb = embed_texts(tokenizer, model, desc_texts)
    print(f"embedding {n} pooled-tweet texts...")
    tweet_emb = embed_texts(tokenizer, model, tweet_texts)
    text_dim = desc_emb.shape[1]
    print(f"text embedding dim: {text_dim}")

    del model
    torch.cuda.empty_cache()

    # ── profile numeric features, split-aware standardization ──────────────────────────────────
    followers = torch.tensor([as_number(r["profile"].get("followers_count")) for r in rows])
    following = torch.tensor([as_number(r["profile"].get("friends_count")) for r in rows])
    tweet_count = torch.tensor([as_number(r["profile"].get("statuses_count")) for r in rows])
    listed_count = torch.tensor([as_number(r["profile"].get("listed_count")) for r in rows])
    verified = torch.tensor([1.0 if str(r["profile"].get("verified")).strip() == "True" else 0.0
                              for r in rows])
    raw_numeric = torch.stack(
        [followers.log1p(), following.log1p(), tweet_count.log1p(), listed_count.log1p(),
         verified], dim=1,
    )

    splits = [internal_split(uid) for uid in ids]
    holdout_mask = torch.tensor([s == "validation" for s in splits], dtype=torch.bool)
    non_holdout = ~holdout_mask
    mean = raw_numeric[non_holdout].mean(dim=0, keepdim=True)
    std = raw_numeric[non_holdout].std(dim=0, keepdim=True).clamp(min=1e-6)
    numeric_norm = (raw_numeric - mean) / std
    print(f"numeric feature stats fit on {non_holdout.sum().item()} non-holdout accounts only "
          f"(holdout={holdout_mask.sum().item()} accounts transformed with those same stats)")

    x_user = torch.cat([desc_emb, tweet_emb, numeric_norm], dim=1)
    print(f"final user feature dim: {x_user.shape[1]} "
          f"(desc={text_dim} + tweet={text_dim} + numeric={numeric_norm.shape[1]})")

    data = HeteroData()
    data["user"].x = x_user
    data["user"].y = labels
    data["user"].holdout_mask = holdout_mask
    data["user"].account_id = ids
    data["user", "follows", "user"].edge_index = follows_edge
    data["user", "rev_follows", "user"].edge_index = follows_edge.flip(0)

    out_path = OUT_DIR / "clean_train_only_graph.pt"
    torch.save(data, out_path)
    print(f"saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    from collections import Counter
    label_counts_all = Counter(int(v.item()) for v in labels)
    label_counts_holdout = Counter(int(labels[i].item()) for i in range(n) if holdout_mask[i])
    manifest = {
        "input_file": str(TRAIN_JSON),
        "input_sha256": input_sha256,
        "n_accounts": n,
        "n_isolated_no_neighbor_field": n_isolated,
        "n_follows_edges": int(follows_edge.shape[1]),
        "n_holdout": int(holdout_mask.sum().item()),
        "holdout_frac": float(holdout_mask.float().mean().item()),
        "feature_dim": int(x_user.shape[1]),
        "text_embed_dim": text_dim,
        "text_model": model_path,
        "text_model_commit_hash": commit_hash,
        "text_model_weights_sha256": sha256_file(Path(model_path) / "model.safetensors"),
        "n_tweets_per_account": N_TWEETS_PER_ACCOUNT,
        "profile_features": ["log_followers_count", "log_following_count", "log_tweet_count",
                             "log_listed_count", "verified"],
        "label_counts_all": {str(k): v for k, v in label_counts_all.items()},
        "label_counts_holdout": {str(k): v for k, v in label_counts_holdout.items()},
    }
    with open(OUT_DIR / "clean_train_only_graph_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
