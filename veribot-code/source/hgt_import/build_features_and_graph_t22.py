"""TwiBot-22 analogue of build_features_and_graph.py, scaled for 700,000 accounts with resumable,
chunked, fp16 embedding generation (so an interruption doesn't require starting over).

Reads ONLY /path/to/data_root/data/twibot22_train/ -- confirmed by its own
EXPORT_MANIFEST.json to be official_split="train_only", contains_validation=false,
contains_test=false, 700000 accounts. No other TwiBot-22 file/directory is touched.

Resumability: embeddings are computed in fixed-size chunks and written to a single pre-allocated
fp16 memory-mapped .npy array on first run; a small JSON state file records which chunk indices are
already done. Re-running the script skips any chunk already marked done, so a killed/crashed run
only redoes its last partial chunk, not the whole 700K accounts.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData
from transformers import AutoModel, AutoTokenizer

SRC_DIR = Path("/path/to/data_root/data/twibot22_train")
OUT_DIR = Path(__file__).resolve().parent / "data_t22"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TWEETS_PER_ACCOUNT = 5
MAX_TEXT_CHARS = 500
EMBED_BATCH_SIZE = 128
CHUNK_SIZE = 5000  # accounts per resumable chunk

MODEL_HUB_ROOT = Path(
    "/path/to/data_root/cache/huggingface/hub/models--Qwen--Qwen3-0.6B")


def resolve_model_snapshot(hub_root: Path) -> tuple[str, str]:
    snapshots_dir = hub_root / "snapshots"
    candidates = [d for d in snapshots_dir.iterdir() if d.is_dir() and (d / "config.json").exists()]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly 1 snapshot under {snapshots_dir}, found {len(candidates)}")
    return str(candidates[0]), candidates[0].name


def internal_split(uid: str) -> str:
    """Verbatim from verifiable_bot/data.py -- must stay byte-identical."""
    return "validation" if int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) % 20 == 0 else "train"


def as_number(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def verify_manifest():
    manifest = json.loads((SRC_DIR / "EXPORT_MANIFEST.json").read_text())
    assert manifest["official_split"] == "train_only"
    assert manifest["contains_validation"] is False
    assert manifest["contains_test"] is False
    print(f"[provenance] EXPORT_MANIFEST.json confirmed: {manifest}")
    return manifest


class ResumableEmbedder:
    """Pre-allocates a fp16 memmap .npy of shape [n, dim] for one text field (description or
    pooled-tweet), plus a JSON state file recording which chunk indices are already written.
    Re-running with the same out_path resumes from the state file rather than recomputing
    everything."""

    def __init__(self, out_path: Path, n: int, dim: int):
        self.out_path = out_path
        self.state_path = out_path.with_suffix(".state.json")
        self.n, self.dim = n, dim
        if out_path.exists():
            self.arr = np.lib.format.open_memmap(out_path, mode="r+", dtype=np.float16, shape=(n, dim))
        else:
            self.arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, dim))
        self.done_chunks: set[int] = set(json.loads(self.state_path.read_text())) if self.state_path.exists() else set()

    def is_chunk_done(self, chunk_idx: int) -> bool:
        return chunk_idx in self.done_chunks

    def write_chunk(self, chunk_idx: int, start: int, end: int, values: np.ndarray):
        self.arr[start:end] = values.astype(np.float16)
        self.arr.flush()
        self.done_chunks.add(chunk_idx)
        # atomic state update: write to temp then replace
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self.done_chunks)))
        tmp.replace(self.state_path)


@torch.no_grad()
def embed_batch(tokenizer, model, texts: list[str]) -> np.ndarray:
    texts = [t if t.strip() else "(empty)" for t in texts]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    out = model(**enc)
    hidden = out.last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    return pooled.float().cpu().numpy()


def run_resumable_embedding(tokenizer, model, texts: list[str], embedder: ResumableEmbedder, label: str):
    n = len(texts)
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_idx in range(n_chunks):
        if embedder.is_chunk_done(chunk_idx):
            continue
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n)
        chunk_texts = texts[start:end]
        chunk_vecs = np.zeros((end - start, embedder.dim), dtype=np.float32)
        for i in range(0, len(chunk_texts), EMBED_BATCH_SIZE):
            batch = chunk_texts[i:i + EMBED_BATCH_SIZE]
            chunk_vecs[i:i + len(batch)] = embed_batch(tokenizer, model, batch)
        embedder.write_chunk(chunk_idx, start, end, chunk_vecs)
        print(f"  [{label}] chunk {chunk_idx + 1}/{n_chunks} done ({end}/{n} accounts) "
              f"{datetime.now(timezone.utc).isoformat()}", flush=True)


def main():
    manifest = verify_manifest()
    n = manifest["accounts"]
    print(f"n accounts (train-only): {n}")

    uid_to_idx: dict[str, int] = torch.load(SRC_DIR / "uid_to_idx.pt", map_location="cpu")
    idx_to_uid = {v: k for k, v in uid_to_idx.items()}
    ids = [idx_to_uid[i] for i in range(n)]

    labels = torch.load(SRC_DIR / "label.pt", map_location="cpu").float()
    assert labels.shape[0] == n

    edge_index = torch.load(SRC_DIR / "edge_index.pt", map_location="cpu").long()
    edge_type = torch.load(SRC_DIR / "edge_type.pt", map_location="cpu").long()

    # direction convention verified against verifiable_bot/data.py's own twibot22_records loader
    # (the ONLY other place this raw edge_type convention is interpreted in this codebase):
    # rel==0 means "stored src follows dst"; rel==1 means "stored row denotes dst follows src".
    src, dst, rel = edge_index[0], edge_index[1], edge_type
    is_type0 = rel == 0
    follows_type0 = torch.stack([src[is_type0], dst[is_type0]], dim=0)          # already "a follows b"
    follows_type1 = torch.stack([dst[~is_type0], src[~is_type0]], dim=0)        # flip to "a follows b"
    follows_all = torch.cat([follows_type0, follows_type1], dim=1)
    follows_all = torch.unique(follows_all, dim=1)
    print(f"follows edges (direction-corrected per data.py's own rel convention, deduped): "
          f"{follows_all.shape[1]}")

    # ── text: real Qwen3-0.6B embeddings, resumable ─────────────────────────────────────────────
    model_path, commit_hash = resolve_model_snapshot(MODEL_HUB_ROOT)
    print(f"loading Qwen3-0.6B from {model_path} (commit {commit_hash})")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).to(DEVICE).eval()
    text_dim = model.config.hidden_size

    print("loading description.pkl and user_tweets.pkl (large, may take a moment)...", flush=True)
    with open(SRC_DIR / "description.pkl", "rb") as f:
        descriptions = pickle.load(f)
    with open(SRC_DIR / "user_tweets.pkl", "rb") as f:
        tweets_by_idx = pickle.load(f)

    desc_texts = [str(descriptions[i] or "").strip()[:MAX_TEXT_CHARS] for i in range(n)]
    tweet_texts = []
    for i in range(n):
        raw_tweets = tweets_by_idx.get(i, []) or []
        original = [t for t in raw_tweets if not str(t).strip().startswith("RT @")]
        chosen = (original[:N_TWEETS_PER_ACCOUNT] if len(original) >= N_TWEETS_PER_ACCOUNT
                  else (original + raw_tweets)[:N_TWEETS_PER_ACCOUNT])
        tweet_texts.append(" ".join(str(t) for t in chosen)[:MAX_TEXT_CHARS])
    del tweets_by_idx  # 8GB pickle, free ASAP once texts are extracted

    desc_embedder = ResumableEmbedder(OUT_DIR / "desc_emb_fp16.npy", n, text_dim)
    tweet_embedder = ResumableEmbedder(OUT_DIR / "tweet_emb_fp16.npy", n, text_dim)

    run_resumable_embedding(tokenizer, model, desc_texts, desc_embedder, "description")
    run_resumable_embedding(tokenizer, model, tweet_texts, tweet_embedder, "tweet")
    del model
    torch.cuda.empty_cache()

    # ── profile numerics, split-aware standardization ───────────────────────────────────────────
    num = np.load(SRC_DIR / "num_properties_raw.npy", mmap_mode="r")  # [n, 5]: followers, following, tweet, listed, active_days
    cat = torch.load(SRC_DIR / "cat_properties_tensor.pt", map_location="cpu")  # [n, >=1], col0 = verified
    followers = torch.tensor(np.asarray(num[:, 0]), dtype=torch.float32)
    following = torch.tensor(np.asarray(num[:, 1]), dtype=torch.float32)
    tweet_count = torch.tensor(np.asarray(num[:, 2]), dtype=torch.float32)
    listed_count = torch.tensor(np.asarray(num[:, 3]), dtype=torch.float32)
    active_years = torch.tensor(np.asarray(num[:, 4]), dtype=torch.float32) / 365.25
    verified = cat[:, 0].float()

    raw_numeric = torch.stack(
        [followers.clamp(min=0).log1p(), following.clamp(min=0).log1p(),
         tweet_count.clamp(min=0).log1p(), listed_count.clamp(min=0).log1p(),
         verified, active_years], dim=1,
    )

    splits = [internal_split(uid) for uid in ids]
    holdout_mask = torch.tensor([s == "validation" for s in splits], dtype=torch.bool)
    non_holdout = ~holdout_mask
    mean = raw_numeric[non_holdout].mean(dim=0, keepdim=True)
    std = raw_numeric[non_holdout].std(dim=0, keepdim=True).clamp(min=1e-6)
    numeric_norm = (raw_numeric - mean) / std
    print(f"numeric stats fit on {non_holdout.sum().item()} non-holdout accounts only "
          f"(holdout={holdout_mask.sum().item()})")

    desc_emb = torch.from_numpy(np.asarray(desc_embedder.arr)).float()
    tweet_emb = torch.from_numpy(np.asarray(tweet_embedder.arr)).float()
    x_user = torch.cat([desc_emb, tweet_emb, numeric_norm], dim=1)
    print(f"final user feature dim: {x_user.shape[1]}")

    data = HeteroData()
    data["user"].x = x_user
    data["user"].y = labels
    data["user"].holdout_mask = holdout_mask
    data["user"].account_id = ids
    data["user", "follows", "user"].edge_index = follows_all
    data["user", "rev_follows", "user"].edge_index = follows_all.flip(0)

    out_path = OUT_DIR / "clean_train_only_graph_t22.pt"
    torch.save(data, out_path)
    print(f"saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    from collections import Counter
    label_counts_all = Counter(int(v.item()) for v in labels)
    label_counts_holdout = Counter(int(labels[i].item()) for i in range(n) if holdout_mask[i])
    manifest_out = {
        "source_dir": str(SRC_DIR),
        "source_export_manifest": manifest,
        "n_accounts": n,
        "n_follows_edges": int(follows_all.shape[1]),
        "n_holdout": int(holdout_mask.sum().item()),
        "holdout_frac": float(holdout_mask.float().mean().item()),
        "feature_dim": int(x_user.shape[1]),
        "text_embed_dim": text_dim,
        "text_model": model_path,
        "text_model_commit_hash": commit_hash,
        "label_counts_all": {str(k): v for k, v in label_counts_all.items()},
        "label_counts_holdout": {str(k): v for k, v in label_counts_holdout.items()},
    }
    with open(OUT_DIR / "clean_train_only_graph_t22_manifest.json", "w") as f:
        json.dump(manifest_out, f, indent=2)
    print(json.dumps(manifest_out, indent=2))


if __name__ == "__main__":
    main()
