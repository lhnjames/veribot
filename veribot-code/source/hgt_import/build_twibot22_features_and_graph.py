"""Build a leakage-free TwiBot-22 HGT graph with resumable Qwen3-0.6B embeddings.

Only the project's train-only export is read.  The two text channels are embedded by the frozen
local Qwen snapshot and written incrementally to ``.npy`` memmaps, so an interrupted A800 job can
resume without recomputing completed rows.  The final graph uses the same six cross-dataset fields
as the unified pipeline: description, selected tweets, four numeric profile counts, and verified.
The fifth TwiBot-22-only numeric column is deliberately excluded.

Raw edge_type semantics (verified against the exporter):
  * type zero: stored ``src -> dst`` means ``src follows dst``;
  * type one: stored ``src -> dst`` means ``dst follows src`` and is flipped.
The canonical follows graph is deduplicated before its reverse relation is created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData
from transformers import AutoModel, AutoTokenizer


DEFAULT_ROOT = Path("/path/to/data_root/data/twibot22_train")
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "twibot22"
DEFAULT_MODEL_ROOT = Path(
    "/path/to/data_root/cache/huggingface/hub/models--Qwen--Qwen3-0.6B"
)
N_TWEETS = 5
MAX_TEXT_CHARS = 500
MAX_TOKENS = 128


def internal_split(uid: str) -> str:
    return "validation" if int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) % 20 == 0 else "train"


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def resolve_snapshot(model_root: Path) -> tuple[Path, str]:
    candidates = sorted(
        p for p in (model_root / "snapshots").iterdir()
        if p.is_dir() and (p / "config.json").exists()
    )
    if len(candidates) != 1:
        raise RuntimeError(f"expected one local model snapshot under {model_root}, found {candidates}")
    return candidates[0], candidates[0].name


def value_at(container, index: int):
    if isinstance(container, dict):
        if index in container:
            return container[index]
        return container.get(str(index), "")
    return container[index]


def description_text(value) -> str:
    return str(value or "").replace("\n", " ").strip()[:MAX_TEXT_CHARS]


def tweets_text(value) -> str:
    values = value if isinstance(value, (list, tuple)) else ([] if value is None else [value])
    cleaned = [str(v).replace("\n", " ").strip() for v in values if str(v or "").strip()]
    originals = [t for t in cleaned if not t.startswith("RT @")]
    if len(originals) >= N_TWEETS:
        selected = originals[:N_TWEETS]
    else:
        seen = set(originals)
        selected = originals + [t for t in cleaned if t not in seen]
        selected = selected[:N_TWEETS]
    return " ".join(selected)[:MAX_TEXT_CHARS]


@torch.inference_mode()
def embed_batch(tokenizer, model, texts: list[str], device: torch.device) -> np.ndarray:
    normalized = [text if text.strip() else "(empty)" for text in texts]
    encoded = tokenizer(
        normalized, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS
    ).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    return pooled.float().cpu().numpy().astype(np.float16)


def embed_channel(
    name: str,
    source_path: Path,
    out_dir: Path,
    n: int,
    hidden_size: int,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
) -> Path:
    output_path = out_dir / f"{name}_embeddings.float16.npy"
    state_path = out_dir / f"{name}_embedding_progress.json"
    if output_path.exists():
        mmap = np.lib.format.open_memmap(output_path, mode="r+")
        if tuple(mmap.shape) != (n, hidden_size) or mmap.dtype != np.float16:
            raise RuntimeError(f"existing {output_path} has incompatible shape/dtype")
    else:
        mmap = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float16, shape=(n, hidden_size)
        )

    completed = 0
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("source_size") != source_path.stat().st_size:
            raise RuntimeError(f"{name} source changed since partial embedding was produced")
        completed = int(state.get("completed", 0))
    if completed >= n:
        print(f"[{name}] already complete: {completed}/{n}", flush=True)
        return output_path

    print(f"[{name}] loading {source_path} (resume row {completed}/{n})", flush=True)
    with source_path.open("rb") as handle:
        values = pickle.load(handle)
    if isinstance(values, dict):
        bad_keys = [key for key in values if int(key) < 0 or int(key) >= n]
        if bad_keys:
            raise RuntimeError(f"{source_path} contains out-of-range account keys: {bad_keys[:5]}")
        print(f"[{name}] sparse mapping contains {len(values)}/{n} accounts; missing rows use (empty)",
              flush=True)
    elif len(values) != n:
        raise RuntimeError(f"{source_path} has {len(values)} rows, expected {n}")

    formatter = description_text if name == "description" else tweets_text
    for start in range(completed, n, batch_size):
        end = min(start + batch_size, n)
        texts = [formatter(value_at(values, i)) for i in range(start, end)]
        mmap[start:end] = embed_batch(tokenizer, model, texts, device)
        mmap.flush()
        atomic_json(
            state_path,
            {
                "channel": name,
                "completed": end,
                "n_accounts": n,
                "hidden_size": hidden_size,
                "source": str(source_path),
                "source_size": source_path.stat().st_size,
                "output": str(output_path),
            },
        )
        if end == n or end % max(batch_size * 20, 1) == 0:
            print(f"[{name}] embedded {end}/{n}", flush=True)
    del values
    return output_path


def build_graph(root: Path, out_dir: Path, desc_path: Path, tweets_path: Path, model_meta: dict) -> None:
    labels = torch.load(root / "label.pt", map_location="cpu", weights_only=False).long()
    n = len(labels)
    uid_to_idx = torch.load(root / "uid_to_idx.pt", map_location="cpu", weights_only=False)
    if len(uid_to_idx) != n or set(map(int, uid_to_idx.values())) != set(range(n)):
        raise RuntimeError("uid_to_idx is not a complete zero-based mapping")
    account_ids = [""] * n
    for uid, index in uid_to_idx.items():
        account_ids[int(index)] = str(uid)

    edge_index = torch.load(root / "edge_index.pt", map_location="cpu", weights_only=False).long()
    edge_type = torch.load(root / "edge_type.pt", map_location="cpu", weights_only=False).long()
    if edge_index.shape[1] != edge_type.numel():
        raise RuntimeError("edge_index and edge_type lengths differ")
    type_zero = edge_index[:, edge_type == 0]
    type_one = edge_index[:, edge_type == 1].flip(0)
    follows = torch.unique(torch.cat([type_zero, type_one], dim=1), dim=1)

    raw_numeric_np = np.load(root / "num_properties_raw.npy", mmap_mode="r")
    if raw_numeric_np.shape[0] != n or raw_numeric_np.shape[1] < 4:
        raise RuntimeError(f"unexpected numeric tensor shape: {raw_numeric_np.shape}")
    # Cross-dataset fields only: first four counts.  The fifth active-days field is source-only.
    counts = torch.from_numpy(np.asarray(raw_numeric_np[:, :4]).copy()).float().clamp(min=0).log1p()
    categorical = torch.load(
        root / "cat_properties_tensor.pt", map_location="cpu", weights_only=False
    ).float()
    verified = categorical[:, :1]
    raw_numeric = torch.cat([counts, verified], dim=1)

    holdout_mask = torch.tensor([internal_split(uid) == "validation" for uid in account_ids])
    fit_mask = ~holdout_mask
    mean = raw_numeric[fit_mask].mean(0, keepdim=True)
    std = raw_numeric[fit_mask].std(0, keepdim=True).clamp(min=1e-6)
    numeric_norm = (raw_numeric - mean) / std

    desc = torch.from_numpy(np.load(desc_path, mmap_mode="r")).to(torch.float16)
    tweets = torch.from_numpy(np.load(tweets_path, mmap_mode="r")).to(torch.float16)
    if desc.shape[0] != n or tweets.shape != desc.shape:
        raise RuntimeError(f"embedding shape mismatch: desc={desc.shape}, tweets={tweets.shape}, n={n}")
    # Keep the large artifact compact on disk. train_twibot22_hgt.py casts it to float32 on GPU.
    x = torch.cat([desc, tweets, numeric_norm.to(torch.float16)], dim=1)

    data = HeteroData()
    data["user"].x = x
    data["user"].y = labels.float()
    data["user"].holdout_mask = holdout_mask
    data["user"].account_id = account_ids
    data["user", "follows", "user"].edge_index = follows
    data["user", "rev_follows", "user"].edge_index = follows.flip(0)
    graph_path = out_dir / "clean_train_only_graph.pt"
    torch.save(data, graph_path)

    in_degree = torch.bincount(follows[1], minlength=n)
    out_degree = torch.bincount(follows[0], minlength=n)
    manifest = {
        "source": "twibot22",
        "train_only_root": str(root),
        "n_accounts": n,
        "n_project_holdout": int(holdout_mask.sum()),
        "n_hgt_eligible_train": int(fit_mask.sum()),
        "n_raw_edges": int(edge_index.shape[1]),
        "n_canonical_follows_edges": int(follows.shape[1]),
        "n_isolated": int(((in_degree + out_degree) == 0).sum()),
        "feature_dim": int(x.shape[1]),
        "text_embedding_dim": int(desc.shape[1]),
        "profile_features": [
            "log_followers_count", "log_following_count", "log_tweet_count",
            "log_listed_count", "verified",
        ],
        "excluded_source_only_numeric_columns": list(range(4, raw_numeric_np.shape[1])),
        "numeric_fit_scope": "deterministic non-project-holdout official-train accounts only",
        "numeric_mean": mean.squeeze(0).tolist(),
        "numeric_std": std.squeeze(0).tolist(),
        "edge_semantics": {
            "type_zero": "stored src follows dst",
            "type_one": "stored dst follows src; flipped into canonical direction",
        },
        **model_meta,
    }
    atomic_json(out_dir / "clean_train_only_graph_manifest.json", manifest)
    print(f"saved {graph_path} ({graph_path.stat().st_size / 2**30:.2f} GiB)", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--stage", choices=("all", "embed", "build"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    labels = torch.load(args.root / "label.pt", map_location="cpu", weights_only=False)
    n = len(labels)
    snapshot, commit = resolve_snapshot(args.model_root)
    model_meta = {"text_model_snapshot": str(snapshot), "text_model_commit": commit}
    desc_path = args.output / "description_embeddings.float16.npy"
    tweets_path = args.output / "tweets_embeddings.float16.npy"

    if args.stage in {"all", "embed"}:
        if not torch.cuda.is_available():
            raise RuntimeError("embedding extraction is permitted only on the remote CUDA server")
        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(
            snapshot, torch_dtype=torch.bfloat16, local_files_only=True
        ).to(device).eval()
        hidden_size = int(model.config.hidden_size)
        desc_path = embed_channel(
            "description", args.root / "description.pkl", args.output, n, hidden_size,
            tokenizer, model, device, args.batch_size,
        )
        tweets_path = embed_channel(
            "tweets", args.root / "user_tweets.pkl", args.output, n, hidden_size,
            tokenizer, model, device, args.batch_size,
        )
        del model
        torch.cuda.empty_cache()

    if args.stage in {"all", "build"}:
        if not desc_path.exists() or not tweets_path.exists():
            raise RuntimeError("both completed embedding files are required for graph construction")
        for name in ("description", "tweets"):
            state = json.loads((args.output / f"{name}_embedding_progress.json").read_text())
            if int(state.get("completed", 0)) != n:
                raise RuntimeError(f"{name} embeddings are incomplete: {state}")
        build_graph(args.root, args.output, desc_path, tweets_path, model_meta)


if __name__ == "__main__":
    main()
