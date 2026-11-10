"""Embed official TwiBot-20 labeled account text with local frozen Qwen3.

The original row, description and chosen tweet ordinals are recorded. Support
accounts retain the numeric/graph facts but use zero text vectors. The encoder
is frozen and reads no account labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import ijson
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def account_text(row: dict) -> tuple[str, str, list[int]]:
    profile = row.get("profile") or {}
    description = str(profile.get("description") or "").strip()[:500]
    tweets = [str(text or "") for text in (row.get("tweet") or [])]
    original = [i for i, text in enumerate(tweets)
                if not text.strip().startswith("RT @")]
    original_set = set(original)
    chosen = (original + [i for i in range(len(tweets))
                          if i not in original_set])[:5]
    joined = " ".join(tweets[i] for i in chosen)[:500]
    return description, joined, chosen


@torch.inference_mode()
def embed(texts: list[str], tokenizer, model, device: torch.device) -> np.ndarray:
    batch = [text if text.strip() else "(empty)" for text in texts]
    encoded = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(device)
    hidden = model(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled.cpu().numpy().astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0 or not torch.cuda.is_available():
        raise RuntimeError("positive batch size and remote CUDA GPU required")
    with np.load(args.bundle, allow_pickle=False) as bundle:
        ids = bundle["ids"].astype(str)
        split = bundle["split"].astype(str)
        members = bundle["members"].astype(str)
        ordinals = bundle["ordinals"].astype(np.int64)
    if len(ids) != 229580 or int(np.sum(split != "support")) != 11826:
        raise ValueError("official graph population differs")
    model_config = json.loads((args.model / "config.json").read_text())
    width = int(model_config["hidden_size"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda")
    model = AutoModel.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=np.float16, shape=(len(ids), width * 2))
    matrix[:] = 0
    locator_path = args.output.with_suffix(".sources.jsonl")
    cursor = 0
    with zipfile.ZipFile(args.zip) as archive, locator_path.open(
            "w", encoding="utf-8") as source_stream:
        for part in ("train", "dev", "test"):
            member = f"Twibot-20/{part}.json"
            with archive.open(member) as stream:
                iterator = enumerate(ijson.items(stream, "item"))
                while True:
                    block = []
                    for _ in range(args.batch_size):
                        try:
                            block.append(next(iterator))
                        except StopIteration:
                            break
                    if not block:
                        break
                    descriptions, tweet_texts = [], []
                    for offset, (ordinal, row) in enumerate(block):
                        pos = cursor + offset
                        uid = str(row["ID"])
                        if (uid != ids[pos] or split[pos] != part
                                or members[pos] != member or ordinals[pos] != ordinal):
                            raise ValueError(f"text provenance differs at row {pos}")
                        description, tweet_text, selected = account_text(row)
                        descriptions.append(description)
                        tweet_texts.append(tweet_text)
                        source_stream.write(json.dumps({
                            "account_id": uid, "archive_member": member,
                            "row_ordinal": ordinal,
                            "description_sha256": hashlib.sha256(
                                description.encode("utf-8")).hexdigest(),
                            "tweet_ordinals": selected,
                            "joined_tweets_sha256": hashlib.sha256(
                                tweet_text.encode("utf-8")).hexdigest(),
                        }) + "\n")
                    end = cursor + len(block)
                    matrix[cursor:end, :width] = embed(
                        descriptions, tokenizer, model, device)
                    matrix[cursor:end, width:] = embed(
                        tweet_texts, tokenizer, model, device)
                    cursor = end
                    if cursor % 512 < args.batch_size:
                        print(json.dumps({"split": part, "rows_done": cursor}),
                              flush=True)
            print(json.dumps({"split": part, "rows_done": cursor}), flush=True)
    if cursor != 11826 or not np.all(matrix[11826:] == 0):
        raise ValueError("labeled/support text population differs")
    matrix.flush()
    manifest = {
        "feature_family": "frozen_Qwen3_0.6B_description_plus_five_tweets",
        "native_graph_bundle_sha256": sha256(args.bundle),
        "source_archive_sha256": sha256(args.zip),
        "model_config_sha256": sha256(args.model / "config.json"),
        "model_weights_sha256": sha256(args.model / "model.safetensors"),
        "source_locators_sha256": sha256(locator_path),
        "nodes": len(ids), "labeled_text_nodes": cursor,
        "support_text_nodes_zero": len(ids) - cursor,
        "description_width": width, "tweet_width": width,
        "max_text_chars": 500, "max_token_length": 128,
        "label_access": "encoder reads no label",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": manifest}), flush=True)


if __name__ == "__main__":
    main()
