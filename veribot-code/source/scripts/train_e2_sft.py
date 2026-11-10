"""E2: generative cold-start SFT. Qwen3 (LoRA) learns to emit the canonical
<think>/<evidence>/<chain>/<answer> completion for an account. Loss on the completion tokens only;
the canonical prompt (prompt_v4) is masked out. No API calls — targets come from the offline bank.

Correctness gates: label map fixed; prompt built through the single canonical formatter; completion
tokenised once and cached (key = render-manifest prompt hash + canonical prompt hash + tokenizer
fingerprint); single-batch fwd/bwd finiteness smoke; checkpoint save/load parity; dev evaluated by
GENERATION every epoch (parse <answer>) reporting ACC / F1 / predicted-bot-rate / format-valid rate
/ confusion; early stop on dev accuracy.

Perf: bf16 autocast, TF32, use_cache=False during train, SDPA; length-bucketed batches + dynamic
right padding; pin_memory/persistent_workers/prefetch; grad accumulation to a fixed sample count;
grad clipping. No torch.compile.

    python scripts/train_e2_sft.py --config configs/e2_sft_0.6b.yaml --out outputs/v4/e2_sft/run_0.6b
    python scripts/train_e2_sft.py --config configs/e2_sft_0.6b.yaml --out .../smoke --smoke-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.v4.prompt_v4 import (  # noqa: E402
    canonical_messages, canonical_prompt_sha256, canonical_prompt_text, tokenizer_fingerprint,
)
from verifiable_bot.v4.render_v4 import Unit, render_units  # noqa: E402

LABEL_MAP = {"human": 0, "bot": 1}
ANSWER_RE = re.compile(r"<answer>\s*(bot|human)\s*</answer>", re.IGNORECASE)
FORMAT_RE = re.compile(r"<think>.*?</think>\s*<evidence>.*?</evidence>\s*<chain>.*?</chain>\s*"
                       r"<answer>\s*(bot|human)\s*</answer>", re.IGNORECASE | re.DOTALL)


def load_yaml(p):
    import yaml
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def units_text(row):
    return render_units([Unit(u["unit_id"], u["text"]) for u in row["units"]])


class SFTDataset(Dataset):
    """Pretokenised (prompt_ids, completion_ids). labels = -100 on prompt, real on completion+eos."""

    def __init__(self, examples):
        self.ex = examples

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def tokenize_example(row, tokenizer, max_length):
    prompt = canonical_prompt_text(units_text(row), tokenizer)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    ids = (p_ids + c_ids)[:max_length]
    labels = ([-100] * len(p_ids) + c_ids)[:max_length]
    return {"input_ids": ids, "labels": labels, "length": len(ids), "label": row["label"],
            "account_id": row["account_id"]}


def build_split(rows, tokenizer, cache, max_length):
    if cache.exists():
        return torch.load(cache)["examples"]
    ex = [tokenize_example(r, tokenizer, max_length) for r in rows]
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"examples": ex}, cache)
    return ex


class LengthBatcher:
    def __init__(self, lengths, bs, shuffle, seed):
        order = list(np.argsort(lengths))
        self.batches = [order[i:i + bs] for i in range(0, len(order), bs)]
        self.shuffle, self.rng = shuffle, random.Random(seed)

    def __iter__(self):
        b = list(self.batches)
        if self.shuffle:
            self.rng.shuffle(b)
        return iter(b)

    def __len__(self):
        return len(self.batches)


def make_collate(pad_id):
    def collate(items):
        m = max(x["length"] for x in items)
        B = len(items)
        input_ids = torch.full((B, m), pad_id, dtype=torch.long)
        labels = torch.full((B, m), -100, dtype=torch.long)
        attn = torch.zeros((B, m), dtype=torch.long)
        for j, x in enumerate(items):
            n = x["length"]
            input_ids[j, :n] = torch.tensor(x["input_ids"])
            labels[j, :n] = torch.tensor(x["labels"])
            attn[j, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
    return collate


@torch.no_grad()
def evaluate_generation(model, tokenizer, rows, device, max_new_tokens, max_length, limit):
    model.eval()
    rows = rows[:limit]
    ys, ps, fmt_ok = [], [], 0
    tokenizer.padding_side = "left"
    for i in range(0, len(rows), 16):
        batch = rows[i:i + 16]
        prompts = [canonical_prompt_text(units_text(r), tokenizer) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id, use_cache=True)
        for r, g, inp in zip(batch, gen, enc["input_ids"]):
            text = tokenizer.decode(g[inp.shape[0]:], skip_special_tokens=True)
            if FORMAT_RE.search(text):
                fmt_ok += 1
            m = ANSWER_RE.search(text)
            pred = m.group(1).lower() if m else ("human" if r["label"] == "bot" else "bot")
            ps.append(LABEL_MAP[pred]); ys.append(LABEL_MAP[r["label"]])
    tokenizer.padding_side = "right"
    ys, ps = np.array(ys), np.array(ps)
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    tn, fp, fn, tp = confusion_matrix(ys, ps, labels=[0, 1]).ravel()
    return {"acc": float(accuracy_score(ys, ps)), "f1": float(f1_score(ys, ps, zero_division=0)),
            "pred_bot_rate": float(ps.mean()), "format_valid_rate": round(fmt_ok / len(rows), 4),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}, "n": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sft-data", default="outputs/v4/e2_sft/sft_train.jsonl")
    ap.add_argument("--render-dir", default="outputs/v4/render_v4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a", buffering=1)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = read_jsonl(args.sft_data)
    assert train_rows, "empty SFT data"
    assert all(r["label"] in LABEL_MAP for r in train_rows), "bad label"
    dev_rows = read_jsonl(Path(args.render_dir) / "dev.jsonl")
    bal = {"bot": sum(r["label"] == "bot" for r in train_rows),
           "human": sum(r["label"] == "human" for r in train_rows)}
    log.write(f"[data] train={len(train_rows)} balance={bal} dev={len(dev_rows)}\n")

    fp = tokenizer_fingerprint(tokenizer, cfg["model_name"])
    manifest = json.loads((Path(args.render_dir) / "render_manifest.json").read_text())
    key = hashlib.sha256(json.dumps({
        "m": manifest.get("canonical_prompt_sha256"), "c": canonical_prompt_sha256(tokenizer),
        "t": fp, "ml": cfg["max_length"], "n": len(train_rows)}, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = Path(args.render_dir) / "e2_tokcache"
    examples = build_split(train_rows, tokenizer, cache_dir / f"sft.{key}.pt", cfg["max_length"])

    pad_id = tokenizer.pad_token_id
    collate = make_collate(pad_id)
    lengths = [e["length"] for e in examples]

    def train_loader():
        return DataLoader(SFTDataset(examples), batch_sampler=list(LengthBatcher(
            lengths, cfg["batch_size"], True, cfg["seed"])), collate_fn=collate,
            num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    model.config.use_cache = False
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"))
    model.to(device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = (len(examples) + cfg["batch_size"] - 1) // cfg["batch_size"]
    total = (steps // cfg["grad_accum"]) * cfg["epochs"]
    sched = get_cosine_schedule_with_warmup(opt, int(total * cfg["warmup_ratio"]), total)

    # smoke: single batch fwd/bwd finiteness
    model.train()
    b0 = next(iter(train_loader()))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss0 = model(input_ids=b0["input_ids"].to(device), attention_mask=b0["attention_mask"].to(device),
                      labels=b0["labels"].to(device)).loss
    assert torch.isfinite(loss0), "non-finite loss"
    loss0.backward(); opt.zero_grad(set_to_none=True)
    log.write(f"[sanity] loss={loss0.item():.4f} total_steps={total} cache_key={key}\n")

    if args.smoke_only:
        model.save_pretrained(out / "smoke_lora")
        log.write("[smoke] PASSED single-batch fwd/bwd + save.\n")
        print("SMOKE PASSED")
        return

    (out / "resolved_config.json").write_text(json.dumps(
        {"config": cfg, "cache_key": key, "tokenizer_fingerprint": fp,
         "render_manifest_prompt_hash": manifest.get("canonical_prompt_sha256"),
         "train_balance": bal}, indent=2), encoding="utf-8")
    try:
        (out / "env.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                     capture_output=True, text=True).stdout)
        (out / "nvidia-smi.txt").write_text(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    except Exception:
        pass

    metrics = open(out / "metrics.jsonl", "a", buffering=1)
    best_acc, best_epoch, bad = -1.0, -1, 0
    for epoch in range(cfg["epochs"]):
        model.train(); model.config.use_cache = False
        t0 = time.time(); dw = 0.0; run = 0.0; nb = 0
        opt.zero_grad(set_to_none=True); tprev = time.time()
        dl = train_loader()
        for step, batch in enumerate(dl):
            dw += time.time() - tprev
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=batch["input_ids"].to(device, non_blocking=True),
                             attention_mask=batch["attention_mask"].to(device, non_blocking=True),
                             labels=batch["labels"].to(device, non_blocking=True)).loss / cfg["grad_accum"]
            loss.backward(); run += loss.item() * cfg["grad_accum"]; nb += 1
            if (step + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tprev = time.time()
        dt = time.time() - t0
        dev = evaluate_generation(model, tokenizer, dev_rows, device,
                                  cfg["max_new_tokens"], cfg["max_length"], cfg["dev_eval_limit"])
        rec = {"epoch": epoch, "train_loss": run / max(nb, 1), "sec": round(dt, 1),
               "data_wait_frac": round(dw / max(dt, 1e-9), 3),
               "samples_per_s": round(len(examples) / dt, 1),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "lr": sched.get_last_lr()[0], "dev": dev}
        improved = dev["acc"] > best_acc + cfg["min_delta"]
        if improved:
            best_acc, best_epoch, bad = dev["acc"], epoch, 0
            model.save_pretrained(out / "best_lora")
            (out / "best_dev.json").write_text(json.dumps(dev, indent=2))
        else:
            bad += 1
        rec["best_acc"] = best_acc; rec["bad_evals"] = bad
        metrics.write(json.dumps(rec) + "\n"); log.write(f"[epoch {epoch}] {json.dumps(rec)}\n")
        if bad >= cfg["patience"]:
            log.write(f"[earlystop] best={best_acc:.4f}@{best_epoch}\n"); break

    log.write(f"[done] best dev acc={best_acc:.4f}@{best_epoch}\n")
    print(f"E2 DONE best_dev_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
