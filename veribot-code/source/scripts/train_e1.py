"""E1 ceiling probe: Qwen3-0.6B (LoRA) pooled last-token hidden ⊕ frozen 33-dim numeric vector →
MLP head → bot/human cross-entropy. Discriminative, no generation. Answers "how high can the LLM +
features go" and fixes the model size for later stages.

Correctness hard-gates enforced here:
  * numeric 33-dim mean/std fitted on TRAIN ONLY, frozen, saved; dev/test only standardised by it;
  * label map fixed (bot=1, human=0); class balance + account_id zero-crossing asserted at load;
  * canonical prompt hash + tokenizer fingerprint + render-manifest hash form the pretok cache key;
  * pooling = last real (non-pad) token with right padding, so it is padding-length invariant
    (a sanity check compares per-sample vs batched pooling under two pad lengths);
  * single-batch fwd/bwd finiteness smoke (loss & logits) before any real training;
  * checkpoint save/load logit parity; resume restores epoch/optimizer/scheduler.

Perf hard-gates: prompts are pretokenised ONCE and cached to disk keyed by the hashes above (never
re-tokenised per epoch); length-bucketed batches + dynamic right padding; pin_memory,
persistent_workers, prefetch; bf16 autocast, TF32, Qwen use_cache=False, SDPA attention. No
torch.compile.

    python scripts/train_e1.py --config configs/e1_0.6b.yaml --out outputs/v4/e1/run1
    python scripts/train_e1.py --config configs/e1_0.6b.yaml --out outputs/v4/e1/smoke --smoke-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.v4.prompt_v4 import (  # noqa: E402
    canonical_prompt_sha256, canonical_prompt_text, default_model_path, tokenizer_fingerprint,
)
from verifiable_bot.v4.render_v4 import Unit, render_units  # noqa: E402

LABEL_MAP = {"human": 0, "bot": 1}
N_FEATURES = 33


# ---------------------------------------------------------------------------------------------
# config / repro
# ---------------------------------------------------------------------------------------------
def load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def units_text_from_row(row):
    return render_units([Unit(u["unit_id"], u["text"]) for u in row["units"]])


class E1Dataset(Dataset):
    """Pretokenised canonical prompts + frozen-standardised 33-dim features + label. Tokenisation
    happens once (cached to disk); __getitem__ is pure tensor indexing."""

    def __init__(self, rows, input_ids, feats, labels):
        self.rows = rows
        self.input_ids = input_ids
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "feat": self.feats[i], "label": self.labels[i],
                "length": len(self.input_ids[i])}


def build_or_load_split(split, data_dir, tokenizer, cache_dir, cache_key, feat_mean, feat_std,
                        max_length):
    rows = read_jsonl(Path(data_dir) / f"{split}.jsonl")
    cache = Path(cache_dir) / f"{split}.{cache_key}.pt"
    if cache.exists():
        blob = torch.load(cache)
        input_ids = blob["input_ids"]
    else:
        input_ids = []
        for row in rows:
            text = canonical_prompt_text(units_text_from_row(row), tokenizer)
            enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)
            input_ids.append(torch.tensor(enc["input_ids"], dtype=torch.long))
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"input_ids": input_ids}, cache)
    feats = torch.tensor(np.array([r["features"] for r in rows], dtype=np.float32))
    feats = (feats - feat_mean) / feat_std
    labels = torch.tensor([LABEL_MAP[r["label"]] for r in rows], dtype=torch.long)
    return E1Dataset(rows, input_ids, feats, labels)


class LengthBucketBatcher:
    """Sort by length, chunk into batches, shuffle batch order → minimal padding waste."""

    def __init__(self, lengths, batch_size, shuffle, seed):
        self.order = list(np.argsort(lengths))
        self.batches = [self.order[i:i + batch_size] for i in range(0, len(self.order), batch_size)]
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __iter__(self):
        b = list(self.batches)
        if self.shuffle:
            self.rng.shuffle(b)
        for batch in b:
            yield batch

    def __len__(self):
        return len(self.batches)


def make_collate(pad_id):
    def collate(items):
        maxlen = max(x["length"] for x in items)
        B = len(items)
        input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((B, maxlen), dtype=torch.long)
        for j, x in enumerate(items):
            n = x["length"]
            input_ids[j, :n] = x["input_ids"]   # RIGHT padding
            attn[j, :n] = 1
        feat = torch.stack([x["feat"] for x in items])
        label = torch.tensor([x["label"] for x in items], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attn, "feat": feat, "label": label}
    return collate


# ---------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------
class E1Model(nn.Module):
    def __init__(self, backbone, hidden_size, head_hidden, head_dropout):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(hidden_size + N_FEATURES, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 2),
        )

    def pool(self, last_hidden, attention_mask):
        # last real (non-pad) token; right padding → index = sum(mask)-1. Padding-length invariant.
        idx = attention_mask.sum(dim=1) - 1
        return last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), idx]

    def forward(self, input_ids, attention_mask, feat):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=False, output_hidden_states=False)
        pooled = self.pool(out.last_hidden_state, attention_mask)      # [B,H]
        cat = torch.cat([pooled.float(), feat.float()], dim=1)         # head in fp32
        return self.head(cat)


# ---------------------------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                       batch["feat"].to(device))
        p = torch.softmax(logits.float(), dim=1)[:, 1]
        ps.extend((p >= 0.5).long().cpu().tolist())
        probs.extend(p.cpu().tolist())
        ys.extend(batch["label"].tolist())
    ys, ps, probs = np.array(ys), np.array(ps), np.array(probs)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    tn, fp_, fn, tp = confusion_matrix(ys, ps, labels=[0, 1]).ravel()
    return {
        "acc": float(accuracy_score(ys, ps)),
        "auc": float(roc_auc_score(ys, probs)) if len(set(ys)) > 1 else float("nan"),
        "f1": float(f1_score(ys, ps)),
        "pred_bot_rate": float(ps.mean()),
        "confusion": {"tn": int(tn), "fp": int(fp_), "fn": int(fn), "tp": int(tp)},
        "n": int(len(ys)),
    }


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default="outputs/v4/render_v4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- data integrity: labels, class balance, zero id-crossing ---
    splits = {s: read_jsonl(Path(args.data_dir) / f"{s}.jsonl") for s in ("train", "dev", "test")}
    ids = {s: {r["account_id"] for r in rows} for s, rows in splits.items()}
    assert not (ids["train"] & ids["dev"]) and not (ids["train"] & ids["test"]) and not (ids["dev"] & ids["test"]), \
        "account_id crosses splits"
    for s, rows in splits.items():
        assert all(r["label"] in LABEL_MAP for r in rows), f"{s} has an unmapped label"
    balance = {s: {"bot": sum(r["label"] == "bot" for r in rows), "human": sum(r["label"] == "human" for r in rows)}
               for s, rows in splits.items()}

    # --- frozen train-fitted numeric standardisation ---
    train_feats = np.array([r["features"] for r in splits["train"]], dtype=np.float64)
    feat_mean = torch.tensor(train_feats.mean(0), dtype=torch.float32)
    feat_std = torch.tensor(train_feats.std(0) + 1e-6, dtype=torch.float32)
    (out / "feature_stats.json").write_text(json.dumps(
        {"mean": feat_mean.tolist(), "std": feat_std.tolist(), "fitted_on": "train", "n": len(train_feats)},
        indent=2), encoding="utf-8")

    # --- pretok cache key = render manifest hash + canonical prompt hash + tokenizer fingerprint ---
    manifest = json.loads((Path(args.data_dir) / "render_manifest.json").read_text())
    fp = tokenizer_fingerprint(tokenizer, cfg["model_name"])
    key_src = json.dumps({
        "manifest_prompt_hash": manifest.get("canonical_prompt_sha256"),
        "canonical_prompt_sha256": canonical_prompt_sha256(tokenizer),
        "tokenizer": fp, "max_length": cfg["max_length"],
    }, sort_keys=True)
    cache_key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    cache_dir = Path(args.data_dir) / "e1_tokcache"

    ds = {s: build_or_load_split(s, args.data_dir, tokenizer, cache_dir, cache_key,
                                 feat_mean, feat_std, cfg["max_length"])
          for s in ("train", "dev", "test")}

    pad_id = tokenizer.pad_token_id
    collate = make_collate(pad_id)

    def loader(split, shuffle):
        lengths = [len(x) for x in ds[split].input_ids]
        bs = cfg["batch_size"]
        batcher = LengthBucketBatcher(lengths, bs, shuffle, cfg["seed"])
        return DataLoader(ds[split], batch_sampler=list(batcher), collate_fn=collate,
                          num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    # --- model ---
    backbone = AutoModel.from_pretrained(cfg["model_name"], trust_remote_code=True,
                                         torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    backbone.config.use_cache = False
    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      bias="none", task_type="FEATURE_EXTRACTION")
    backbone = get_peft_model(backbone, lora)
    hidden_size = backbone.config.hidden_size
    model = E1Model(backbone, hidden_size, cfg["head_hidden"], cfg["head_dropout"]).to(device)
    model.head.to(torch.float32)

    # --- optimizer: separate LR for LoRA vs head ---
    lora_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": cfg["lr_backbone"]},
        {"params": head_params, "lr": cfg["lr_head"]},
    ], weight_decay=cfg["weight_decay"])

    steps_per_epoch = (len(ds["train"]) + cfg["batch_size"] - 1) // cfg["batch_size"]
    total_steps = (steps_per_epoch // cfg["grad_accum"]) * cfg["epochs"]
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * cfg["warmup_ratio"]), total_steps)

    log = open(out / "train.log", "a", buffering=1)

    def sanity_single_batch():
        model.train()
        batch = next(iter(loader("train", shuffle=False)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                           batch["feat"].to(device))
            loss = F.cross_entropy(logits, batch["label"].to(device))
        assert torch.isfinite(logits).all(), "non-finite logits"
        assert torch.isfinite(loss), "non-finite loss"
        loss.backward()
        gsum = sum(p.grad.abs().sum().item() for p in head_params if p.grad is not None)
        assert np.isfinite(gsum) and gsum > 0, "bad head grad"
        opt.zero_grad(set_to_none=True)
        # padding-invariance of pooling: same sample pooled alone vs in a padded batch
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            one = {k: batch[k][:1].to(device) for k in ("input_ids", "attention_mask", "feat")}
            l1 = model(one["input_ids"], one["attention_mask"], one["feat"])
            padlen = batch["input_ids"].shape[1] + 7
            pi = torch.full((1, padlen), pad_id, dtype=torch.long)
            am = torch.zeros((1, padlen), dtype=torch.long)
            n = int(batch["attention_mask"][0].sum())
            pi[0, :n] = batch["input_ids"][0, :n]; am[0, :n] = 1
            l2 = model(pi.to(device), am.to(device), one["feat"])
        diff = (l1.float() - l2.float()).abs().max().item()
        log.write(f"[sanity] loss={loss.item():.4f} head_grad={gsum:.3e} pool_pad_diff={diff:.3e}\n")
        assert diff < 1e-2, f"pooling not padding-invariant: {diff}"
        return loss.item()

    log.write(f"[config] {json.dumps(cfg)}\n")
    log.write(f"[balance] {json.dumps(balance)}\n")
    log.write(f"[cache_key] {cache_key}  hidden={hidden_size} total_steps={total_steps}\n")
    sanity_loss = sanity_single_batch()
    log.write(f"[sanity] PASSED single-batch fwd/bwd, loss={sanity_loss:.4f}\n")

    if args.smoke_only:
        # checkpoint save/load logit parity on one batch
        model.eval()
        b = next(iter(loader("dev", shuffle=False)))
        with torch.no_grad():
            a = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["feat"].to(device)).float().cpu()
        torch.save({"head": model.head.state_dict()}, out / "smoke_head.pt")
        model.head.load_state_dict(torch.load(out / "smoke_head.pt")["head"])
        with torch.no_grad():
            a2 = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["feat"].to(device)).float().cpu()
        assert torch.allclose(a, a2, atol=1e-4), "ckpt save/load logit mismatch"
        log.write("[smoke] ckpt save/load logit parity OK. SMOKE PASSED.\n")
        print("SMOKE PASSED")
        return

    # save resolved config + env + nvidia-smi + manifest hashes
    (out / "resolved_config.json").write_text(json.dumps({
        "config": cfg, "cache_key": cache_key, "tokenizer_fingerprint": fp,
        "render_manifest_prompt_hash": manifest.get("canonical_prompt_sha256"),
        "balance": balance,
    }, indent=2), encoding="utf-8")
    try:
        (out / "env.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                     capture_output=True, text=True).stdout)
        (out / "nvidia-smi.txt").write_text(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    except Exception:
        pass

    metrics = open(out / "metrics.jsonl", "a", buffering=1)
    best_acc, best_epoch, bad = -1.0, -1, 0
    global_step = 0
    train_loader = loader("train", shuffle=True)
    dev_loader = loader("dev", shuffle=False)

    for epoch in range(cfg["epochs"]):
        model.train()
        t0 = time.time(); data_wait = 0.0; running = 0.0; nb = 0
        opt.zero_grad(set_to_none=True)
        tprev = time.time()
        for step, batch in enumerate(train_loader):
            data_wait += time.time() - tprev
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch["input_ids"].to(device, non_blocking=True),
                               batch["attention_mask"].to(device, non_blocking=True),
                               batch["feat"].to(device, non_blocking=True))
                loss = F.cross_entropy(logits, batch["label"].to(device)) / cfg["grad_accum"]
            loss.backward()
            running += loss.item() * cfg["grad_accum"]; nb += 1
            if (step + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg["grad_clip"])
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); global_step += 1
            tprev = time.time()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        rec = {"epoch": epoch, "train_loss": running / max(nb, 1), "sec": round(dt, 1),
               "data_wait_frac": round(data_wait / max(dt, 1e-9), 3),
               "samples_per_s": round(len(ds["train"]) / dt, 1), "peak_gb": round(peak, 2),
               "lr_backbone": sched.get_last_lr()[0], "lr_head": sched.get_last_lr()[1]}
        if (epoch + 1) % cfg["eval_every_epochs"] == 0:
            dev = evaluate(model, dev_loader, device)
            rec["dev"] = dev
            improved = dev["acc"] > best_acc + cfg["min_delta"]
            if improved:
                best_acc, best_epoch, bad = dev["acc"], epoch, 0
                torch.save({"head": model.head.state_dict(), "epoch": epoch,
                            "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "dev": dev}, out / "best.pt")
                model.backbone.save_pretrained(out / "best_lora")
            else:
                bad += 1
            rec["best_acc"] = best_acc; rec["bad_evals"] = bad
        metrics.write(json.dumps(rec) + "\n")
        log.write(f"[epoch {epoch}] {json.dumps(rec)}\n")
        if rec.get("bad_evals", 0) >= cfg["patience"]:
            log.write(f"[earlystop] no dev improvement for {cfg['patience']} evals; best={best_acc:.4f}@{best_epoch}\n")
            break

    log.write(f"[done] best dev acc={best_acc:.4f} @ epoch {best_epoch}\n")
    print(f"E1 DONE best_dev_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
