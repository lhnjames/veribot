"""v4-correct held-out evaluation on TwiBot-20 dev (default) — ACC / Precision / Recall / F1 (bot as
positive), plus macro-F1 / balanced-accuracy (dev is imbalanced, bot-rate ~0.32) / confusion /
format-valid / pred-bot-rate. Generation + parse <answer>, identical to the trainer's dev eval.

Model stack: SFT best_lora is a LoRA on the base; E4/GRPO checkpoints are LoRA on the E2-MERGED base,
so pass --merge-lora <best_lora> to merge E2 first, then --checkpoint <grpo adapter>.

test stays SEALED — this evaluates dev only unless --data is explicitly pointed elsewhere.

  # SFT best_lora:
  python scripts/v5/eval_dev.py --base <qwen> --checkpoint outputs/v4/e2_sft/run_0.6b/best_lora --tag sft_best
  # E4 checkpoints (merge E2 first):
  python scripts/v5/eval_dev.py --base <qwen> --merge-lora outputs/v4/e2_sft/run_0.6b/best_lora \
      --checkpoint outputs/v4/grpo/run_0.6b/checkpoint-500,outputs/v4/grpo/run_0.6b/checkpoint-600
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verifiable_bot.v4.prompt_v4 import canonical_prompt_text  # noqa: E402
from verifiable_bot.v4.render_v4 import Unit, render_units  # noqa: E402

import os
BATCH = int(os.environ.get("EVAL_BATCH", 24))
LABEL_MAP = {"human": 0, "bot": 1}
ANSWER_RE = re.compile(r"<answer>\s*(bot|human)\s*</answer>", re.IGNORECASE)
FORMAT_RE = re.compile(r"<think>.*?</think>\s*<evidence>.*?</evidence>\s*<chain>.*?</chain>\s*"
                       r"<answer>\s*(bot|human)\s*</answer>", re.IGNORECASE | re.DOTALL)


def units_text(row):
    return render_units([Unit(u["unit_id"], u["text"]) for u in row["units"]])


def metrics(ys, ps, fmt_ok, n):
    ys, ps = np.asarray(ys), np.asarray(ps)
    tp = int(((ys == 1) & (ps == 1)).sum()); tn = int(((ys == 0) & (ps == 0)).sum())
    fp = int(((ys == 0) & (ps == 1)).sum()); fn = int(((ys == 1) & (ps == 0)).sum())
    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f
    p_bot, r_bot, f_bot = prf(tp, fp, fn)
    p_hum, r_hum, f_hum = prf(tn, fn, fp)
    acc = (tp + tn) / n if n else 0.0
    return {
        "acc": round(acc, 4),
        "precision_bot": round(p_bot, 4), "recall_bot": round(r_bot, 4), "f1_bot": round(f_bot, 4),
        "precision_human": round(p_hum, 4), "recall_human": round(r_hum, 4), "f1_human": round(f_hum, 4),
        "macro_precision": round((p_bot + p_hum) / 2, 4), "macro_recall": round((r_bot + r_hum) / 2, 4),
        "macro_f1": round((f_bot + f_hum) / 2, 4), "balanced_acc": round((r_bot + r_hum) / 2, 4),
        "pred_bot_rate": round(float(ps.mean()), 4), "format_valid_rate": round(fmt_ok / n, 4),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}, "n": n,
    }


@torch.no_grad()
def evaluate(model, tok, rows, max_new_tokens, max_length):
    model.eval(); tok.padding_side = "left"
    ys, ps, fmt_ok = [], [], 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        prompts = [canonical_prompt_text(units_text(r), tok) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, use_cache=True)
        for r, g, inp in zip(batch, gen, enc["input_ids"]):
            text = tok.decode(g[inp.shape[0]:], skip_special_tokens=True)
            if FORMAT_RE.search(text):
                fmt_ok += 1
            m = ANSWER_RE.search(text)
            pred = m.group(1).lower() if m else ("human" if r["label"] == "bot" else "bot")
            ps.append(LABEL_MAP[pred]); ys.append(LABEL_MAP[r["label"]])
    return metrics(ys, ps, fmt_ok, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--merge-lora", default=None, help="LoRA to merge into base first (E2 best_lora for GRPO ckpts)")
    ap.add_argument("--checkpoint", required=True, help="comma-separated adapter dirs to evaluate")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--data", default="outputs/v4/render_v4/dev.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--max-length", type=int, default=3600)
    ap.add_argument("--out", default="outputs/v4/reports/dev_eval.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    bot_rate = sum(r["label"] == "bot" for r in rows) / len(rows)
    print(f"[data] {args.data} n={len(rows)} bot_rate={bot_rate:.3f}")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base, trust_remote_code=True,
                                                torch_dtype=torch.bfloat16, device_map="cuda")
    if args.merge_lora:
        base = PeftModel.from_pretrained(base, args.merge_lora).merge_and_unload()
        print(f"[stack] merged E2 lora: {args.merge_lora}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    outfh = open(args.out, "a", buffering=1)
    for ckpt in args.checkpoint.split(","):
        ckpt = ckpt.strip()
        t0 = time.time()
        model = PeftModel.from_pretrained(base, ckpt)
        m = evaluate(model, tok, rows, args.max_new_tokens, args.max_length)
        model = model.unload() if hasattr(model, "unload") else base  # drop adapter, keep merged base
        tag = args.tag or Path(ckpt).name
        rec = {"tag": tag, "checkpoint": ckpt, "sec": round(time.time() - t0, 1), **m}
        outfh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{tag}] acc={m['acc']} F1_bot={m['f1_bot']} P_bot={m['precision_bot']} "
              f"R_bot={m['recall_bot']} macroF1={m['macro_f1']} balAcc={m['balanced_acc']} "
              f"fmt={m['format_valid_rate']} predBot={m['pred_bot_rate']} conf={m['confusion']} ({rec['sec']}s)")
    print("[done] ->", args.out)


if __name__ == "__main__":
    main()
