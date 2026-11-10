"""Build the final deployable model SemVerifier-0.6B-0724 = base Qwen3-0.6B + E2 best_lora + E4
GRPO checkpoint-550, all merged into self-contained full weights. Then sanity-eval on 200 dev
accounts to confirm the merge reproduces the adapter-stack accuracy (~0.81). ZERO test usage.
"""
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
from verifiable_bot.v4.prompt_v4 import canonical_prompt_text
from verifiable_bot.v4.render_v4 import Unit, render_units

BASE = "/path/to/data_root/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
BEST = "outputs/v4/e2_sft/run_0.6b/best_lora"
CKPT = "outputs/v4/grpo/run_0.6b/checkpoint-550"
OUT = "/path/to/data_root/outputs/v4/artifacts/SemVerifier-0.6B-0724"
LABEL = {"human": 0, "bot": 1}
ANS = re.compile(r"<answer>\s*(bot|human)\s*</answer>", re.I)


def main():
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("[merge] loading base + E2 best_lora ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True, torch_dtype=torch.bfloat16)
    m1 = PeftModel.from_pretrained(base, BEST).merge_and_unload()
    print("[merge] merging E4 GRPO checkpoint-550 ...", flush=True)
    m2 = PeftModel.from_pretrained(m1, CKPT).merge_and_unload()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    m2.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"[merge] saved full-weight model -> {OUT}", flush=True)

    # sanity: reload merged model fresh, eval 200 dev accounts
    del base, m1, m2
    torch.cuda.empty_cache()
    print("[verify] reloading merged SemVerifier for sanity eval ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(OUT, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval(); tok.padding_side = "left"
    rows = [json.loads(l) for l in open("outputs/v4/render_v4/dev.jsonl")][:200]
    ys, ps, fmt = [], [], 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), 24):
            b = rows[i:i + 24]
            prompts = [canonical_prompt_text(render_units([Unit(u["unit_id"], u["text"]) for u in r["units"]]), tok) for r in b]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=3600, add_special_tokens=False).to("cuda")
            gen = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
            for r, g, inp in zip(b, gen, enc["input_ids"]):
                txt = tok.decode(g[inp.shape[0]:], skip_special_tokens=True)
                if "<answer>" in txt and "</answer>" in txt:
                    fmt += 1
                mm = ANS.search(txt)
                pred = mm.group(1).lower() if mm else ("human" if r["label"] == "bot" else "bot")
                ps.append(LABEL[pred]); ys.append(LABEL[r["label"]])
    ys, ps = np.array(ys), np.array(ps)
    acc = float((ys == ps).mean())
    print(f"[verify] merged SemVerifier dev(200) acc={acc:.4f} format={fmt/len(rows):.3f} ({time.time()-t0:.0f}s)", flush=True)

    card = f"""# SemVerifier-0.6B-0724

Final deployable model for VeriRoute (TwiBot-20 bot detection).

- **Base**: Qwen3-0.6B (snapshot c1899de2)
- **Stack**: base + E2 SFT best_lora (epoch2) + E4 GRPO checkpoint-550 (triple-gate reward, lr=1e-5),
  all merged into these full weights (self-contained; no adapters needed at inference).
- **Selection**: best-by-dev among {{SFT best, E4 ckpt500/550/600}} on full dev-2365.

## Dev-2365 metrics (checkpoint-550, real bot-rate 0.551; test SEALED)
- ACC 0.8106 | bot P/R/F1 0.774/0.927/0.844 | macro-F1 0.8017 | balanced_acc 0.797 | format 0.997
- Beats E2 SFT by +1.7pp ACC / +2.3pp bot-F1 / +1.2pp macro-F1.

## Inference
Standard HF: `AutoModelForCausalLM.from_pretrained("SemVerifier-0.6B-0724")`; prompt via the v4
canonical formatter (`verifiable_bot.v4.prompt_v4.canonical_prompt_text`); parse `<answer>bot|human</answer>`.

Provenance: best_lora={BEST}, grpo_ckpt={CKPT}. Built 2026-07-24.
"""
    (Path(OUT) / "MODEL_CARD.md").write_text(card)
    print("[done] SemVerifier-0.6B-0724 built + card written.", flush=True)


if __name__ == "__main__":
    main()
