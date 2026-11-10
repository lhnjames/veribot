"""Batched S2: run SemVerifier on the FULL split, extract structured citations + full evidence text.
Left-padded batched generation on the A800. Output one JSON line per account:
{account_id, label, answer, cited_units, cited_full_text{unit_id:text}, n_content_cites}
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/path/to/project_root")
from verifiable_bot.v4.prompt_v4 import canonical_prompt_text
from verifiable_bot.v4.render_v4 import Unit, render_units

EVID_RE = re.compile(r"<evidence>(.*?)</evidence>", re.S | re.I)
ANS_RE = re.compile(r"<answer>\s*(bot|human)\s*</answer>", re.I)
UNIT_RE = re.compile(r"[A-Z]+_\d+")
CONTENT_PREFIX = ("TWEET", "HASHTAG", "URL", "MENTION")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/path/to/data_root/outputs/v4/artifacts/SemVerifier-0.6B-0724")
    ap.add_argument("--data", default="outputs/v4/render_v4/test.jsonl")
    ap.add_argument("--out", default="outputs/v5/s2_citations_test.jsonl")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda").eval()

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    from collections import Counter
    typ = Counter(); n_with_content = 0
    with open(args.out, "w") as fo:
        for i in range(0, len(rows), args.batch):
            batch = rows[i:i + args.batch]
            by_ids = [{u["unit_id"]: u["text"] for u in r["units"]} for r in batch]
            prompts = [canonical_prompt_text(render_units([Unit(u["unit_id"], u["text"]) for u in r["units"]]), tok)
                       for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=3600, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
            for r, by_id, g, inp in zip(batch, by_ids, gen, enc["input_ids"]):
                out = tok.decode(g[inp.shape[0]:], skip_special_tokens=True)
                ev = EVID_RE.search(out); an = ANS_RE.search(out)
                cited = UNIT_RE.findall(ev.group(1)) if ev else []
                cited = [c for c in cited if c in by_id]  # keep only real units
                content = [c for c in cited if c.startswith(CONTENT_PREFIX)]
                if content:
                    n_with_content += 1
                typ.update(re.match(r"[A-Z]+", c).group() for c in cited)
                fo.write(json.dumps({"account_id": r["account_id"], "label": r["label"],
                                     "answer": an.group(1).lower() if an else None,
                                     "cited_units": cited,
                                     "cited_full_text": {c: by_id[c] for c in cited},
                                     "n_content_cites": len(content)}, ensure_ascii=False) + "\n")
            print(f"  [{min(i+args.batch,len(rows))}/{len(rows)}] content_accts={n_with_content} types={dict(typ)}", flush=True)

    total = sum(typ.values()); content = sum(v for k, v in typ.items() if k in CONTENT_PREFIX)
    print(f"[S2 FULL] accounts={len(rows)} cite-type-dist={dict(typ)}")
    print(f"[S2 FULL] content-object citations = {content}/{total} = {content/max(total,1)*100:.2f}%")
    print(f"[S2 FULL] accounts citing >=1 content object = {n_with_content}/{len(rows)} = {n_with_content/len(rows)*100:.2f}%")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
