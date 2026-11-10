"""Sharded citation extraction: run ANY 0.6B (trained SemVerifier OR base Qwen3) on a shard of a split,
extract structured evidence citations + full text. Launch 8 shards in parallel for speed.
Output rows/account: {account_id,label,answer,cited_units,cited_full_text,n_content_cites}.
"""
import argparse, json, re, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, "/path/to/project_root")
from verifiable_bot.v4.prompt_v4 import canonical_prompt_text
from verifiable_bot.v4.render_v4 import Unit, render_units

EVID = re.compile(r"<evidence>(.*?)</evidence>", re.S | re.I)
ANS = re.compile(r"<answer>\s*(bot|human)\s*</answer>", re.I)
UNIT = re.compile(r"[A-Z]+_\d+")
CONTENT = ("TWEET", "HASHTAG", "URL", "MENTION")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="outputs/v4/render_v4/test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    rows = [json.loads(l) for l in open(args.data)]
    rows = rows[args.shard::args.nshards]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fo:
        for i in range(0, len(rows), args.batch):
            b = rows[i:i + args.batch]
            by = [{u["unit_id"]: u["text"] for u in r["units"]} for r in b]
            pr = [canonical_prompt_text(render_units([Unit(u["unit_id"], u["text"]) for u in r["units"]]), tok) for r in b]
            enc = tok(pr, return_tensors="pt", padding=True, truncation=True, max_length=3600, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
            for r, bid, gg, inp in zip(b, by, g, enc["input_ids"]):
                out = tok.decode(gg[inp.shape[0]:], skip_special_tokens=True)
                ev = EVID.search(out); an = ANS.search(out)
                cited = [c for c in (UNIT.findall(ev.group(1)) if ev else []) if c in bid]
                fo.write(json.dumps({"account_id": r["account_id"], "label": r["label"],
                                     "answer": an.group(1).lower() if an else None,
                                     "cited_units": cited, "cited_full_text": {c: bid[c] for c in cited},
                                     "n_content_cites": sum(c.startswith(CONTENT) for c in cited),
                                     "has_evidence_tag": ev is not None}, ensure_ascii=False) + "\n")
    print(f"shard {args.shard}/{args.nshards} done: {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
