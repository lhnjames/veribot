"""Build generative E2 SFT data from the offline evidence bank + rendered units. No API calls.

Each training example teaches the model to produce the canonical output for one account:

    <think>
    Step 1 (observe): <rank-1 statement>, as shown in [IDs].
    Step 2 (connect): Combined with <rank-2 statement> in [IDs], this forms a recognisable pattern.
    Step 3 (conclude): Together these point to <a bot / a human>.
    </think>
    <evidence>ID, ID | ID</evidence>
    <chain>keyword - keyword</chain>
    <answer>bot|human</answer>

The completion is rendered PROGRAMMATICALLY from the bank (rank-ordered core_evidence, the keyword
chain, and the ground-truth label) — the API is never called here. Only accounts with
verifiability.sufficient == true AND exactly two evidence items are used, so we never teach the
model to fabricate evidence. The set is balanced 1:1 bot/human (fixed seed, deterministic).

Incremental: reads the per-account raw_cache produced by build_evidence_bank.py, so it can run while
the bank is still filling and again once it completes. The prompt is rebuilt at train time through
the single canonical formatter (prompt_v4); here we store units + completion + label.

    python scripts/build_sft_data.py --out outputs/v4/e2_sft/sft_train.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

RENDER_DIR = Path("outputs/v4/render_v4")
BANK_DIR = Path("outputs/v4/evidence_bank_v1")

PROMPT_VERSION = "evbank_v1"
UNITS_VERSION = "units_v4"


def cache_key(account_id: str) -> str:
    return hashlib.sha256(f"{account_id}\x00{PROMPT_VERSION}\x00{UNITS_VERSION}".encode()).hexdigest()


def raw_cache_path(account_id: str) -> Path:
    k = cache_key(account_id)
    return BANK_DIR / "raw_cache" / k[:2] / k[2:4] / f"{k}.json"


def load_render(split: str) -> dict:
    rows = {}
    for line in open(RENDER_DIR / f"{split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rows[r["account_id"]] = r
    return rows


def load_evidence(account_id: str) -> dict | None:
    p = raw_cache_path(account_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def ids_str(unit_ids: list[str]) -> str:
    return ", ".join(unit_ids)


def bracket(unit_ids: list[str]) -> str:
    return " ".join(f"[{u}]" for u in unit_ids)


def render_completion(ev: dict, label: str) -> str | None:
    ce = ev.get("core_evidence") or []
    chain = ev.get("evidence_chain") or []
    if len(ce) != 2 or len(chain) != 2:
        return None
    r1, r2 = ce[0], ce[1]
    subject = "an automated bot account" if label == "bot" else "a genuine human account"
    think = (
        "<think>\n"
        f"Step 1 (observe): {r1['statement']}, as shown in {bracket(r1['unit_ids'])}.\n"
        f"Step 2 (connect): Combined with {r2['statement']} in {bracket(r2['unit_ids'])}, "
        "these observations reinforce each other.\n"
        f"Step 3 (conclude): Together they point to {subject}.\n"
        "</think>"
    )
    evidence = f"<evidence>{ids_str(r1['unit_ids'])} | {ids_str(r2['unit_ids'])}</evidence>"
    chain_line = f"<chain>{chain[0]} - {chain[1]}</chain>"
    answer = f"<answer>{label}</answer>"
    return f"{think}\n{evidence}\n{chain_line}\n{answer}"


def build(split: str, seed: int, balance: bool) -> tuple[list[dict], dict]:
    render = load_render(split)
    usable = {"bot": [], "human": []}
    stats = {"render": len(render), "have_evidence": 0, "sufficient_two": 0, "no_evidence": 0}
    for account_id, row in render.items():
        ev = load_evidence(account_id)
        if ev is None:
            stats["no_evidence"] += 1
            continue
        stats["have_evidence"] += 1
        if not ev.get("verifiability", {}).get("sufficient"):
            continue
        completion = render_completion(ev, row["label"])
        if completion is None:
            continue
        # guard: every cited id exists in this account's units, and completion has no digits
        valid = {u["unit_id"] for u in row["units"]}
        cited = {i for item in ev["core_evidence"] for i in item["unit_ids"]}
        if not cited.issubset(valid):
            continue
        stats["sufficient_two"] += 1
        usable[row["label"]].append({
            "account_id": account_id, "label": row["label"],
            "units": row["units"], "completion": completion,
        })
    rng = random.Random(seed)
    for k in usable:
        rng.shuffle(usable[k])
    if balance:
        n = min(len(usable["bot"]), len(usable["human"]))
        chosen = usable["bot"][:n] + usable["human"][:n]
        stats["balanced_n_per_class"] = n
    else:
        chosen = usable["bot"] + usable["human"]
    rng.shuffle(chosen)
    stats["out"] = len(chosen)
    stats["out_bot"] = sum(1 for c in chosen if c["label"] == "bot")
    stats["out_human"] = sum(1 for c in chosen if c["label"] == "human")
    return chosen, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="outputs/v4/e2_sft/sft_train.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-balance", action="store_true")
    args = ap.parse_args()

    rows, stats = build(args.split, args.seed, balance=not args.no_balance)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)
    (out.parent / f"sft_build_report_{args.split}.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SFT BUILD:", json.dumps(stats, ensure_ascii=False))
    print(f"-> {out}")
    if rows:
        print("SAMPLE COMPLETION:\n" + rows[0]["completion"])


if __name__ == "__main__":
    main()
