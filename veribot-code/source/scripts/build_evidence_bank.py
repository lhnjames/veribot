"""Offline evidence bank: for every TwiBot-20 account, ask deepseek-v4-pro to name the 2 most
important pieces of evidence supporting its KNOWN label, as citable unit ids + a closed-vocabulary
keyword chain. Runs once; training/eval/reward then read this by table lookup with zero API calls.

Inputs the model sees: the canonical visible units (PROFILE/OBS/ACTIVITY/RELATION/TWEET) + the
ground-truth label. It never sees unit_metadata, raw features, or the eval bank.

Robustness (all required, all here): sha256(account_id + prompt_version + units_version) cache key;
two-level hex raw_cache; atomic writes; resume (skip cached); per-account retry x3 with exponential
backoff; failed.jsonl; manifest + validation_report. train and eval banks are physically separate
files and this script refuses to mix them.

    python scripts/build_evidence_bank.py --split train --smoke 5      # endpoint/model/schema check
    python scripts/build_evidence_bank.py --split train --pilot        # fixed 100 bot + 100 human
    python scripts/build_evidence_bank.py --split train --workers 32   # full train
    python scripts/build_evidence_bank.py --split eval  --workers 32   # dev+test -> *_eval bank
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deepseek_client import DeepSeekClient  # noqa: E402

PROMPT_VERSION = "evbank_v1"
UNITS_VERSION = "units_v4"
DATA_DIR = Path("data/twibot20_complete")
RENDER_DIR = Path("outputs/v4/render_v4")
OUT_DIR = Path("outputs/v4/evidence_bank_v1")

KEYWORDS = [
    "follow imbalance", "low follower base", "no profile description", "default avatar",
    "account name entropy", "recent account age", "burst posting", "uniform posting interval",
    "duplicate text", "near-duplicate across accounts", "high retweet ratio", "no original content",
    "hashtag stuffing", "url spam", "mention flooding", "promotional template",
    "neighbor bot cluster", "low degree isolation", "cross-account co-posting", "topic monotony",
]
KEYWORD_SET = set(KEYWORDS)

SYSTEM = (
    "You are an expert social-media forensics annotator building a training dataset. Given the "
    "numbered evidence units of one Twitter/X account AND its confirmed ground-truth label, name "
    "the TWO most important pieces of evidence that support that label, most important first.\n"
    "Rules:\n"
    "- Output STRICT JSON only.\n"
    "- Provide exactly 2 items in core_evidence when possible (rank 1 = most important). If the "
    "account genuinely has only one solid piece of evidence, give 1 and set sufficient=false.\n"
    "- Every unit_ids entry MUST be an id that appears in the provided units (e.g. OBS_01). Never "
    "invent ids.\n"
    "- keyword MUST be chosen from this fixed list (verbatim): " + ", ".join(KEYWORDS) + ".\n"
    "- evidence_chain is the ordered list of the chosen keywords (length 1 or 2), matching "
    "core_evidence order.\n"
    "- statement must contain NO digits.\n"
    "JSON schema: {\"core_evidence\":[{\"rank\":int,\"unit_ids\":[str],\"keyword\":str,"
    "\"statement\":str}],\"evidence_chain\":[str],\"verifiability\":{\"sufficient\":bool}}"
)


def cache_key(account_id: str) -> str:
    return hashlib.sha256(f"{account_id}\x00{PROMPT_VERSION}\x00{UNITS_VERSION}".encode()).hexdigest()


def raw_cache_path(account_id: str) -> Path:
    key = cache_key(account_id)
    return OUT_DIR / "raw_cache" / key[:2] / key[2:4] / f"{key}.json"


def user_prompt(row: dict) -> str:
    unit_lines = "\n".join(f"[{u['unit_id']}] {u['text']}" for u in row["units"])
    return (f"Ground-truth label: {row['label']}\n\nEvidence units:\n{unit_lines}\n\n"
            "Return the JSON now.")


def valid_unit_ids(row: dict) -> set[str]:
    return {u["unit_id"] for u in row["units"]}


def validate(obj: dict, row: dict) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not an object"
    ce = obj.get("core_evidence")
    if not isinstance(ce, list) or not (1 <= len(ce) <= 2):
        return False, "core_evidence must have 1-2 items"
    valid_ids = valid_unit_ids(row)
    kws = []
    for item in ce:
        if not isinstance(item, dict):
            return False, "core_evidence item not object"
        if not isinstance(item.get("unit_ids"), list) or not item["unit_ids"]:
            return False, "unit_ids missing"
        for uid in item["unit_ids"]:
            if uid not in valid_ids:
                return False, f"unit id out of range: {uid}"
        kw = item.get("keyword")
        if kw not in KEYWORD_SET:
            return False, f"keyword not in vocab: {kw}"
        if re.search(r"\d", str(item.get("statement", ""))):
            return False, "statement contains a digit"
        kws.append(kw)
    chain = obj.get("evidence_chain")
    if chain != kws:
        return False, "evidence_chain must equal ordered core_evidence keywords"
    if not isinstance(obj.get("verifiability", {}).get("sufficient"), bool):
        return False, "verifiability.sufficient must be bool"
    return True, "ok"


def parse_json(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.startswith("json") else text
    return json.loads(text)


def process_account(row: dict, client: DeepSeekClient, max_retries: int = 3) -> dict:
    account_id = row["account_id"]
    cache = raw_cache_path(account_id)
    if cache.exists():
        blob = json.loads(cache.read_text(encoding="utf-8"))
        return {"account_id": account_id, "cached": True, **blob}
    last_err = ""
    for attempt in range(max_retries):
        try:
            # deepseek-v4-pro is a reasoning model: it spends a large token budget on chain-of-
            # thought before emitting the answer. max_tokens must cover reasoning + the JSON, or the
            # answer is truncated to empty content. 6000 leaves ample room for both.
            resp = client.chat(SYSTEM, user_prompt(row), temperature=0.2 + 0.2 * attempt,
                               max_tokens=6000)
            obj = parse_json(resp["content"])
            ok, reason = validate(obj, row)
            if not ok:
                last_err = f"schema: {reason}"
                continue
            record = {
                "account_id": account_id, "split": row.get("split_tag"), "label": row["label"],
                "core_evidence": obj["core_evidence"], "evidence_chain": obj["evidence_chain"],
                "verifiability": obj["verifiability"], "prompt_version": PROMPT_VERSION,
                "units_version": UNITS_VERSION, "usage": resp.get("usage", {}),
                "latency_s": resp.get("latency_s"),
            }
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            tmp.replace(cache)
            return {"account_id": account_id, "cached": False, **record}
        except Exception as exc:  # noqa: BLE001 - network/parse; retry with backoff
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
    return {"account_id": account_id, "failed": True, "error": last_err}


def load_rows(split: str) -> list[dict]:
    if split == "train":
        rows = [json.loads(l) for l in open(RENDER_DIR / "train.jsonl", encoding="utf-8")]
        for r in rows:
            r["split_tag"] = "train"
    elif split == "eval":
        rows = []
        for s in ("dev", "test"):
            for l in open(RENDER_DIR / f"{s}.jsonl", encoding="utf-8"):
                r = json.loads(l); r["split_tag"] = s; rows.append(r)
    else:
        raise ValueError("split must be 'train' or 'eval'")
    return rows


def pilot_subset(rows: list[dict]) -> list[dict]:
    """Fixed 100 bot + 100 human from TRAIN, sorted by account_id (reproducible)."""
    rows = sorted(rows, key=lambda r: r["account_id"])
    bots = [r for r in rows if r["label"] == "bot"][:100]
    humans = [r for r in rows if r["label"] == "human"][:100]
    return bots + humans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train", "eval"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--smoke", type=int, default=0, help="only process N accounts (endpoint check)")
    ap.add_argument("--pilot", action="store_true", help="fixed 100 bot + 100 human (train only)")
    ap.add_argument("--model", default="deepseek-v4-pro")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bank_name = "evidence_bank_train.jsonl" if args.split == "train" else "evidence_bank_eval.jsonl"
    out_bank = OUT_DIR / bank_name
    failed_path = OUT_DIR / f"failed_{args.split}.jsonl"

    rows = load_rows(args.split)
    if args.pilot:
        assert args.split == "train", "pilot is train-only"
        rows = pilot_subset(rows)
        out_bank = OUT_DIR / "evidence_bank_train_pilot.jsonl"
        failed_path = OUT_DIR / "failed_train_pilot.jsonl"
    elif args.smoke:
        rows = sorted(rows, key=lambda r: r["account_id"])[: args.smoke]
        out_bank = OUT_DIR / f"evidence_bank_{args.split}_smoke.jsonl"
        failed_path = OUT_DIR / f"failed_{args.split}_smoke.jsonl"

    client = DeepSeekClient(model=args.model)
    t0 = time.time()
    ok = fail = cached = suff = two = inrange = 0
    total_prompt_toks = total_completion_toks = 0
    results = []
    failures = []

    def run(row):
        return process_account(row, client)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run, r): r for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res.get("failed"):
                fail += 1
                failures.append({"account_id": res["account_id"], "error": res["error"]})
            else:
                ok += 1
                cached += 1 if res.get("cached") else 0
                results.append(res)
                if res.get("verifiability", {}).get("sufficient"):
                    suff += 1
                if len(res.get("core_evidence", [])) == 2:
                    two += 1
                inrange += 1
                u = res.get("usage", {})
                total_prompt_toks += u.get("prompt_tokens", 0)
                total_completion_toks += u.get("completion_tokens", 0)
            if i % 50 == 0 or i == len(rows):
                el = time.time() - t0
                eta = (len(rows) - i) / max(i / el, 1e-9)
                print(f"[{i}/{len(rows)}] ok={ok} fail={fail} cached={cached} "
                      f"suff={suff} 2ev={two} elapsed={el:.0f}s eta={eta:.0f}s", flush=True)

    # atomic bank write (sorted for determinism)
    results.sort(key=lambda r: r["account_id"])
    tmp = out_bank.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps({k: r[k] for k in
                     ("account_id", "split", "label", "core_evidence", "evidence_chain",
                      "verifiability", "prompt_version", "units_version") if k in r},
                     ensure_ascii=False) + "\n")
    tmp.replace(out_bank)
    if failures:
        with failed_path.open("w", encoding="utf-8") as fh:
            for f in failures:
                fh.write(json.dumps(f, ensure_ascii=False) + "\n")

    n = len(rows)
    report = {
        "split": args.split, "mode": "pilot" if args.pilot else ("smoke" if args.smoke else "full"),
        "n": n, "ok": ok, "failed": fail, "from_cache": cached,
        "schema_pass_rate": round(ok / n, 4) if n else 0,
        "exactly_two_rate": round(two / ok, 4) if ok else 0,
        "sufficient_rate": round(suff / ok, 4) if ok else 0,
        "bot_sufficient_rate": round(
            sum(1 for r in results if r["label"] == "bot" and r["verifiability"].get("sufficient"))
            / max(sum(1 for r in results if r["label"] == "bot"), 1), 4),
        "prompt_tokens": total_prompt_toks, "completion_tokens": total_completion_toks,
        "elapsed_s": round(time.time() - t0, 1),
        "prompt_version": PROMPT_VERSION, "units_version": UNITS_VERSION, "model": args.model,
        "bank": str(out_bank),
    }
    rep_name = f"validation_report_{report['mode']}_{args.split}.json"
    (OUT_DIR / rep_name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("REPORT:", json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
