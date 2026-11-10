"""Async HTTP/2 evidence-bank builder. Fixes the urllib TLS connection storm: httpx.AsyncClient
(http2=True) multiplexes thousands of concurrent request streams over a SMALL pool of persistent
connections, so there is no per-request TLS handshake / SYN flood. Everything else (SYSTEM prompt,
closed keyword vocab, schema validation, sha256 two-level hex cache, atomic writes, resume, retry)
is reused verbatim from build_evidence_bank.py.

Concurrency: a single asyncio.Semaphore caps in-flight requests. Run train at ~2000 and eval at
~500 (2500 total) as two processes, or one split at a time. Connection pool stays small
(max_connections≈64) — multiplexing does the rest.

The bank is assembled by scanning the per-account raw_cache for EVERY account in the split, so a
partial run simply fills more cache and a later run completes it; the report's `complete` flag is
true only when all accounts in the split have a cached success. sft_watch keys off that flag.

    python scripts/build_evidence_bank_async.py --split train --concurrency 2000
    python scripts/build_evidence_bank_async.py --split eval  --concurrency 500
    python scripts/build_evidence_bank_async.py --split train --concurrency 50 --limit 20   # smoke
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter
from contextlib import AsyncExitStack
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_evidence_bank import (  # noqa: E402
    OUT_DIR, PROMPT_VERSION, SYSTEM, UNITS_VERSION, load_rows, parse_json,
    raw_cache_path, user_prompt, validate,
)
from scripts.deepseek_client import load_api_key  # noqa: E402

ENDPOINT = "https://api.deepseek.com/chat/completions"


async def call_api(client, key, system, user, *, model, max_tokens=6000, temperature=0.2):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature, "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    r = await client.post(ENDPOINT, json=body, headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    d = r.json()
    return d["choices"][0]["message"]["content"], d.get("usage", {})


async def process_account(client, sem, key, row, model, stats, retries=4):
    aid = row["account_id"]
    cache = raw_cache_path(aid)
    if cache.exists():
        stats["cached"] += 1
        return "cached"
    async with sem:
        last = ""
        for attempt in range(retries):
            try:
                content, usage = await call_api(client, key, SYSTEM, user_prompt(row), model=model,
                                                temperature=0.2 + 0.15 * attempt)
                if not content.strip():
                    stats["empty_content"] += 1; last = "empty"; continue
                obj = parse_json(content)
                ok, reason = validate(obj, row)
                if not ok:
                    stats["schema_reject"] += 1; last = f"schema:{reason}"; continue
                record = {
                    "account_id": aid, "split": row.get("split_tag"), "label": row["label"],
                    "core_evidence": obj["core_evidence"], "evidence_chain": obj["evidence_chain"],
                    "verifiability": obj["verifiability"], "prompt_version": PROMPT_VERSION,
                    "units_version": UNITS_VERSION, "usage": usage,
                }
                cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_suffix(".tmp")
                tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                tmp.replace(cache)
                stats["ok_new"] += 1
                stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                stats["completion_tokens"] += usage.get("completion_tokens", 0)
                return "ok"
            except httpx.HTTPStatusError as e:
                code = e.response.status_code
                stats[f"http_{code}"] += 1; last = f"http_{code}"
                await asyncio.sleep(min(30, 2 ** attempt) + random.random())
            except httpx.TimeoutException:
                stats["timeout"] += 1; last = "timeout"
                await asyncio.sleep(min(30, 2 ** attempt) + random.random())
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError,
                    httpx.PoolTimeout, httpx.WriteError) as e:
                stats[f"conn_{type(e).__name__}"] += 1; last = type(e).__name__
                await asyncio.sleep(min(30, 2 ** attempt) + random.random())
            except Exception as e:  # noqa: BLE001
                stats[f"other_{type(e).__name__}"] += 1; last = f"{type(e).__name__}:{e}"
                await asyncio.sleep(min(30, 2 ** attempt) + random.random())
        stats["failed"] += 1
        stats["_fail_examples"].append({"account_id": aid, "last": last})
        return "failed"


def assemble_bank(split, rows):
    """Build the bank from cache for every account in the split; report completeness."""
    bank_name = "evidence_bank_train.jsonl" if split == "train" else "evidence_bank_eval.jsonl"
    out_bank = OUT_DIR / bank_name
    present, missing = [], 0
    for row in rows:
        p = raw_cache_path(row["account_id"])
        if p.exists():
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                present.append({k: rec[k] for k in
                                ("account_id", "split", "label", "core_evidence", "evidence_chain",
                                 "verifiability", "prompt_version", "units_version") if k in rec})
            except Exception:
                missing += 1
        else:
            missing += 1
    present.sort(key=lambda r: r["account_id"])
    tmp = out_bank.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in present:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out_bank)
    return out_bank, len(present), missing


async def main_async(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key = load_api_key()
    rows = load_rows(args.split)
    if args.limit:
        rows = sorted(rows, key=lambda r: r["account_id"])[: args.limit]
    stats = Counter()
    stats["_fail_examples"] = []  # type: ignore
    sem = asyncio.Semaphore(args.concurrency)
    # httpx intentionally reuses one HTTP/2 connection per client. DeepSeek advertises a bounded
    # number of concurrent streams per connection, so one client cannot consume the account's full
    # 2500-request allowance. Use several independent clients/connections and distribute accounts
    # across them. This preserves HTTP/2 multiplexing without recreating the urllib TLS/SYN storm.
    shards = args.client_shards or max(1, (args.concurrency + 99) // 100)
    per_client_connections = max(1, (args.max_connections + shards - 1) // shards)
    limits = httpx.Limits(max_connections=per_client_connections,
                          max_keepalive_connections=per_client_connections,
                          keepalive_expiry=120)
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    t0 = time.time()
    async with AsyncExitStack() as stack:
        clients = [
            await stack.enter_async_context(
                httpx.AsyncClient(http2=True, limits=limits, timeout=timeout)
            )
            for _ in range(shards)
        ]
        tasks = [
            asyncio.create_task(
                process_account(clients[i % shards], sem, key, r, args.model, stats)
            )
            for i, r in enumerate(rows)
        ]
        done = 0
        for fut in asyncio.as_completed(tasks):
            await fut
            done += 1
            if done % 100 == 0 or done == len(rows):
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                print(f"[{done}/{len(rows)}] ok_new={stats['ok_new']} cached={stats['cached']} "
                      f"failed={stats['failed']} rate={rate:.1f}/s el={el:.0f}s "
                      f"eta={(len(rows)-done)/max(rate,1e-9):.0f}s", flush=True)

    bank, present, missing = assemble_bank(args.split, load_rows(args.split))
    complete = (missing == 0)
    fail_types = {k: v for k, v in stats.items() if k not in
                  ("ok_new", "cached", "failed", "prompt_tokens", "completion_tokens", "_fail_examples")}
    report = {
        "split": args.split, "mode": "full" if not args.limit else "smoke",
        "n_accounts": len(load_rows(args.split)),
        "bank_lines": present, "missing": missing, "complete": complete,
        "ok_new": stats["ok_new"], "cached": stats["cached"], "failed": stats["failed"],
        "failure_types": fail_types,
        "prompt_tokens": stats["prompt_tokens"], "completion_tokens": stats["completion_tokens"],
        "elapsed_s": round(time.time() - t0, 1), "model": args.model,
        "concurrency": args.concurrency, "client_shards": shards, "bank": str(bank),
    }
    if not args.limit:
        (OUT_DIR / f"validation_report_full_{args.split}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if stats["_fail_examples"]:
        (OUT_DIR / f"failed_{args.split}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in stats["_fail_examples"][:500]),
            encoding="utf-8")
    print("REPORT:", json.dumps({k: v for k, v in report.items() if k != "failure_types"},
                                ensure_ascii=False))
    print("FAILURE_TYPES:", json.dumps(fail_types, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train", "eval"))
    ap.add_argument("--concurrency", type=int, default=2000)
    ap.add_argument("--max-connections", type=int, default=64)
    ap.add_argument("--client-shards", type=int, default=0,
                    help="independent HTTP/2 clients; 0 selects ceil(concurrency/100)")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="deepseek-v4-pro")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
