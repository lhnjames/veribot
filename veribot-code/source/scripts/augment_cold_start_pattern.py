"""Backfills a "pattern" field onto every claim in an already-critiqued cold-start SFT file.

This is NOT a re-generation of the cold-start samples -- evidence_ids, claim text, and
evidence_status are all kept exactly as already critiqued/revised (verifiable_bot/schema.py's
Claim.pattern is a new field this codebase didn't have before). For each claim, DeepSeek is shown
the account's FULL evidence-unit record (all information available for that account, not just the
cited units) plus the specific claim and its cited source text, and asked to name the general
behavioral/profile PATTERN that claim exemplifies -- never asserting or implying the bot/human
label itself (same anti-leakage boundary as PATTERN_BANNED_WORDS in prompting.py).

If a claim's pattern still leaks a label word after retries, the WHOLE ROW is dropped rather than
writing invalid training data (format_check.py now hard-rejects empty/leaking pattern fields, so
a cold-start row with a bad pattern could never pass the same check a real rollout has to pass).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiable_bot.prompting import PATTERN_BANNED_WORDS
from verifiable_bot.schema import AccountRecord
from verifiable_bot.units import build_units, render_units, unit_map

PATTERN_SYSTEM = (
    "You describe behavioral/profile PATTERNS in social-media account evidence. You never assert "
    "or imply whether an account is a bot or a human -- you only name the general shape of behavior "
    "a fact instantiates, as a short noun phrase. Never use the words bot, human, automated, "
    "automation, fake, spam, malicious, suspicious, inauthentic, or any other word that names a "
    "label or conclusion."
)


def _has_banned_word(text: str) -> list[str]:
    lowered = text.lower()
    return [w for w in PATTERN_BANNED_WORDS if re.search(rf"\b{re.escape(w)}\b", lowered)]


def pattern_prompt(units_text: str, claim_text: str, cited_texts: list[str]) -> str:
    cited = "\n".join(f"- {t}" for t in cited_texts)
    return (
        f"FULL ACCOUNT RECORD (context only -- do not introduce facts not already cited below)\n"
        f"{units_text}\n\n"
        f"CLAIM (already verified as an accurate, extractive reading of its cited evidence)\n"
        f"Cited evidence:\n{cited}\n"
        f"Claim: {claim_text}\n\n"
        "In one short phrase (roughly 4-10 words), name the general behavioral or profile PATTERN "
        "this claim is an instance of (e.g. \"rapid bulk posting\", \"long-standing organic "
        "engagement\", \"generic under-specified profile\", \"high following-to-follower ratio\"). "
        "Do not restate the claim itself verbatim, do not argue for a label, and do not use the "
        "words bot, human, automated, fake, spam, malicious, suspicious, inauthentic, or similar.\n\n"
        "Respond with the pattern phrase only, nothing else."
    )


class CachedClient:
    def __init__(self, cache_dir: Path, model: str, base_url: str, api_key: str):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, system: str, user: str, temperature: float, repeat: int = 0) -> str:
        key_src = f"{self.model}|{temperature}|{repeat}|{system}|{user}"
        key = hashlib.sha256(key_src.encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())["content"]
            except (json.JSONDecodeError, KeyError, OSError):
                pass
        result = self.client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=60,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = result.choices[0].message.content or ""
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"content": content}, ensure_ascii=False))
        tmp.replace(path)
        return content


def get_pattern(client: CachedClient, units_text: str, claim_text: str, cited_texts: list[str],
                retries: int) -> str | None:
    prompt = pattern_prompt(units_text, claim_text, cited_texts)
    for attempt in range(retries + 1):
        temperature = 0.2 if attempt == 0 else 0.8
        try:
            raw = client.complete(PATTERN_SYSTEM, prompt, temperature=temperature, repeat=attempt)
        except Exception:
            continue
        pattern = raw.strip().strip('"').strip("'").strip()
        # Reject anything that looks like it tried to answer in a full sentence/JSON rather than a
        # short phrase -- keeps completions short (token budget) and easier to validate.
        if pattern and len(pattern) <= 120 and not _has_banned_word(pattern):
            return pattern
    return None


def process_row(client: CachedClient, row: dict, retries: int) -> tuple[dict | None, str]:
    record = AccountRecord.model_validate(row["record"])
    units = build_units(record)
    units_text = render_units(units)
    umap = unit_map(units)

    new_claims = []
    for claim in row["target"]["claims"]:
        cited_texts = [umap[sid] for sid in claim["source_ids"] if sid in umap]
        pattern = get_pattern(client, units_text, claim["claim"], cited_texts, retries)
        if pattern is None:
            return None, f"dropped: claim could not get a clean pattern after retries: {claim['claim'][:80]!r}"
        new_claims.append({**claim, "pattern": pattern})

    new_target = {**row["target"], "claims": new_claims}
    return {"record": row["record"], "target": new_target}, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--max-workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    client = CachedClient(args.cache, args.model, args.base_url, args.api_key)

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            rows.append(json.loads(line))

    results: list[tuple[dict | None, str]] = [None] * len(rows)
    write_lock = threading.Lock()
    counts = {"ok": 0, "dropped": 0}

    def process(i: int) -> None:
        row = rows[i]
        try:
            outcome, reason = process_row(client, row, args.retries)
        except Exception as exc:
            outcome, reason = None, f"dropped: unexpected error: {exc!r}"
        with write_lock:
            results[i] = (outcome, reason)
            counts["ok" if outcome is not None else "dropped"] += 1
            done = sum(counts.values())
            if done % 200 == 0:
                print(f"progress: {counts} / {len(rows)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        list(pool.map(process, range(len(rows))))

    with open(args.output, "w", encoding="utf-8") as out, open(args.log, "w", encoding="utf-8") as log:
        for row, (outcome, reason) in zip(rows, results):
            account_id = row["record"]["account_id"]
            log.write(json.dumps({"account_id": account_id, "reason": reason}, ensure_ascii=False) + "\n")
            if outcome is not None:
                out.write(json.dumps(outcome, ensure_ascii=False) + "\n")

    print(f"done: {counts} ({counts['ok'] / len(rows):.1%} kept)")


if __name__ == "__main__":
    main()
