"""Generate -> critique -> revise loop for cold-start SFT samples.

The one-shot teacher (build_cold_start_sft.py) already writes format-valid samples, but "format
valid" only proves references are traceable -- it says nothing about whether a claim's wording is
actually a faithful, extractive reading of its cited evidence, or whether it silently smuggles in
an unsupported inference ("indicating bot behavior"). This script closes that gap in two LLM passes
per sample, using the ORIGINAL evidence and the ORIGINAL generation as context each time:

  1. CRITIQUE: an LLM reads the account's evidence units + the already-generated sample and lists
     every concrete problem (fabricated/out-of-context claims, unsupported inference language,
     dishonest evidence_status), or says NONE.
  2. REVISE: if any problems were found, a second LLM call gets the evidence + the original sample
     + the critique, and rewrites the sample to fix every listed problem, staying purely extractive.
  3. VALIDATE: the revised sample must still pass format_check.py AND the deterministic
     inference-leakage keyword filter (units.py's BANNED_WORDS-style check, reused from
     audit_cold_start_samples.py). If it still fails either check, the sample is dropped -- there is
     no unlimited retry loop, so nothing ungrounded is ever allowed through by exhaustion.

Samples that pass CRITIQUE with no problems found skip the revise call entirely (saves cost).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiable_bot.format_check import check_format
from verifiable_bot.prompting import OUTPUT_SCHEMA_HINT
from verifiable_bot.schema import AccountRecord
from verifiable_bot.units import build_units, render_units, unit_map

from audit_cold_start_samples import find_inference_leakage  # deterministic keyword pre-filter

CRITIQUE_SYSTEM = (
    "You are a strict, literal-minded auditor of training data for a bot-detection model. You only "
    "judge what is written in front of you; you never use outside knowledge about the account."
)

REVISE_SYSTEM = (
    "You rewrite flawed training samples for a bot-detection model so that every claim is a purely "
    "extractive, faithful reading of its cited evidence, with the label held fixed."
)


def critique_prompt(record: AccountRecord, target: dict) -> str:
    units_text = render_units(build_units(record))
    claims_text = "\n".join(f"{i+1}. (cites {', '.join(c['source_ids'])}) {c['claim']}"
                            for i, c in enumerate(target["claims"]))
    return (
        f"EVIDENCE_UNITS\n{units_text}\n\n"
        f"GENERATED SAMPLE\n"
        f'Given label: {target["prediction"]}\n'
        f'Cited evidence ids: {target["evidence_ids"]}\n'
        f"Claims:\n{claims_text}\n"
        f'Evidence status: {target["evidence_status"]}\n\n'
        "List every concrete problem with this sample. For each claim, check:\n"
        "(a) does it state ONLY what its cited evidence literally shows (no fabricated or "
        "out-of-context facts)?\n"
        "(b) does it avoid appending an unsupported inference linking the fact to bot/human-ness "
        '(e.g. "indicating X behavior", "typical of X", "consistent with X", "characteristic of X")? '
        "A claim may state a fact; it may not argue that the fact means something.\n"
        "Also check: is evidence_status honestly \"insufficient\" unless the cited evidence has "
        "multiple independent, specific, mutually-reinforcing signals (not just one generic fact "
        "like a single follower count or verified flag)?\n\n"
        "If you find no problems, respond with exactly: NONE\n"
        "Otherwise, list each problem on its own line, prefixed with which claim number it concerns "
        "(or EVIDENCE_STATUS for that issue). Be specific and concrete, not general commentary."
    )


def revise_prompt(record: AccountRecord, target: dict, critique: str) -> str:
    units_text = render_units(build_units(record))
    return (
        f"EVIDENCE_UNITS\n{units_text}\n\n"
        f"ORIGINAL SAMPLE (flawed)\n{json.dumps(target, ensure_ascii=False)}\n\n"
        f"PROBLEMS FOUND\n{critique}\n\n"
        f'Rewrite the sample to fix every listed problem. The label is fixed: set "prediction" to '
        f'exactly "{record.label}", unchanged. Every claim must be purely extractive: it states only '
        "what its cited evidence literally shows, with no inference about bot/human-ness appended to "
        "it. You may drop, merge, or rewrite claims, and you may change evidence_ids and "
        "evidence_status if that makes the sample honest -- most accounts should end up "
        '"insufficient" unless the evidence really has multiple independent, specific, '
        "mutually-reinforcing signals. Every evidence id you use must appear in EVIDENCE_UNITS above; "
        "never invent one.\n\n"
        "Respond with one JSON object shaped like:\n" + OUTPUT_SCHEMA_HINT
    )


class CachedClient:
    def __init__(self, cache_dir: Path, model: str, base_url: str, api_key: str, temperature: float = 0.2):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        key = hashlib.sha256(f"{self.model}|{self.temperature}|{system}|{user}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["content"]
        result = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = result.choices[0].message.content or ""
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"content": content}, ensure_ascii=False))
        tmp.replace(path)
        return content


def process_one(client: CachedClient, record: AccountRecord, target: dict, valid_ids: dict[str, str]) -> dict:
    critique = client.complete(CRITIQUE_SYSTEM, critique_prompt(record, target), max_tokens=400)
    if critique.strip().upper().startswith("NONE") and not find_inference_leakage(target):
        return {"account_id": record.account_id, "action": "kept_as_is", "target": target}

    revised_raw = client.complete(REVISE_SYSTEM, revise_prompt(record, target, critique), max_tokens=800)
    fc = check_format(revised_raw, valid_ids)
    if not fc.ok:
        return {"account_id": record.account_id, "action": "dropped", "reason": f"revision still invalid: {fc.reason}"}
    revised_target = fc.output.model_dump()
    revised_target["prediction"] = record.label  # defensive, same guarantee as build_cold_start_sft.py
    leakage = find_inference_leakage(revised_target)
    if leakage:
        return {"account_id": record.account_id, "action": "dropped", "reason": f"revision still leaks inference: {leakage}"}
    return {"account_id": record.account_id, "action": "revised", "target": revised_target, "critique": critique}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="cold-start jsonl to critique+revise")
    ap.add_argument("--output", type=Path, required=True, help="final grounded jsonl, same record/target shape")
    ap.add_argument("--log", type=Path, required=True, help="per-sample action log (kept_as_is/revised/dropped)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--max-workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    client = CachedClient(args.cache, args.model, args.base_url, args.api_key)

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            rows.append(json.loads(line))

    results = [None] * len(rows)
    write_lock = threading.Lock()
    counts = {"kept_as_is": 0, "revised": 0, "dropped": 0}

    def process(i: int) -> None:
        row = rows[i]
        record = AccountRecord.model_validate(row["record"])
        valid_ids = unit_map(build_units(record))
        outcome = process_one(client, record, row["target"], valid_ids)
        with write_lock:
            results[i] = outcome
            counts[outcome["action"]] += 1
            done = sum(counts.values())
            if done % 100 == 0:
                print(f"progress: {counts} / {len(rows)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        list(pool.map(process, range(len(rows))))

    with open(args.output, "w", encoding="utf-8") as out, open(args.log, "w", encoding="utf-8") as log:
        for row, outcome in zip(rows, results):
            log.write(json.dumps(outcome, ensure_ascii=False) + "\n")
            if outcome["action"] in ("kept_as_is", "revised"):
                out.write(json.dumps({"record": row["record"], "target": outcome["target"]}, ensure_ascii=False) + "\n")

    print(f"done: {counts} ({(counts['kept_as_is'] + counts['revised']) / len(rows):.1%} final pass rate)")


if __name__ == "__main__":
    main()
