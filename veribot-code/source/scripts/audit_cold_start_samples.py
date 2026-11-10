"""Audits cold-start SFT samples for reasonableness (not format validity -- format_check.py
already guarantees that; this checks whether the generated claims are a sane, grounded reading of
the cited evidence, i.e. the "语义质量" that build_cold_start_sft.py deliberately does NOT vet).

For each {"record": ..., "target": ...} line:
  - Ask an LLM judge a single yes/no question, independently, 5 times (K=5, matching the
    training/screening K used everywhere else -- see configs/veriroute_lite.yaml judge_k and
    llm_client.DeepSeekJudge's default).
  - Keep the sample only if a strict majority of votes are YES (>= 3 of 5 at K=5).
Writes:
  --output       : the input file filtered down to majority-YES samples (drop-in replacement for
                   the un-audited file, same record/target JSON shape)
  --audit-log    : one line per input sample: {account_id, votes: [bool, ...] (length K), decision}

This is a read-only judge pass -- it never calls the policy model and never produces gradients,
same posture as llm_client.py's reward-time judge calls.
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

from verifiable_bot.schema import AccountRecord
from verifiable_bot.units import build_units, render_units

AUDIT_SYSTEM = (
    "You are a strict, literal-minded auditor of training data for a bot-detection model. You only "
    "judge what is written in front of you; you never use outside knowledge about the account. "
    "Always answer with exactly one word: YES or NO. No other text."
)

# Cheap, deterministic pre-filter: catches the most common failure mode found by manual review --
# a teacher that already knows the target label writes a bare fact, then bolts an unsupported
# inference onto it ("The account is not verified, indicating bot behavior."). "not verified" is a
# checkable fact; "indicating bot behavior" is not something any evidence unit states, it's the
# teacher's own conclusion smuggled into the claim text. This never depends on which label was
# given, so it's checked before spending any API call.
INFERENCE_LEAKAGE_PHRASES = (
    "indicating", "indicative of", "suggests", "suggesting", "consistent with bot",
    "consistent with human", "typical of bot", "typical of human", "common for bot",
    "common for human", "characteristic of", "sign of a bot", "sign of a human",
    "evidence of being a bot", "evidence of being a human", "likely a bot", "likely a human",
    "which is often associated with", "pointing to bot", "pointing to human",
)


def find_inference_leakage(target: dict) -> list[str]:
    hits = []
    for claim in target["claims"]:
        lowered = claim["claim"].lower()
        for phrase in INFERENCE_LEAKAGE_PHRASES:
            if phrase in lowered:
                hits.append(f'{phrase!r} in claim: {claim["claim"]!r}')
    return hits


def render_audit_prompt(record: AccountRecord, target: dict) -> str:
    units_text = render_units(build_units(record))
    claims_text = "\n".join(f"- (cites {', '.join(c['source_ids'])}) {c['claim']}" for c in target["claims"])
    return (
        f"EVIDENCE_UNITS\n{units_text}\n\n"
        f"GENERATED TRAINING SAMPLE\n"
        f'Given label: {target["prediction"]}\n'
        f'Cited evidence ids: {target["evidence_ids"]}\n'
        f"Claims:\n{claims_text}\n"
        f'Evidence status: {target["evidence_status"]}\n\n'
        "This sample was written by a teacher model that was TOLD the label in advance and asked to "
        "justify it -- so it is prone to writing a plausible-sounding rationalization rather than a "
        "faithful reading of the evidence. Answer YES only if ALL of the following hold:\n"
        "1) EXTRACTIVE, not inferential: each claim states only what its cited evidence literally "
        "shows (a fact), and does NOT append an unsupported inference tying that fact to bot/human-ness "
        '(e.g. "not verified" is fine; "not verified, indicating bot behavior" is NOT, because no '
        "evidence unit says non-verification indicates anything -- that link is the teacher's own "
        "opinion, not a cited fact);\n"
        "2) no fabricated or out-of-context numbers/facts;\n"
        '3) if evidence_status is "sufficient", the cited evidence is not just one weak, generically '
        "reusable signal (a single follower count, a single verified flag, a single tweet) that could "
        "equally have been cited to argue the opposite label for a different account -- there should be "
        "multiple, mutually reinforcing, account-specific observations;\n"
        "4) if the account's evidence units are sparse or generic, evidence_status should be "
        '"insufficient", not "sufficient".\n'
        "Otherwise answer NO. Answer with exactly one word: YES or NO."
    )


class YesNoJudge:
    def __init__(self, cache_dir: Path, model: str, base_url: str, api_key: str, k: int = 5,
                 temperature: float = 0.7, max_workers: int = 16):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.k = k
        self.temperature = temperature
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def _call_once(self, user: str, repeat: int) -> str:
        key = hashlib.sha256(f"{self.model}|{self.temperature}|{repeat}|{AUDIT_SYSTEM}|{user}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["content"]
        result = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=20,
            messages=[{"role": "system", "content": AUDIT_SYSTEM}, {"role": "user", "content": user}],
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = result.choices[0].message.content or ""
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"content": content}, ensure_ascii=False))
        tmp.replace(path)
        return content

    def votes(self, user: str) -> list[bool]:
        futures = [self._pool.submit(self._call_once, user, r) for r in range(self.k)]
        return [f.result().strip().upper().startswith("YES") for f in futures]


def audit_one(judge: YesNoJudge, record: AccountRecord, target: dict) -> dict:
    leakage = find_inference_leakage(target)
    if leakage:
        return {"account_id": record.account_id, "votes": [], "decision": "reject",
                "reason": "inference_leakage", "detail": leakage}
    prompt = render_audit_prompt(record, target)
    votes = judge.votes(prompt)
    decision = "keep" if sum(votes) >= (len(votes) // 2 + 1) else "reject"  # majority, K-agnostic
    return {"account_id": record.account_id, "votes": votes, "decision": decision}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="cold-start SFT jsonl to audit")
    ap.add_argument("--output", type=Path, required=True, help="filtered jsonl, majority-YES only")
    ap.add_argument("--audit-log", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-workers", type=int, default=16, help="samples audited concurrently "
                    "(each sample itself fires k=5 concurrent calls, so real concurrency is higher)")
    ap.add_argument("--limit", type=int, default=None, help="audit only the first N samples (for a quick check)")
    args = ap.parse_args()

    judge = YesNoJudge(args.cache, args.model, args.base_url, args.api_key, k=args.k)

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            rows.append(json.loads(line))

    results = [None] * len(rows)
    write_lock = threading.Lock()
    kept = rejected = 0

    def process(i: int) -> None:
        nonlocal kept, rejected
        row = rows[i]
        record = AccountRecord.model_validate(row["record"])
        outcome = audit_one(judge, record, row["target"])
        with write_lock:
            results[i] = outcome
            if outcome["decision"] == "keep":
                kept += 1
            else:
                rejected += 1
            if (kept + rejected) % 200 == 0:
                print(f"progress: kept={kept} rejected={rejected} / {len(rows)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        list(pool.map(process, range(len(rows))))

    with open(args.output, "w", encoding="utf-8") as out, open(args.audit_log, "w", encoding="utf-8") as log:
        for row, outcome in zip(rows, results):
            log.write(json.dumps(outcome, ensure_ascii=False) + "\n")
            if outcome["decision"] == "keep":
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"done: kept={kept} rejected={rejected} ({kept / len(rows):.1%} pass rate)")


if __name__ == "__main__":
    main()
