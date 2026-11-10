"""Build the v3 cold-start SFT set with real DeepSeek generation and K-five self-screening.

Every accepted bot row is format-valid, digit-free in generated prose, uses multiple evidence
categories when declared sufficient, and clears the same sufficiency judge used by GRPO. A failed
first draft is explicitly re-distilled once at higher temperature; only the second failure is
discarded. A deterministic share of graph-eligible bot accounts must cite a GRAPH unit so the
teacher cannot collapse back to PROFILE-only explanations while still allowing genuinely
irrelevant graph summaries to be skipped on the remaining accounts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiable_bot.format_check import check_format
from verifiable_bot.llm_client import DeepSeekJudge
from verifiable_bot.prompting import cold_start_teacher_prompt
from verifiable_bot.schema import AccountRecord
from verifiable_bot.units import build_units, hidden_unit_metadata, unit_map


class CachedTeacher:
    def __init__(self, cache_dir: Path, model: str, base_url: str, api_key: str):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.lock = threading.Lock()

    def complete(self, messages: list[dict], temperature: float, repeat: int) -> str:
        key = hashlib.sha256(
            (self.model + repr(temperature) + str(repeat) + json.dumps(messages, sort_keys=True)).encode()
        ).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))["content"]
            except (OSError, KeyError, json.JSONDecodeError):
                pass
        last_error = None
        for _ in range(2):
            try:
                result = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature, max_tokens=900,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = result.choices[0].message.content or ""
                tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, path)
                return content
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"teacher call failed twice: {last_error}")


def required_agreement(record: AccountRecord) -> str:
    return "agree" if record.upstream_verdict == record.label else "disagree"


def graph_required(record: AccountRecord, graph_ids: set[str]) -> bool:
    if record.label != "bot" or not graph_ids:
        return False
    # Fixed seventy-percent bot stratum, independent of generation order and worker scheduling.
    # With balanced bot/human cold start this targets >30% citations among all graph-eligible rows
    # without forcing every graph summary into evidence where it is not diagnostic.
    key = f"v3_graph_required:{record.source}:{record.account_id}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10 < 7


def evaluate_draft(record: AccountRecord, raw: str, judge: DeepSeekJudge) -> tuple[dict | None, dict]:
    units = build_units(record)
    texts = unit_map(units)
    metadata = hidden_unit_metadata(units)
    checked = check_format(raw, texts, metadata)
    if not checked.ok:
        return None, {"failure": "format", "reason": checked.reason}
    target = checked.output.model_dump()
    target["agreement"] = required_agreement(record)
    checked = check_format(json.dumps(target), texts, metadata)
    if not checked.ok:
        return None, {"failure": "post_normalization_format", "reason": checked.reason}
    graph_ids = {unit.unit_id for unit in units if unit.unit_id.startswith("GRAPH_")}
    cited = set(target["evidence_ids"])
    categories = {source.split("_", 1)[0] for source in cited}
    graph_is_required = graph_required(record, graph_ids)
    if record.label == "bot" and target["evidence_status"] != "sufficient":
        return None, {"failure": "bot_not_declared_sufficient"}
    if record.label == "bot" and target["evidence_status"] == "sufficient" and len(categories) < 2:
        return None, {"failure": "bot_single_category", "categories": sorted(categories)}
    if graph_is_required and not (cited & graph_ids):
        return None, {"failure": "required_graph_not_cited"}

    claims_text = "\n".join(f"- {claim['claim']}" for claim in target["claims"])
    sufficiency = judge.sufficiency_vote(claims_text, record.label)
    if record.label == "bot" and sufficiency < 0.6:
        return None, {"failure": "bot_sufficiency", "sufficiency": sufficiency}
    if record.label == "human":
        if target["evidence_status"] == "sufficient" and sufficiency < 0.4:
            return None, {"failure": "human_declared_sufficient_but_unsupported", "sufficiency": sufficiency}
        if target["evidence_status"] == "insufficient" and sufficiency >= 0.8:
            return None, {"failure": "human_dishonest_insufficient", "sufficiency": sufficiency}
    quality = {
        "sufficiency_k5": sufficiency,
        "graph_available": bool(graph_ids),
        "graph_required": graph_is_required,
        "graph_cited": bool(cited & graph_ids),
        "evidence_categories": sorted(categories),
        "redistilled": False,
    }
    return target, quality


def process_record(teacher: CachedTeacher, judge: DeepSeekJudge, record: AccountRecord) -> tuple[dict | None, dict]:
    messages = cold_start_teacher_prompt(record)
    failure_history = []
    for attempt, temperature in enumerate((0.2, 0.9)):
        attempt_messages = messages
        if failure_history:
            attempt_messages = messages + [{
                "role": "user",
                "content": (
                    "The previous draft failed screening for: "
                    + json.dumps(failure_history[-1], ensure_ascii=False)
                    + ". Redraft from scratch. For a bot label, use genuinely independent evidence "
                      "categories and cite a diagnostic GRAPH unit when available. Keep claim and "
                      "pattern prose digit-free and purely grounded. Return only the JSON object."
                ),
            }]
        try:
            raw = teacher.complete(attempt_messages, temperature=temperature, repeat=attempt)
            target, quality = evaluate_draft(record, raw, judge)
        except Exception as exc:
            target, quality = None, {"failure": "exception", "reason": repr(exc)}
        if target is not None:
            quality["redistilled"] = attempt > 0
            quality["attempts"] = attempt + 1
            return target, quality
        failure_history.append(quality)
    return None, {"failure": "redistillation_exhausted", "history": failure_history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-sidecar", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--judge-cache", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key", default=os.environ.get("JUDGE_API_KEY"))
    parser.add_argument("--sampling-strategy", choices=("balanced", "proportional"), default="balanced")
    parser.add_argument("--per-stratum", type=int, default=2500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.api_key:
        raise RuntimeError("JUDGE_API_KEY is required")
    teacher = CachedTeacher(args.cache, args.model, args.base_url, args.api_key)
    judge = DeepSeekJudge(
        cache_dir=args.judge_cache, model=args.model, base_url=args.base_url, api_key=args.api_key,
        k=5, temperature=0.7, max_workers=max(20, args.max_workers * 5),
    )
    for path in (args.output, args.hidden_sidecar, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                record = row["record"]
                done.add((record["source"], record["account_id"]))
    buckets = defaultdict(list)
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            record = AccountRecord.model_validate_json(line)
            if record.label and (record.source, record.account_id) not in done:
                buckets[(record.source, record.label)].append(record)
    rng = random.Random(args.seed)
    for group in buckets.values():
        rng.shuffle(group)
    if args.sampling_strategy == "balanced":
        records = [record for group in buckets.values() for record in group[:args.per_stratum]]
    else:
        records = [record for group in buckets.values() for record in group]
        rng.shuffle(records)
        if args.limit is not None:
            records = records[:args.limit]
    rng.shuffle(records)
    print("sampled", Counter((r.source, r.label) for r in records), flush=True)

    mode = "a" if args.resume else "w"
    accepted = Counter()
    failures = Counter()
    sufficiency_by_label = defaultdict(list)
    graph_available = graph_cited = redistilled = 0
    lock = threading.Lock()
    with args.output.open(mode, encoding="utf-8") as destination, args.hidden_sidecar.open(
        mode, encoding="utf-8"
    ) as hidden, ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(process_record, teacher, judge, record): record for record in records}
        completed = 0
        for future in as_completed(futures):
            record = futures[future]
            target, quality = future.result()
            completed += 1
            with lock:
                if target is None:
                    failures[quality.get("failure", "unknown")] += 1
                else:
                    destination.write(json.dumps({
                        "record": record.model_dump(), "target": target, "quality": quality,
                    }, ensure_ascii=False) + "\n")
                    units = build_units(record)
                    hidden.write(json.dumps({
                        "source": record.source, "account_id": record.account_id,
                        "bucket_version": record.bucket_version,
                        "units": hidden_unit_metadata(units),
                    }, ensure_ascii=False) + "\n")
                    accepted[(record.source, record.label)] += 1
                    sufficiency_by_label[record.label].append(quality["sufficiency_k5"])
                    graph_available += int(quality["graph_available"])
                    graph_cited += int(quality["graph_cited"])
                    redistilled += int(quality["redistilled"])
                if completed % 100 == 0:
                    destination.flush()
                    hidden.flush()
                    print(
                        f"progress {completed}/{len(records)} accepted={sum(accepted.values())} "
                        f"failures={sum(failures.values())}", flush=True,
                    )

    report = {
        "input": str(args.input), "output": str(args.output), "judge_model": args.model,
        "judge_k": 5, "judge_temperature": 0.7,
        "sampled": {f"{s}/{label}": len(group) for (s, label), group in sorted(buckets.items())},
        "accepted": {f"{s}/{label}": count for (s, label), count in sorted(accepted.items())},
        "failures": dict(failures), "redistilled_accepted": redistilled,
        "mean_sufficiency_by_label": {
            label: sum(values) / len(values) if values else 0.0
            for label, values in sufficiency_by_label.items()
        },
        "bot_sufficiency_at_least_three_of_five_rate": (
            sum(v >= 0.6 for v in sufficiency_by_label["bot"]) / len(sufficiency_by_label["bot"])
            if sufficiency_by_label["bot"] else 0.0
        ),
        "graph_available": graph_available, "graph_cited": graph_cited,
        "graph_citation_rate_of_available": graph_cited / graph_available if graph_available else 0.0,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
