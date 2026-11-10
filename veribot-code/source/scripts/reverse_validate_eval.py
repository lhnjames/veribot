"""Reverse-validation audit of the just-completed official TwiBot-20 eval outputs.

For a stratified sample of accounts (correct-bot / correct-human / wrong-as-bot / wrong-as-human),
re-run the SAME judge-based checks used during GRPO reward computation (claim faithfulness,
sufficiency, necessity via counterfactual masking, cross-evidence consistency) against the model's
ALREADY-GENERATED evaluation completion -- but WITHOUT the reward module's early-return on wrong
label, so we can see whether even incorrect predictions were backed by internally coherent
(sufficient+necessary) evidence, or were essentially unsupported guesses.

This does not call the model again -- it only re-scores the raw completions already saved by
eval_twibot20_official_shard.py, using the DeepSeek judge as an independent auditor.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifiable_bot.data import as_bool, as_number, content_key, select_posts
from verifiable_bot.format_check import check_format
from verifiable_bot.llm_client import DeepSeekJudge
from verifiable_bot.schema import AccountRecord, GraphEdge
from verifiable_bot.units import build_units, render_masked, render_units, unit_map

LAMBDA_MIN = 0.5


def make_record(user: dict, owners: dict[str, list[str]], max_posts: int) -> AccountRecord:
    uid = str(user.get("ID", "")).strip()
    raw = user.get("profile") or {}
    profile = {
        "description": str(raw.get("description", "")).strip(),
        "followers_count": as_number(raw.get("followers_count", 0)),
        "following_count": as_number(raw.get("friends_count", 0)),
        "tweet_count": as_number(raw.get("statuses_count", 0)),
        "listed_count": as_number(raw.get("listed_count", 0)),
        "verified": as_bool(raw.get("verified", False)),
    }
    edges = []
    for relation, values in (user.get("neighbor") or {}).items():
        if not isinstance(values, list):
            continue
        for target in values[:24]:
            edges.append(GraphEdge(edge_id=f"e{len(edges)}", source="self",
                                   target=f"n{len(edges)}", relation=str(relation)))
    selected = select_posts(user.get("tweet") or [], max_posts)
    facts = [
        f"post:{post.post_id} has the same normalized original content as {len(owners[key]) - 1} other test accounts"
        for post in selected
        if (key := content_key(post.text)) and len(owners.get(key, [])) > 1
    ]
    label = "bot" if str(user.get("label", "")).lower() in {"1", "bot"} else "human"
    return AccountRecord(account_id=uid, source="twibot20", split="test", profile=profile,
                         posts=selected, graph_edges=edges, graph_facts=facts, label=label)


def audit_one(record: AccountRecord, raw_text: str, judge: DeepSeekJudge) -> dict:
    """Same checks as rewards.py::score_completion, but WITHOUT the wrong-label early return,
    so we get sufficiency/necessity/consistency even for incorrect predictions."""
    units = build_units(record)
    units_map = unit_map(units)
    valid_ids = units_map

    fc = check_format(raw_text, valid_ids)
    if not fc.ok:
        return {"format_ok": False, "reason": fc.reason}

    output = fc.output
    result: dict = {
        "format_ok": True,
        "prediction": output.prediction,
        "true_label": record.label,
        "correct": output.prediction == record.label,
        "evidence_status": output.evidence_status,
        "n_claims": len(output.claims),
        "n_evidence_ids": len(output.evidence_ids),
        "cites_non_profile": any(not sid.startswith("PROFILE_") for sid in output.evidence_ids),
    }

    if output.evidence_status != "sufficient":
        result["tier"] = "declared_insufficient"
        return result

    claims_text = "\n".join(f"- {c.claim}" for c in output.claims)
    s_suf = judge.sufficiency_vote(claims_text, output.prediction)
    result["s_suf"] = s_suf

    s_consistency = 1.0
    if len(output.claims) >= 2:
        s_consistency = judge.consistency_vote(claims_text)
    result["s_consistency"] = s_consistency

    full_units_text = render_units(units)
    masked_full_text = render_masked(units, set(output.evidence_ids))

    with ThreadPoolExecutor(max_workers=len(output.claims) + 2) as pool:
        claim_futures = [
            pool.submit(judge.claim_approval, c.claim,
                       [units_map[sid] for sid in c.source_ids if sid in units_map], c.pattern)
            for c in output.claims
        ]
        full_future = pool.submit(judge.label_approve_rate, full_units_text, output.prediction)
        masked_future = pool.submit(judge.label_approve_rate, masked_full_text, output.prediction)
        claim_scores = [f.result() for f in claim_futures]
        approve_full = full_future.result()
        approve_masked = masked_future.result()

    q = LAMBDA_MIN * min(claim_scores) + (1 - LAMBDA_MIN) * (sum(claim_scores) / len(claim_scores)) if claim_scores else 0.0
    s_nec = max(0.0, min(1.0, approve_full - approve_masked))

    result["q_claims"] = q
    result["s_nec"] = s_nec
    result["approve_rate_full"] = approve_full
    result["approve_rate_masked"] = approve_masked

    if s_suf < 2 / 3:
        result["tier"] = "unsupported"
    elif len(output.claims) >= 2 and s_consistency < 2 / 3:
        result["tier"] = "inconsistent"
    elif s_nec < 1 / 3:
        result["tier"] = "unnecessary"
    else:
        result["tier"] = "genuinely_grounded"
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-json", type=Path, required=True)
    ap.add_argument("--shard-glob", required=True)
    ap.add_argument("--judge-cache", type=Path, required=True)
    ap.add_argument("--per-group", type=int, default=40)
    ap.add_argument("--max-posts", type=int, default=32)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    users = json.loads(args.test_json.read_text(encoding="utf-8"))
    by_id = {str(u.get("ID", "")).strip(): u for u in users}
    owners: dict[str, list[str]] = defaultdict(list)
    for user in users:
        uid = str(user.get("ID", "")).strip()
        for post in select_posts(user.get("tweet") or [], args.max_posts):
            key = content_key(post.text)
            if key and uid not in owners[key] and len(owners[key]) < 6:
                owners[key].append(uid)

    rows = []
    for f in sorted(glob.glob(args.shard_glob)):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    by_index = {r["index"]: r for r in rows}
    rows = list(by_index.values())

    groups: dict[str, list[dict]] = {
        "TP_bot": [], "TP_human": [], "FN_as_human": [], "FP_as_bot": [],
    }
    for r in rows:
        if not r["format_ok"]:
            continue
        tl, pred = r["true_label"], r["prediction"]
        if tl == "bot" and pred == "bot":
            groups["TP_bot"].append(r)
        elif tl == "human" and pred == "human":
            groups["TP_human"].append(r)
        elif tl == "bot" and pred == "human":
            groups["FN_as_human"].append(r)
        elif tl == "human" and pred == "bot":
            groups["FP_as_bot"].append(r)

    rng = random.Random(args.seed)
    sample = []
    for g, items in groups.items():
        rng.shuffle(items)
        chosen = items[: args.per_group]
        print(f"group {g}: pool={len(items)} sampled={len(chosen)}", flush=True)
        for item in chosen:
            sample.append((g, item))

    judge = DeepSeekJudge(cache_dir=args.judge_cache, k=5, max_workers=32)

    def process(entry):
        group, row = entry
        user = by_id.get(row["account_id"])
        if user is None:
            return None
        record = make_record(user, owners, args.max_posts)
        audit = audit_one(record, row["raw"], judge)
        audit["group"] = group
        audit["account_id"] = row["account_id"]
        return audit

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, res in enumerate(pool.map(process, sample), 1):
            if res is not None:
                results.append(res)
            if i % 20 == 0:
                print(f"audited {i}/{len(sample)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== SUMMARY BY GROUP ===")
    by_group = defaultdict(list)
    for r in results:
        by_group[r["group"]].append(r)
    for g, items in by_group.items():
        n = len(items)
        if n == 0:
            continue
        tier_counts = defaultdict(int)
        for r in items:
            tier_counts[r.get("tier", "format_invalid")] += 1
        avg = lambda key: sum(r.get(key, 0) or 0 for r in items) / n
        print(f"\n-- {g} (n={n}) --")
        print(f"  tier counts: {dict(tier_counts)}")
        print(f"  mean s_suf={avg('s_suf'):.3f} s_nec={avg('s_nec'):.3f} "
              f"s_consistency={avg('s_consistency'):.3f} q_claims={avg('q_claims'):.3f}")
        cites_non_profile = sum(1 for r in items if r.get("cites_non_profile")) / n
        print(f"  cites_non_profile_evidence_rate={cites_non_profile:.3f}")


if __name__ == "__main__":
    main()
