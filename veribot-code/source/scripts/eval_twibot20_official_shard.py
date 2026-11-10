#!/usr/bin/env python3
"""Evaluate one deterministic shard of the official TwiBot-20 test.json."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_checkpoint import LABELS, classification_report, load_model
from verifiable_bot.data import as_bool, as_number, content_key, select_posts
from verifiable_bot.format_check import check_format
from verifiable_bot.prompting import inference_messages
from verifiable_bot.schema import AccountRecord, GraphEdge
from verifiable_bot.units import build_units, unit_map


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="adapter dir or merged full-weight dir")
    parser.add_argument("--base-model", default=None, help="required if --checkpoint is an adapter-only dir")
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--max-posts", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    args = parser.parse_args()

    users = json.loads(args.test_json.read_text(encoding="utf-8"))
    owners: dict[str, list[str]] = defaultdict(list)
    for user in users:
        uid = str(user.get("ID", "")).strip()
        for post in select_posts(user.get("tweet") or [], args.max_posts):
            key = content_key(post.text)
            if key and uid not in owners[key] and len(owners[key]) < 6:
                owners[key].append(uid)
    selected = [user for index, user in enumerate(users) if index % args.num_shards == args.shard_index]
    tokenizer, model = load_model(args.checkpoint, args.base_model)

    confusion = defaultdict(int)
    json_valid = format_valid = correct = 0
    status_counts = defaultdict(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", buffering=1) as output:
        for done, user in enumerate(selected, 1):
            record = make_record(user, owners, args.max_posts)
            valid_ids = unit_map(build_units(record))
            prompt = tokenizer.apply_chat_template(inference_messages(record), tokenize=False,
                                                   add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            raw = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            try:
                json.loads(raw.strip().strip("`"))
                json_valid += 1
            except json.JSONDecodeError:
                pass
            checked = check_format(raw, valid_ids)
            prediction = checked.output.prediction if checked.ok else None
            if checked.ok:
                format_valid += 1
                status_counts[checked.output.evidence_status] += 1
                confusion[(prediction, record.label)] += 1
                correct += int(prediction == record.label)
            output.write(json.dumps({"index": (done - 1) * args.num_shards + args.shard_index,
                                     "account_id": record.account_id, "true_label": record.label,
                                     "format_ok": checked.ok, "prediction": prediction, "raw": raw},
                                    ensure_ascii=False) + "\n")
            if done % 25 == 0:
                print(f"shard={args.shard_index} progress={done}/{len(selected)}", flush=True)

    report = classification_report(confusion, LABELS) if format_valid else None
    print(json.dumps({"shard": args.shard_index, "n": len(selected),
                      "json_valid": json_valid, "format_valid": format_valid, "correct": correct,
                      "status_counts": dict(status_counts), "report": report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
