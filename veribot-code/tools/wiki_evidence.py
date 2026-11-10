"""Source-linked account/passage/entity graph inspired by WFM's Wiki schema.

This is a deterministic audit index. It does not train WFM, infer bot labels,
or alter the frozen HGT/RelNet prediction. A retrieval hit is not evidence
that a prediction is correct or faithful to the detector.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FOX_PATH = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson"
BOT_USERS = ROOT / "data/raw/botsim_24/Users.csv"
BOT_POSTS = ROOT / "data/raw/botsim_24/user_post_comment.json"
TOKEN = re.compile(r"(?u)\b\w+\b")
HASHTAG = re.compile(r"(?<!\w)#([\w_]+)")
MENTION = re.compile(r"(?<!\w)@([\w_]+)")
URL = re.compile(r"https?://[^\s]+")


def stable_id(prefix: str, value: str) -> str:
    return prefix + ":" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def link_entity(graph: dict, source_id: str, entity_type: str, value: str,
                relation: str) -> None:
    value = str(value or "").strip()
    if not value:
        return
    entity_id = stable_id(entity_type, value.lower())
    if all(node["id"] != entity_id for node in graph["nodes"]):
        graph["nodes"].append({"id": entity_id, "type": "entity",
                               "kind": entity_type, "value": value})
    edge = {"from": source_id, "to": entity_id, "type": relation}
    if edge not in graph["edges"]:
        graph["edges"].append(edge)


def add_passage(graph: dict, account_id: str, text: str, source_id: str,
                field: str, kind: str, extra: dict | None = None) -> None:
    text = str(text or "").strip()
    if not text:
        return
    passage_id = stable_id("passage", account_id + "|" + kind + "|" + source_id)
    graph["nodes"].append({"id": passage_id, "type": "passage", "kind": kind,
                           "text": text, "source_id": source_id, "source_field": field,
                           "source_dataset": graph["dataset"], **(extra or {})})
    graph["edges"].append({"from": "account:" + account_id, "to": passage_id,
                           "type": "has_passage"})
    entities = [("hashtag", v.lower()) for v in HASHTAG.findall(text)]
    entities += [("mention", v.lower()) for v in MENTION.findall(text)]
    entities += [("url", v.rstrip(".,;:!?)")) for v in URL.findall(text)]
    for entity_type, value in sorted(set(entities)):
        link_entity(graph, passage_id, entity_type, value, "mentions")


def fox8_graph(record: dict) -> dict:
    account_id = str(record["user_id"])
    graph = {"dataset": "fox8_23", "account_id": account_id,
             "nodes": [{"id": "account:" + account_id, "type": "account",
                        "source_id": account_id, "source_dataset": "fox8_23"}],
             "edges": []}
    tweets = record.get("user_tweets") or []
    profile = next((t.get("user") for t in tweets if isinstance(t, dict) and t.get("user")), {})
    add_passage(graph, account_id, profile.get("description", ""),
                account_id, "user.description", "profile_description")
    for i, tweet in enumerate(tweets):
        if not isinstance(tweet, dict):
            continue
        tweet_id = str(tweet.get("id_str") or tweet.get("id") or f"ordinal:{i}")
        add_passage(graph, account_id, tweet.get("full_text") or tweet.get("text") or "",
                    tweet_id, "user_tweets[].text", "tweet",
                    {"ordinal": i})
    return graph


def botsim_graph(user: dict, activities: dict) -> dict:
    account_id = str(user["user_id"])
    graph = {"dataset": "botsim_24", "account_id": account_id,
             "nodes": [{"id": "account:" + account_id, "type": "account",
                        "source_id": account_id, "source_dataset": "botsim_24"}],
             "edges": []}
    add_passage(graph, account_id, user.get("description", ""),
                account_id, "Users.csv:description", "profile_description")
    try:
        subreddits = ast.literal_eval(user.get("subreddit") or "[]")
    except (SyntaxError, ValueError):
        subreddits = []
    for subreddit in subreddits if isinstance(subreddits, list) else []:
        link_entity(graph, "account:" + account_id, "subreddit",
                    str(subreddit), "participates_in")
    for kind, text_field, id_field in (("posts", "posts", "submission_id"),
                                       ("comment_1", "comment_body", "comment_id"),
                                       ("comment_2", "comment_body", "comment_id")):
        for i, item in enumerate(activities.get(kind) or []):
            if not isinstance(item, dict):
                continue
            item_id = str(item.get(id_field) or f"{kind}:ordinal:{i}")
            add_passage(graph, account_id, item.get(text_field) or "", item_id,
                        f"user_post_comment.json:{kind}[].{text_field}", kind,
                        {"ordinal": i, "subreddit": item.get("subreddit", "")})
            passage_id = stable_id("passage", account_id + "|" + kind + "|" + item_id)
            if any(n["id"] == passage_id for n in graph["nodes"]):
                link_entity(graph, passage_id, "subreddit",
                            item.get("subreddit", ""), "posted_in")
                if kind.startswith("comment"):
                    link_entity(graph, passage_id, "parent_object",
                                item.get("parent_id", ""), "replies_to")
    return graph


def retrieve(graph: dict, query: str, k: int = 5) -> dict:
    """Rank passages by deterministic lexical overlap, then expose linked entities.

    This is query-conditioned graph lookup, not WFM's learned attentive GFM.
    """
    query_terms = {term.lower() for term in TOKEN.findall(query) if len(term) > 2}
    passages = [node for node in graph["nodes"] if node["type"] == "passage"]
    ranked = sorted(passages, key=lambda node: (
        -len(query_terms & {t.lower() for t in TOKEN.findall(node["text"])}), node["id"]))[:k]
    selected = {node["id"] for node in ranked}
    entity_ids = {edge["to"] for edge in graph["edges"]
                  if edge["from"] in selected and edge["type"] == "mentions"}
    return {"query": query, "passages": ranked,
            "linked_entities": [n for n in graph["nodes"] if n["id"] in entity_ids],
            "links": [e for e in graph["edges"] if e["from"] in selected]}


def attach_rule_claim(graph: dict, source_node_ids: list[str], rule_id: str,
                      rule_version: str, tier: str, claim: str) -> str:
    """Attach a previously computed rule claim to its explicit source nodes.

    The caller must supply the claim and its provenance; this function does not
    evaluate T1--T4 or certify that a classifier relied on the cited inputs.
    """
    known = {node["id"] for node in graph["nodes"]}
    if not source_node_ids or any(source_id not in known for source_id in source_node_ids):
        raise ValueError("Every claim must cite at least one existing source node")
    if tier not in {"T1", "T2", "T3", "T4"}:
        raise ValueError("Unknown evidence tier")
    claim_id = stable_id("claim", "|".join((graph["dataset"], graph["account_id"],
                                               rule_id, rule_version, claim)))
    if claim_id in known:
        raise ValueError("Duplicate claim")
    graph["nodes"].append({"id": claim_id, "type": "claim", "tier": tier,
                           "rule_id": rule_id, "rule_version": rule_version,
                           "text": claim})
    for source_id in sorted(set(source_node_ids)):
        graph["edges"].append({"from": source_id, "to": claim_id,
                               "type": "derives_claim"})
    return claim_id


def load_fox8(account_id: str) -> dict:
    with FOX_PATH.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if str(record["user_id"]) == account_id:
                return fox8_graph(record)
    raise KeyError(f"Fox8 account not found: {account_id}")


def load_botsim(account_id: str) -> dict:
    with BOT_USERS.open(encoding="utf-8-sig", newline="") as source:
        user = next((u for u in csv.DictReader(source) if u["user_id"] == account_id), None)
    if user is None:
        raise KeyError(f"BotSim account not found: {account_id}")
    activities = json.loads(BOT_POSTS.read_text(encoding="utf-8")).get(account_id, {})
    return botsim_graph(user, activities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fox8_23", "botsim_24"), required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--query", default="profile activity content coordination")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    graph = load_fox8(args.account_id) if args.dataset == "fox8_23" else load_botsim(args.account_id)
    result = {"graph": graph, "retrieval": retrieve(graph, args.query, args.top_k)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.dataset, args.account_id, "nodes", len(graph["nodes"]),
          "edges", len(graph["edges"]), "->", args.out)


if __name__ == "__main__":
    main()
