"""Independently rebuild Fox8 edges from raw posts and replay a VeriBot card."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/scripts"))
sys.path.insert(0, str(ROOT / "source/hgt_import/code"))

from hgt_model import VerifiableHGT
from relnet_reimplementation import GatedRelNet
from train_aaai_veribot_fox8 import load_inputs, sha256_file, tensor_hash
from verifiable_bot.evidence_v2 import SourceRecord
from verifiable_bot.aaai_provenance import SCHEMA_VERSION, verify_code_manifest
from verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS
from verifiable_bot.aaai_veribot_core import VeriBotGraphInput

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
RELATIONS = ("reply", "rev_reply", "retweet", "rev_retweet")


def pointer_location(record: dict) -> tuple[int, int, str]:
    path, suffix = record["raw_location"].split("#line=", 1)
    if Path(path) != RAW.resolve():
        raise ValueError("certificate source points to a different file")
    line, remainder = suffix.split(";tweet=", 1)
    tweet, field = remainder.split(";", 1)
    return int(line), int(tweet), field


def raw_relation_counts(account_ids: list[str], records: dict) -> tuple[Counter, dict]:
    known = set(account_ids)
    selected = {pointer_location(record): (sid, record)
                for sid, record in records.items()}
    if len(selected) != len(records):
        raise ValueError("selected source locations collide")
    found = set()
    seen_users = set()
    counts: Counter[tuple[str, str, str]] = Counter()
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            source = str(row["user_id"])
            if source in seen_users:
                continue
            seen_users.add(source)
            for ordinal, tweet in enumerate(row.get("user_tweets") or []):
                targets = (
                    ("reply", tweet.get("in_reply_to_user_id_str") or
                     tweet.get("in_reply_to_user_id"),
                     "in_reply_to_user_id_str|in_reply_to_user_id"),
                    ("retweet", ((tweet.get("retweeted_status") or {}).get("user") or {}).get("id_str")
                     or ((tweet.get("retweeted_status") or {}).get("user") or {}).get("id"),
                     "retweeted_status.user.id_str|id"),
                )
                for relation, target, field in targets:
                    if target is None:
                        continue
                    target = str(target)
                    if target not in known or target == source:
                        continue
                    counts[source, target, relation] += 1
                    selected_key = (line_number, ordinal, field)
                    if selected_key in selected:
                        sid, record = selected[selected_key]
                        tweet_id = str(tweet.get("id_str") or tweet.get("id") or ordinal)
                        expected = SourceRecord(
                            sid, source, f"{relation}_target_pointer",
                            {"target_user_id": target, "tweet_id": tweet_id},
                            record["raw_location"])
                        if (expected.sha256 != record["sha256"]
                                or expected.payload != record["payload"]
                                or expected.account_id != record["account_id"]
                                or expected.kind != record["kind"]):
                            raise ValueError(f"raw source differs: {sid}")
                        found.add(sid)
    if seen_users != known or found != set(records):
        raise ValueError("raw account inventory or pointer readback differs")
    return counts, {sid: pointer_location(record) for sid, record in records.items()}


def graph_from_counts(counts: Counter, account_ids: list[str],
                      removed: tuple[str, ...], selected: dict,
                      records: dict) -> dict[str, torch.Tensor]:
    changed = counts.copy()
    for sid in removed:
        record = records[sid]
        relation = record["kind"].removesuffix("_target_pointer")
        key = (record["account_id"], record["payload"]["target_user_id"], relation)
        if changed[key] < 1:
            raise ValueError("source operation has no matching raw pointer")
        changed[key] -= 1
    index = {uid: i for i, uid in enumerate(account_ids)}
    pairs = {relation: set() for relation in RELATIONS}
    for (source, target, relation), count in changed.items():
        if count < 1:
            continue
        i, j = index[source], index[target]
        pairs[relation].add((i, j))
        pairs[f"rev_{relation}"].add((j, i))
    return {name: (torch.tensor(sorted(pairs[name]), dtype=torch.long).T.contiguous()
                   if pairs[name] else torch.empty((2, 0), dtype=torch.long))
            for name in RELATIONS}


def direct_logits(hgt, relnet, x: torch.Tensor,
                  edges: dict[str, torch.Tensor], target: int) -> dict[str, float]:
    graph = {("user", name, "user"): value for name, value in edges.items()}
    with torch.inference_mode():
        h = float(hgt({"user": x}, graph)["bot_logit"][target])
        r = float(relnet(x, edges)[target])
    return {"hgt_logit": h, "relnet_logit": r, "fused_logit": 0.5 * (h + r)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", nargs="?", type=Path,
                        default=ROOT / "research/certificates/aaai_veribot_fox8_native_pointer.json")
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = json.loads(args.certificate.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("certificate schema version differs")
    verify_code_manifest(ROOT, report.get("code_sha256"))
    if (report["raw_source_sha256"] != sha256_file(RAW)
            or report["event_index_sha256"] != sha256_file(EVENTS)
            or report["checkpoint_sha256"] != sha256_file(args.run / "model.pt")):
        raise ValueError("raw source, event index or checkpoint differs")
    x, _, split, ids, _, pack = load_inputs()
    records = report["selected_source_records"]
    counts, selected = raw_relation_counts(ids, records)
    indexed = json.loads(EVENTS.read_text(encoding="utf-8"))["events"]
    indexed_counts = Counter({(str(row["source_id"]), str(row["target_id"]), row["relation"]):
                              int(row["count"]) for row in indexed})
    if counts != indexed_counts:
        raise ValueError("raw interaction counts differ from event index")
    cert = report["certificate"]
    account_id = cert["account_id"]
    if account_id not in {ids[int(i)] for i in split["test"]}:
        raise ValueError("certified account is outside fixed test split")
    candidate = tuple(cert["claim"]["source_ids"])
    control = tuple(cert["controls"][0]["source_ids"])
    if len(candidate) != len(control) or len(candidate) != 1 or set(candidate) & set(control):
        raise ValueError("source and control budgets differ")
    if (cert["claim"]["operation"] != "field" or
            cert["claim"]["recomputed_value"] !=
            records[candidate[0]]["payload"]["target_user_id"] or
            cert["source_locations"][candidate[0]] !=
            records[candidate[0]]["raw_location"]):
        raise ValueError("raw pointer claim differs")
    graphs = {name: graph_from_counts(counts, ids, removed, selected, records)
              for name, removed in (("before", ()), ("after", candidate),
                                    ("control", control))}
    if any(tensor_hash(edge) != pack["contract"]["graph_sha256"][name]
           for name, edge in graphs["before"].items()):
        raise ValueError("direct raw rebuild differs from training graph")
    saved = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    if saved["input_contract"]["feature_sha256"] != tensor_hash(x):
        raise ValueError("checkpoint feature matrix differs")
    for state in ("before", "after"):
        relation_edges = graphs[state]
        built = VeriBotGraphInput(
            x_dict={"user": x},
            edge_index_dict={("user", name, "user"): edge
                             for name, edge in relation_edges.items()},
            relnet_edges=relation_edges, account_ids=tuple(ids),
            user_fact_names=tuple(FEATURE_FIELDS), source_dependencies={},
            graph_rule_sha256=cert["graph_rule_sha256"])
        if built.fingerprint() != cert[state]["model_input_sha256"]:
            raise ValueError(f"{state} complete graph input fingerprint differs")
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3)
    relnet = GatedRelNet(x.shape[1], hidden_channels=128,
                         relation_names=tuple(RELATIONS), num_layers=2, dropout=0.3)
    hgt.load_state_dict(saved["model_state"]["hgt"])
    relnet.load_state_dict(saved["model_state"]["relnet"])
    hgt.eval()
    relnet.eval()
    target = ids.index(account_id)
    actual = {name: direct_logits(hgt, relnet, x, graph, target)
              for name, graph in graphs.items()}
    expected = {"before": cert["before"], "after": cert["after"]}
    for stage in ("before", "after"):
        for key in ("hgt_logit", "relnet_logit", "fused_logit"):
            if abs(actual[stage][key] - expected[stage][key]) > 1e-5:
                raise ValueError(f"{stage} {key} does not replay")
    for branch in ("hgt", "relnet", "fused"):
        key = f"{branch}_delta_logit"
        if abs(actual["control"][f"{branch}_logit"] -
               actual["before"][f"{branch}_logit"] -
               cert["controls"][0][key]) > 1e-5:
            raise ValueError(f"control {branch} does not replay")
    output = {"status": "verified_direct_raw_graph_and_frozen_model_replay",
              "certificate_sha256": sha256_file(args.certificate),
              "account_id": account_id,
              "raw_event_records": len(counts),
              "raw_pointers": int(sum(counts.values())),
              "graphs_rebuilt": list(graphs),
              "max_output_difference": max(
                  abs(actual[stage][key] - expected[stage][key])
                  for stage in expected for key in
                  ("hgt_logit", "relnet_logit", "fused_logit"))}
    destination = args.certificate.with_name(args.certificate.stem + "_verification.json")
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
