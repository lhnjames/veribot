"""Certify one source pointer through frozen VeriBot HGT and RelNet branches."""

from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
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
from verifiable_bot.aaai_fox8_native_graph import native_relation_edges
from verifiable_bot.aaai_veribot_core import (
    VeriBotDecision, VeriBotGraphInput, certify_aaai_source_response,
)
from verifiable_bot.evidence_v2 import AtomicClaim, SourceRecord
from verifiable_bot.aaai_provenance import SCHEMA_VERSION, code_manifest

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
CODE_FILES = (
    "source/scripts/certify_aaai_veribot_fox8.py",
    "source/scripts/train_aaai_veribot_fox8.py",
    "source/verifiable_bot/aaai_veribot_core.py",
    "source/verifiable_bot/aaai_fox8_native_graph.py",
    "source/verifiable_bot/aaai_fox8_feature_schema.py",
    "source/verifiable_bot/evidence_v2.py",
    "source/hgt_import/code/hgt_model.py",
    "source/hgt_import/code/relnet_reimplementation.py",
)


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def event_record(event: dict) -> SourceRecord:
    if event["count"] != 1 or len(event["locators"]) != 1:
        raise ValueError("this certificate removes one raw target-user pointer")
    locator = event["locators"][0]
    source = str(event["source_id"])
    target = str(event["target_id"])
    relation = event["relation"]
    source_id = (f"fox8:{relation}:{source}:{target}:"
                 f"{locator['line']}:{locator['tweet_index']}")
    location = (f"{RAW.resolve()}#line={locator['line']};"
                f"tweet={locator['tweet_index']};{locator['field']}")
    return SourceRecord(source_id, source, f"{relation}_target_pointer",
                        {"target_user_id": target,
                         "tweet_id": str(locator["tweet_id"])}, location)


def readback_sources(sources: dict[str, SourceRecord]) -> None:
    by_line: dict[int, list[SourceRecord]] = {}
    for record in sources.values():
        path, locator = record.raw_location.split("#line=", 1)
        if Path(path) != RAW.resolve():
            raise ValueError("pointer refers to another raw source")
        line = int(locator.split(";", 1)[0])
        by_line.setdefault(line, []).append(record)
    found = set()
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line_no not in by_line:
                continue
            row = json.loads(line)
            for record in by_line[line_no]:
                tail = record.raw_location.split(";tweet=", 1)[1]
                ordinal_text, field = tail.split(";", 1)
                ordinal = int(ordinal_text)
                tweet = row["user_tweets"][ordinal]
                if record.kind == "reply_target_pointer":
                    if field != "in_reply_to_user_id_str|in_reply_to_user_id":
                        raise ValueError("reply field locator differs")
                    target = tweet.get("in_reply_to_user_id_str") or tweet.get("in_reply_to_user_id")
                elif record.kind == "retweet_target_pointer":
                    if field != "retweeted_status.user.id_str|id":
                        raise ValueError("retweet field locator differs")
                    user = (tweet.get("retweeted_status") or {}).get("user") or {}
                    target = user.get("id_str") or user.get("id")
                else:
                    raise ValueError("unexpected pointer kind")
                tweet_id = str(tweet.get("id_str") or tweet.get("id") or ordinal)
                if (str(row["user_id"]) != record.account_id
                        or str(target) != record.payload["target_user_id"]
                        or tweet_id != record.payload["tweet_id"]):
                    raise ValueError(f"raw pointer readback differs: {record.source_id}")
                found.add(record.source_id)
    if found != set(sources):
        raise ValueError("a selected source pointer was not found")


def select_pair(events: list[dict], test_ids: set[str]) -> tuple[str, dict, dict]:
    """Choose by source structure before reading any model outputs or labels."""
    incident: dict[str, list[dict]] = {}
    for event in events:
        if event["count"] != 1:
            continue
        for uid in (str(event["source_id"]), str(event["target_id"])):
            if uid in test_ids:
                incident.setdefault(uid, []).append(event)
    for uid in sorted(incident):
        ordered = sorted(incident[uid], key=lambda row: (
            row["relation"], str(row["source_id"]), str(row["target_id"])))
        if len(ordered) >= 2:
            return uid, ordered[0], ordered[1]
    raise ValueError("test accounts have no equal-count source pointer pair")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/certificates/aaai_veribot_fox8_native_pointer.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    x, _, split, ids, graph, pack = load_inputs()
    saved = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    if saved["input_contract"]["feature_sha256"] != tensor_hash(x):
        raise ValueError("checkpoint feature input differs")
    if any(saved["input_contract"]["graph_sha256"][name] != tensor_hash(edge)
           for name, edge in pack["relnet_edges"].items()):
        raise ValueError("checkpoint native graph differs")
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3)
    relnet = GatedRelNet(x.shape[1], hidden_channels=128,
                         relation_names=tuple(pack["relnet_edges"]),
                         num_layers=2, dropout=0.3)
    hgt.load_state_dict(saved["model_state"]["hgt"])
    relnet.load_state_dict(saved["model_state"]["relnet"])
    decision = VeriBotDecision(
        hgt, relnet,
        hgt_sha256=state_hash(saved["model_state"]["hgt"]),
        relnet_sha256=state_hash(saved["model_state"]["relnet"]))
    payload = json.loads(EVENTS.read_text(encoding="utf-8"))
    account_id, candidate, control = select_pair(
        payload["events"], {ids[int(i)] for i in split["test"]})
    sources = {record.source_id: record for record in
               (event_record(candidate), event_record(control))}
    readback_sources(sources)
    keyed = {event_record(event).source_id: (
        str(event["source_id"]), str(event["target_id"]), event["relation"])
        for event in (candidate, control)}
    original_hashes = pack["contract"]["graph_sha256"]

    def build_input(remaining: dict[str, SourceRecord]) -> VeriBotGraphInput:
        excluded = frozenset(keyed[sid] for sid in sources if sid not in remaining)
        typed, meta = native_relation_edges(
            raw_path=RAW, event_index_path=EVENTS,
            account_ids=ids, excluded_events=excluded)
        if not excluded and any(tensor_hash(typed[name]) != original_hashes[name]
                                for name in typed):
            raise ValueError("baseline graph differs from frozen training input")
        paths = {sid: (f"graph.{key[2]}[{key[0]}->{key[1]}]",
                       f"graph.rev_{key[2]}[{key[1]}->{key[0]}]")
                 for sid, key in keyed.items()}
        return VeriBotGraphInput(
            x_dict={"user": x},
            edge_index_dict={("user", name, "user"): edge
                             for name, edge in typed.items()},
            relnet_edges=typed, account_ids=tuple(ids),
            user_fact_names=tuple(pack["contract"]["feature_names"]),
            source_dependencies=paths,
            graph_rule_sha256=sha256(meta["operation"].encode()).hexdigest())

    candidate_id, control_id = tuple(sources)
    claim = AtomicClaim((candidate_id,), "field",
                        sources[candidate_id].payload["target_user_id"],
                        field="target_user_id")
    certificate = certify_aaai_source_response(
        account_id=account_id, claim=claim, sources=sources,
        build_input=build_input, decision=decision,
        control_source_groups=((control_id,),))
    predictions = json.loads((args.run / "test_predictions.json").read_text(encoding="utf-8"))
    saved_prediction = next(row for row in predictions["predictions"]
                            if row["account_id"] == account_id)
    if any(abs(certificate["before"][field] - saved_prediction[field]) > 1e-5
           for field in ("hgt_logit", "relnet_logit", "fused_logit")):
        raise ValueError("certificate does not replay saved test prediction")
    output = {
        "schema_version": SCHEMA_VERSION,
        "code_sha256": code_manifest(ROOT, CODE_FILES),
        "selection_rule": "lexicographically first test account with two distinct single-pointer incident events; first event candidate, second control",
        "raw_source_sha256": sha256_file(RAW),
        "event_index_sha256": sha256_file(EVENTS),
        "checkpoint_sha256": sha256_file(args.run / "model.pt"),
        "selected_source_records": {
            sid: {"account_id": record.account_id, "kind": record.kind,
                  "payload": dict(record.payload), "raw_location": record.raw_location,
                  "sha256": record.sha256}
            for sid, record in sources.items()},
        "source_readback": "verified_from_raw_gzip",
        "certificate": certificate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"account_id": account_id,
                      "candidate": candidate_id, "control": control_id,
                      "hgt_delta_logit": certificate["hgt_delta_logit"],
                      "relnet_delta_logit": certificate["relnet_delta_logit"],
                      "fused_delta_logit": certificate["fused_delta_logit"],
                      "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
