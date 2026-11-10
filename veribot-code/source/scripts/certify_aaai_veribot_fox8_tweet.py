"""Certify one raw Fox8 tweet through the frozen HGT/RelNet classifier."""

from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/scripts"))
sys.path.insert(0, str(ROOT / "source/hgt_import/code"))

from hgt_model import VerifiableHGT
from relnet_reimplementation import GatedRelNet
from certify_aaai_veribot_fox8 import state_hash
from train_aaai_veribot_fox8 import load_inputs, sha256_file, tensor_hash
from verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS, TWEET_FIELDS
from verifiable_bot.aaai_fox8_raw_features import account_features
from verifiable_bot.aaai_veribot_core import (
    VeriBotDecision, VeriBotGraphInput, certify_aaai_source_response,
)
from verifiable_bot.aaai_provenance import SCHEMA_VERSION, code_manifest
from verifiable_bot.evidence_v2 import AtomicClaim, SourceRecord, digest

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
FEATURES = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
AUDIT = ROOT / "research/certificates/aaai_veribot_fox8_all_features_audit.json"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
OUTPUT = ROOT / "research/certificates/aaai_veribot_fox8_tweet_behavior.json"
CODE_FILES = (
    "source/scripts/certify_aaai_veribot_fox8_tweet.py",
    "source/scripts/train_aaai_veribot_fox8.py",
    "source/verifiable_bot/aaai_fox8_raw_features.py",
    "source/verifiable_bot/aaai_fox8_feature_schema.py",
    "source/verifiable_bot/aaai_fox8_native_graph.py",
    "source/verifiable_bot/aaai_veribot_core.py",
    "source/verifiable_bot/evidence_v2.py",
    "source/hgt_import/code/hgt_model.py",
    "source/hgt_import/code/relnet_reimplementation.py",
)


def graph_target(tweet: dict, relation: str):
    if relation == "reply":
        return tweet.get("in_reply_to_user_id_str") or tweet.get("in_reply_to_user_id")
    user = (tweet.get("retweeted_status") or {}).get("user") or {}
    return user.get("id_str") or user.get("id")


def has_internal_relation(tweet: dict, source_id: str, account_ids: set[str]) -> bool:
    return any(target is not None and str(target) in account_ids and
               str(target) != source_id
               for target in (graph_target(tweet, "reply"),
                              graph_target(tweet, "retweet")))


def read_selected(test_ids: set[str], account_ids: set[str]):
    seen = set()
    eligible = []
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid not in test_ids or uid in seen:
                continue
            seen.add(uid)
            tweets = row["user_tweets"]
            choices = {True: [], False: []}
            for ordinal, tweet in enumerate(tweets):
                if ordinal == 0 or has_internal_relation(tweet, uid, account_ids):
                    continue
                choices[(tweet.get("text") or "").startswith("RT ")].append(ordinal)
            if choices[True] and choices[False]:
                eligible.append((uid, line_number, row,
                                 choices[True][0], choices[False][0]))
    if not eligible:
        raise ValueError("no fixed-test account has an RT and ordinary tweet without internal edges")
    return min(eligible, key=lambda item: item[0]), len(eligible)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    x, _, split, ids, graph, pack = load_inputs()
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if (audit["status"] != "verified_raw_to_all_named_features_and_frozen_input" or
            audit["raw_source_sha256"] != sha256_file(RAW) or
            audit["scaled_input_sha256"] != tensor_hash(x)):
        raise ValueError("raw feature audit does not match frozen input")
    test_ids = {ids[int(index)] for index in split["test"]}
    (uid, line_number, row, candidate_ordinal, control_ordinal), eligible_count = \
        read_selected(test_ids, set(ids))
    index = ids.index(uid)
    frame = pd.read_csv(FEATURES)
    raw_values = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
    baseline = account_features(row)
    expected = np.asarray([baseline[name] for name in FEATURE_FIELDS], dtype=np.float32)
    if not np.allclose(expected, raw_values[index], rtol=1e-6, atol=1e-5):
        raise ValueError("chosen raw account differs from the audited feature table")
    sources = {}
    ordinal_by_id = {}
    source_paths = {}
    for ordinal in (candidate_ordinal, control_ordinal):
        tweet = row["user_tweets"][ordinal]
        sid = f"fox8:{uid}:tweet:{line_number}:{ordinal}"
        sources[sid] = SourceRecord(
            sid, uid, "raw_tweet",
            {"tweet_id": str(tweet.get("id_str") or tweet.get("id")),
             "tweet_sha256": digest(tweet),
             "starts_rt": (tweet.get("text") or "").startswith("RT ")},
            f"{RAW.resolve()}#line={line_number};tweet={ordinal}")
        ordinal_by_id[sid] = ordinal
        remaining = [item for j, item in enumerate(row["user_tweets"]) if j != ordinal]
        changed = account_features({**row, "user_tweets": remaining})
        names = [name for name in TWEET_FIELDS
                 if float(np.float32(changed[name])) != float(np.float32(baseline[name]))]
        if "n_tweets" not in names:
            raise ValueError("tweet removal did not change the tweet count")
        source_paths[sid] = tuple(f"user[{uid}].{name}" for name in names)
    checkpoint = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    contract = checkpoint["input_contract"]
    if (contract["feature_sha256"] != tensor_hash(x) or
            contract["feature_file_sha256"] != sha256_file(FEATURES)):
        raise ValueError("frozen checkpoint uses different facts")
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(contract["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(contract["scaler_scale"], dtype=np.float64)
    scaler.n_features_in_ = len(FEATURE_FIELDS)
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3)
    relnet = GatedRelNet(x.shape[1], hidden_channels=128,
                         relation_names=tuple(pack["relnet_edges"]),
                         num_layers=2, dropout=0.3)
    hgt.load_state_dict(checkpoint["model_state"]["hgt"])
    relnet.load_state_dict(checkpoint["model_state"]["relnet"])
    decision = VeriBotDecision(
        hgt, relnet,
        hgt_sha256=state_hash(checkpoint["model_state"]["hgt"]),
        relnet_sha256=state_hash(checkpoint["model_state"]["relnet"]))
    graph_rule = sha256(pack["contract"]["graph_rule"].encode()).hexdigest()

    def build_input(remaining: dict[str, SourceRecord]) -> VeriBotGraphInput:
        removed = {ordinal_by_id[sid] for sid in sources if sid not in remaining}
        tweets = [item for j, item in enumerate(row["user_tweets"]) if j not in removed]
        computed = account_features({**row, "user_tweets": tweets})
        updated = raw_values.copy()
        updated[index] = [computed[name] for name in FEATURE_FIELDS]
        user_x = torch.from_numpy(
            np.clip(scaler.transform(updated), -3, 3).astype(np.float32))
        if not removed and tensor_hash(user_x) != contract["feature_sha256"]:
            raise ValueError("raw tweet baseline differs from frozen input")
        return VeriBotGraphInput(
            x_dict={"user": user_x}, edge_index_dict=graph,
            relnet_edges=pack["relnet_edges"], account_ids=tuple(ids),
            user_fact_names=tuple(FEATURE_FIELDS),
            source_dependencies=source_paths, graph_rule_sha256=graph_rule)

    candidate = f"fox8:{uid}:tweet:{line_number}:{candidate_ordinal}"
    control = f"fox8:{uid}:tweet:{line_number}:{control_ordinal}"
    claim = AtomicClaim((candidate,), "field", True, field="starts_rt")
    certificate = certify_aaai_source_response(
        account_id=uid, claim=claim, sources=sources, build_input=build_input,
        decision=decision, control_source_groups=((control,),))
    saved = json.loads((args.run / "test_predictions.json").read_text(encoding="utf-8"))
    prediction = next(item for item in saved["predictions"] if item["account_id"] == uid)
    if any(abs(prediction[name] - certificate["before"][name]) > 1e-5
           for name in ("hgt_logit", "relnet_logit", "fused_logit")):
        raise ValueError("baseline prediction differs from the saved fixed-test output")
    operation_facts = {}
    for label, removed in (("before", frozenset()),
                           ("after", frozenset((candidate,))),
                           ("control", frozenset((control,)))):
        tweets = [item for j, item in enumerate(row["user_tweets"])
                  if j not in {ordinal_by_id[sid] for sid in removed}]
        facts = account_features({**row, "user_tweets": tweets})
        operation_facts[label] = {name: float(facts[name]) for name in TWEET_FIELDS}
    output = {
        "schema_version": SCHEMA_VERSION,
        "code_sha256": code_manifest(ROOT, CODE_FILES),
        "selection_rule": "lexicographically first fixed-test account with one RT-prefixed and one ordinary nonfirst tweet, each lacking an internal graph target; first eligible tweet in each group",
        "eligible_test_accounts": eligible_count,
        "source_edit": "remove one complete raw tweet; neither selected tweet supplies a model graph edge",
        "raw_source_sha256": sha256_file(RAW),
        "feature_file_sha256": sha256_file(FEATURES),
        "checkpoint_sha256": sha256_file(args.run / "model.pt"),
        "baseline_scaled_input_sha256": tensor_hash(x),
        "source_records": {sid: {"account_id": record.account_id,
                                 "kind": record.kind, "payload": dict(record.payload),
                                 "raw_location": record.raw_location,
                                 "sha256": record.sha256}
                           for sid, record in sources.items()},
        "source_input_paths": source_paths,
        "operation_facts": operation_facts,
        "certificate": certificate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"account_id": uid, "eligible_test_accounts": eligible_count,
                      "candidate": candidate, "control": control,
                      "before_retweet_ratio": operation_facts["before"]["retweet_ratio"],
                      "after_retweet_ratio": operation_facts["after"]["retweet_ratio"],
                      "hgt_delta_logit": certificate["hgt_delta_logit"],
                      "relnet_delta_logit": certificate["relnet_delta_logit"],
                      "fused_delta_logit": certificate["fused_delta_logit"]}), flush=True)


if __name__ == "__main__":
    main()
