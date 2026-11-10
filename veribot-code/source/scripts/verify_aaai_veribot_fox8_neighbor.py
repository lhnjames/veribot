"""Independently verify a neighbor fact, its raw graph bridge, and model replay."""

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
from train_aaai_veribot_fox8 import load_inputs, sha256_file, tensor_hash
from verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS
from verifiable_bot.aaai_provenance import SCHEMA_VERSION, verify_code_manifest
from verifiable_bot.aaai_veribot_core import VeriBotGraphInput
from verifiable_bot.evidence_v2 import SourceRecord

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
FEATURES = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
AUDIT = ROOT / "research/certificates/aaai_veribot_fox8_all_features_audit.json"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
DEFAULT = ROOT / "research/certificates/aaai_veribot_fox8_neighbor_context.json"
FIELD = "statuses_count"


def independent_selection(ids, split, graph, raw_values):
    neighbor_sets = [set() for _ in ids]
    for relation, edge in graph.items():
        for source, destination in edge.T.tolist():
            neighbor_sets[source].add(destination)
            neighbor_sets[destination].add(source)
    column = FEATURE_FIELDS.index(FIELD)
    choices = []
    for target in split["test"]:
        target = int(target)
        neighbors = sorted((ids[index], index) for index in neighbor_sets[target]
                           if index != target and raw_values[index, column] > 0)
        if len(neighbors) >= 2:
            choices.append((ids[target], target, neighbors[:2]))
    if not choices:
        raise ValueError("no structurally eligible test account")
    return min(choices, key=lambda item: item[0]), len(choices)


def location_parts(location: str):
    path, suffix = location.split("#line=", 1)
    if Path(path) != RAW.resolve():
        raise ValueError("source locator points to another file")
    line, remainder = suffix.split(";tweet=", 1)
    tweet, field = remainder.split(";", 1)
    return int(line), int(tweet), field


def direct_logits(hgt, relnet, user_x, graph, relnet_edges, target):
    with torch.inference_mode():
        h = float(hgt({"user": user_x}, graph)["bot_logit"][target])
        r = float(relnet(user_x, relnet_edges)[target])
    return {"hgt_logit": h, "relnet_logit": r, "fused_logit": (h + r) / 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", nargs="?", type=Path, default=DEFAULT)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = json.loads(args.certificate.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("certificate schema differs")
    verify_code_manifest(ROOT, report.get("code_sha256"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if (audit["status"] != "verified_raw_to_all_named_features_and_frozen_input" or
            report["raw_source_sha256"] != sha256_file(RAW) or
            report["event_index_sha256"] != sha256_file(EVENTS) or
            report["feature_file_sha256"] != sha256_file(FEATURES) or
            report["checkpoint_sha256"] != sha256_file(args.run / "model.pt") or
            audit["raw_source_sha256"] != report["raw_source_sha256"] or
            audit["checkpoint_sha256"] != report["checkpoint_sha256"]):
        raise ValueError("raw source, graph index, features or model differs")
    x, _, split, ids, graph, pack = load_inputs()
    raw = pd.read_csv(FEATURES)[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
    (target_id, target_index, neighbors), eligible_count = \
        independent_selection(ids, split, graph, raw)
    cert = report["certificate"]
    if (target_id != cert["account_id"] or
            eligible_count != report["eligible_test_accounts"]):
        raise ValueError("prediction-blind structural selection differs")
    candidate = cert["claim"]["source_ids"]
    control = cert["controls"][0]["source_ids"]
    if len(candidate) != 1 or len(control) != 1 or candidate[0] == control[0]:
        raise ValueError("source edit budgets differ")
    source_ids = (candidate[0], control[0])
    for sid, (uid, _) in zip(source_ids, neighbors):
        if sid != f"fox8:{uid}:profile:{FIELD}" or \
                report["source_input_paths"][sid] != [f"user[{uid}].{FIELD}"]:
            raise ValueError("neighbor-to-model input path differs")
    wanted_lines = set()
    for sid in source_ids:
        wanted_lines.add(location_parts(report["source_records"][sid]["raw_location"])[0])
        wanted_lines.add(location_parts(report["bridge_records"][sid]["raw_location"])[0])
    rows = {}
    first_line = {}
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            row = json.loads(text)
            uid = str(row["user_id"])
            first_line.setdefault(uid, line)
            if line in wanted_lines:
                rows[line] = row
    if set(rows) != wanted_lines:
        raise ValueError("raw source locator is missing")
    events = json.loads(EVENTS.read_text(encoding="utf-8"))
    if events["raw_source_sha256"] != report["raw_source_sha256"]:
        raise ValueError("graph index differs from raw source")
    for sid, (neighbor_id, neighbor_index) in zip(source_ids, neighbors):
        record = report["source_records"][sid]
        line, ordinal, field = location_parts(record["raw_location"])
        if ordinal != 0 or field != f"user.{FIELD}" or \
                str(rows[line]["user_id"]) != neighbor_id or \
                first_line[neighbor_id] != line:
            raise ValueError("profile locator is not the first released account record")
        value = rows[line]["user_tweets"][0]["user"][FIELD]
        original = SourceRecord(sid, neighbor_id, "profile_field",
                                {FIELD: value}, record["raw_location"])
        if (record["payload"] != original.payload or
                record["sha256"] != original.sha256 or
                float(value) != float(raw[neighbor_index, FEATURE_FIELDS.index(FIELD)])):
            raise ValueError("neighbor profile fact differs from model input")
        if sid == candidate[0] and (
                cert["claim"]["operation"] != "field" or
                float(cert["claim"]["recomputed_value"]) != float(value) or
                cert["input_dependencies"][sid] != [f"user[{neighbor_id}].{FIELD}"] or
                cert["source_locations"][sid] != record["raw_location"]):
            raise ValueError("source claim or declared input path differs")
        bridge = report["bridge_records"][sid]
        bridge_line, bridge_ordinal, bridge_field = location_parts(bridge["raw_location"])
        bridge_row = rows[bridge_line]
        bridge_source = str(bridge_row["user_id"])
        relation = bridge["kind"].removesuffix("_target_pointer")
        tweet = bridge_row["user_tweets"][bridge_ordinal]
        if relation == "reply":
            pointed = tweet.get("in_reply_to_user_id_str") or tweet.get("in_reply_to_user_id")
            expected_field = "in_reply_to_user_id_str|in_reply_to_user_id"
        elif relation == "retweet":
            user = (tweet.get("retweeted_status") or {}).get("user") or {}
            pointed = user.get("id_str") or user.get("id")
            expected_field = "retweeted_status.user.id_str|id"
        else:
            raise ValueError("unknown raw bridge relation")
        pointed = str(pointed)
        if (bridge_field != expected_field or
                {bridge_source, pointed} != {target_id, neighbor_id} or
                bridge["account_id"] != bridge_source):
            raise ValueError("raw relation pointer does not bridge the two accounts")
        bridge_source_id = (f"fox8:bridge:{relation}:{bridge_source}:{pointed}:"
                            f"{bridge_line}:{bridge_ordinal}")
        bridge_original = SourceRecord(
            bridge_source_id,
            bridge_source, bridge["kind"],
            {"target_user_id": pointed,
             "tweet_id": str(tweet.get("id_str") or tweet.get("id"))},
            bridge["raw_location"])
        if (bridge["payload"] != bridge_original.payload or
                bridge["sha256"] != bridge_original.sha256):
            raise ValueError("bridge source hash differs")
        event_found = any(
            str(event["source_id"]) == bridge_source and
            str(event["target_id"]) == pointed and event["relation"] == relation and
            any(int(locator["line"]) == bridge_line and
                int(locator["tweet_index"]) == bridge_ordinal
                for locator in event["locators"])
            for event in events["events"])
        if not event_found:
            raise ValueError("bridge pointer absent from native event index")
        edge = graph[("user", relation, "user")]
        source_index, pointed_index = ids.index(bridge_source), ids.index(pointed)
        if not bool(((edge[0] == source_index) & (edge[1] == pointed_index)).any()):
            raise ValueError("raw bridge does not reach the frozen HGT/RelNet graph")

    checkpoint = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    contract = checkpoint["input_contract"]
    if (tensor_hash(x) != contract["feature_sha256"] or
            tensor_hash(x) != report["baseline_scaled_input_sha256"]):
        raise ValueError("frozen input fingerprint differs")
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
    hgt.eval()
    relnet.eval()

    def replay(removed: tuple[str, ...]):
        values = raw.copy()
        for sid, (_, neighbor_index) in zip(source_ids, neighbors):
            if sid in removed:
                values[neighbor_index, FEATURE_FIELDS.index(FIELD)] = 0.0
        updated = torch.from_numpy(
            np.clip(scaler.transform(values), -3, 3).astype(np.float32))
        built = VeriBotGraphInput(
            x_dict={"user": updated}, edge_index_dict=graph,
            relnet_edges=pack["relnet_edges"], account_ids=tuple(ids),
            user_fact_names=tuple(FEATURE_FIELDS), source_dependencies={},
            graph_rule_sha256=sha256(pack["contract"]["graph_rule"].encode()).hexdigest())
        return (direct_logits(hgt, relnet, updated, graph,
                              pack["relnet_edges"], target_index),
                tensor_hash(updated), built.fingerprint())

    before, before_hash, before_fingerprint = replay(())
    after, after_hash, after_fingerprint = replay((candidate[0],))
    controlled, control_hash, _ = replay((control[0],))
    if before_hash != contract["feature_sha256"] or \
            after_hash == before_hash or control_hash == before_hash:
        raise ValueError("neighbor source edit did not rebuild the frozen input")
    if (before_fingerprint != cert["before"]["model_input_sha256"] or
            after_fingerprint != cert["after"]["model_input_sha256"]):
        raise ValueError("source-edited input fingerprint differs")
    maximum = 0.0
    for stage, actual in (("before", before), ("after", after)):
        for key in ("hgt_logit", "relnet_logit", "fused_logit"):
            difference = abs(actual[key] - cert[stage][key])
            maximum = max(maximum, difference)
            if difference > 1e-5:
                raise ValueError(f"target {stage} model output differs: {key}")
    for branch in ("hgt", "relnet", "fused"):
        key = f"{branch}_logit"
        difference = abs(controlled[key] - before[key] -
                         cert["controls"][0][f"{branch}_delta_logit"])
        maximum = max(maximum, difference)
        if difference > 1e-5:
            raise ValueError(f"neighbor control response differs: {branch}")
    saved = json.loads((args.run / "test_predictions.json").read_text(encoding="utf-8"))
    prediction = next(item for item in saved["predictions"]
                      if item["account_id"] == target_id)
    if any(abs(prediction[key] - before[key]) > 1e-5
           for key in ("hgt_logit", "relnet_logit", "fused_logit")):
        raise ValueError("target baseline does not match saved test prediction")
    result = {"status": "verified_raw_neighbor_bridge_fact_and_frozen_two_branch_replay",
              "certificate_sha256": sha256_file(args.certificate),
              "target_account_id": target_id,
              "candidate_neighbor": neighbors[0][0],
              "control_neighbor": neighbors[1][0],
              "eligible_test_accounts": eligible_count,
              "raw_bridges_verified": 2,
              "max_output_difference": maximum}
    destination = args.certificate.with_name(args.certificate.stem + "_verification.json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
