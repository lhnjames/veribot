"""Independently read raw Fox8 tweets and replay their frozen model effect."""

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
from verifiable_bot.aaai_fox8_feature_schema import FEATURE_FIELDS, TWEET_FIELDS
from verifiable_bot.aaai_provenance import SCHEMA_VERSION, verify_code_manifest
from verifiable_bot.aaai_veribot_core import VeriBotGraphInput
from verifiable_bot.evidence_v2 import SourceRecord, digest

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
FEATURES = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
AUDIT = ROOT / "research/certificates/aaai_veribot_fox8_all_features_audit.json"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
DEFAULT = ROOT / "research/certificates/aaai_veribot_fox8_tweet_behavior.json"


def tweet_values(tweets: list[dict]) -> dict[str, float]:
    count = len(tweets)
    if count == 0:
        raise ValueError("source edit removed every account tweet")
    texts = [tweet.get("text") or "" for tweet in tweets]
    favorite = [tweet.get("favorite_count") or 0 for tweet in tweets]
    retweet = [tweet.get("retweet_count") or 0 for tweet in tweets]
    return {
        "n_tweets": count,
        "retweet_ratio": sum(text[:3] == "RT " for text in texts) / count,
        "reply_ratio": sum(tweet.get("in_reply_to_status_id") is not None
                           for tweet in tweets) / count,
        "avg_tweet_len": sum(map(len, texts)) / count,
        "avg_favorites": sum(favorite) / count,
        "avg_retweets": sum(retweet) / count,
        "avg_score": sum(a + b for a, b in zip(favorite, retweet)) / count,
        "unique_lang_ratio": len({tweet.get("lang", "und") for tweet in tweets}) / count,
    }


def model_edge_target(tweet: dict, relation: str):
    if relation == "reply":
        return tweet.get("in_reply_to_user_id_str") or tweet.get("in_reply_to_user_id")
    retweeted = tweet.get("retweeted_status") or {}
    user = retweeted.get("user") or {}
    return user.get("id_str") or user.get("id")


def no_model_edge(tweet: dict, source: str, known: set[str]) -> bool:
    for relation in ("reply", "retweet"):
        target = model_edge_target(tweet, relation)
        if target is not None and str(target) in known and str(target) != source:
            return False
    return True


def independently_select(test: set[str], known: set[str]):
    seen = set()
    choices = []
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            uid = str(row["user_id"])
            if uid not in test or uid in seen:
                continue
            seen.add(uid)
            rt = ordinary = None
            for ordinal, tweet in enumerate(row["user_tweets"]):
                if ordinal == 0 or not no_model_edge(tweet, uid, known):
                    continue
                if (tweet.get("text") or "").startswith("RT "):
                    if rt is None:
                        rt = ordinal
                elif ordinary is None:
                    ordinary = ordinal
            if rt is not None and ordinary is not None:
                choices.append((uid, line_number, row, rt, ordinary))
    if not choices:
        raise ValueError("no eligible test account")
    return min(choices, key=lambda item: item[0]), len(choices)


def direct_logits(hgt, relnet, user_x, graph, relnet_edges, index):
    with torch.inference_mode():
        h = float(hgt({"user": user_x}, graph)["bot_logit"][index])
        r = float(relnet(user_x, relnet_edges)[index])
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
            report["feature_file_sha256"] != sha256_file(FEATURES) or
            report["checkpoint_sha256"] != sha256_file(args.run / "model.pt") or
            audit["raw_source_sha256"] != report["raw_source_sha256"] or
            audit["checkpoint_sha256"] != report["checkpoint_sha256"]):
        raise ValueError("raw source, feature table, or model differs")
    x, _, split, ids, graph, pack = load_inputs()
    checkpoint = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    contract = checkpoint["input_contract"]
    if (tensor_hash(x) != report["baseline_scaled_input_sha256"] or
            tensor_hash(x) != contract["feature_sha256"]):
        raise ValueError("baseline model input differs")
    test = {ids[int(j)] for j in split["test"]}
    (uid, line, row, rt, ordinary), count = independently_select(test, set(ids))
    if count != report["eligible_test_accounts"]:
        raise ValueError("eligible source population differs")
    cert = report["certificate"]
    if uid != cert["account_id"]:
        raise ValueError("prediction-blind account selection differs")
    candidate = cert["claim"]["source_ids"]
    control = cert["controls"][0]["source_ids"]
    if len(candidate) != len(control) or len(candidate) != 1 or set(candidate) & set(control):
        raise ValueError("source edit budgets differ")
    selected = {candidate[0]: rt, control[0]: ordinary}
    for sid, ordinal in selected.items():
        tweet = row["user_tweets"][ordinal]
        actual = SourceRecord(
            sid, uid, "raw_tweet",
            {"tweet_id": str(tweet.get("id_str") or tweet.get("id")),
             "tweet_sha256": digest(tweet),
             "starts_rt": (tweet.get("text") or "").startswith("RT ")},
            f"{RAW.resolve()}#line={line};tweet={ordinal}")
        recorded = report["source_records"][sid]
        if (recorded["raw_location"] != actual.raw_location or
                recorded["payload"] != actual.payload or
                recorded["sha256"] != actual.sha256):
            raise ValueError(f"raw tweet source differs: {sid}")
        if not no_model_edge(tweet, uid, set(ids)):
            raise ValueError("tweet removal changes the model graph")
    if not report["source_records"][candidate[0]]["payload"]["starts_rt"]:
        raise ValueError("candidate is not an RT-prefixed tweet")
    if report["source_records"][control[0]]["payload"]["starts_rt"]:
        raise ValueError("control is not an ordinary tweet")
    if cert["claim"]["operation"] != "field" or \
            cert["claim"]["recomputed_value"] is not True:
        raise ValueError("source claim differs")

    frame = pd.read_csv(FEATURES)
    raw = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
    index = ids.index(uid)
    baseline_facts = tweet_values(row["user_tweets"])
    for field in TWEET_FIELDS:
        if not np.isclose(np.float32(baseline_facts[field]),
                          raw[index, FEATURE_FIELDS.index(field)], rtol=1e-6, atol=1e-5):
            raise ValueError(f"raw behavior fact differs from frozen table: {field}")
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
        excluded = {selected[sid] for sid in removed}
        tweets = [tweet for ordinal, tweet in enumerate(row["user_tweets"])
                  if ordinal not in excluded]
        facts = tweet_values(tweets)
        values = raw.copy()
        for name in TWEET_FIELDS:
            values[index, FEATURE_FIELDS.index(name)] = facts[name]
        updated = torch.from_numpy(
            np.clip(scaler.transform(values), -3, 3).astype(np.float32))
        built = VeriBotGraphInput(
            x_dict={"user": updated}, edge_index_dict=graph,
            relnet_edges=pack["relnet_edges"], account_ids=tuple(ids),
            user_fact_names=tuple(FEATURE_FIELDS), source_dependencies={},
            graph_rule_sha256=sha256(pack["contract"]["graph_rule"].encode()).hexdigest())
        return (direct_logits(hgt, relnet, updated, graph,
                              pack["relnet_edges"], index),
                tensor_hash(updated), facts, built.fingerprint())

    baseline_output, baseline_hash, before_facts, before_fingerprint = replay(())
    after_output, after_hash, after_facts, after_fingerprint = replay((candidate[0],))
    control_output, control_hash, control_facts, _ = replay((control[0],))
    if baseline_hash != contract["feature_sha256"]:
        raise ValueError("independent raw facts do not rebuild frozen input")
    if (before_fingerprint != cert["before"]["model_input_sha256"] or
            after_fingerprint != cert["after"]["model_input_sha256"]):
        raise ValueError("source-edited complete input fingerprint differs")
    for sid, facts in ((candidate[0], after_facts), (control[0], control_facts)):
        changed = [name for name in TWEET_FIELDS
                   if np.float32(facts[name]) != np.float32(before_facts[name])]
        expected_paths = [f"user[{uid}].{name}" for name in changed]
        if list(report["source_input_paths"][sid]) != expected_paths:
            raise ValueError("declared source-to-input path differs")
    maximum = 0.0
    for name, facts in (("before", before_facts), ("after", after_facts),
                        ("control", control_facts)):
        for field in TWEET_FIELDS:
            difference = abs(float(facts[field]) - report["operation_facts"][name][field])
            maximum = max(maximum, difference)
            if difference > 1e-6:
                raise ValueError(f"recomputed behavior differs: {name}/{field}")
    for stage, actual in (("before", baseline_output), ("after", after_output)):
        for key in ("hgt_logit", "relnet_logit", "fused_logit"):
            difference = abs(actual[key] - cert[stage][key])
            maximum = max(maximum, difference)
            if difference > 1e-5:
                raise ValueError(f"frozen prediction differs: {stage}/{key}")
    for branch in ("hgt", "relnet", "fused"):
        key = f"{branch}_logit"
        difference = abs(control_output[key] - baseline_output[key] -
                         cert["controls"][0][f"{branch}_delta_logit"])
        maximum = max(maximum, difference)
        if difference > 1e-5:
            raise ValueError(f"control effect differs: {branch}")
    if after_hash == baseline_hash or control_hash == baseline_hash:
        raise ValueError("tweet removal did not change classifier input")
    saved = json.loads((args.run / "test_predictions.json").read_text(encoding="utf-8"))
    prediction = next(item for item in saved["predictions"] if item["account_id"] == uid)
    if any(abs(prediction[name] - baseline_output[name]) > 1e-5
           for name in ("hgt_logit", "relnet_logit", "fused_logit")):
        raise ValueError("baseline does not match saved fixed-test prediction")
    result = {"status": "verified_raw_tweet_behavior_and_frozen_two_branch_replay",
              "certificate_sha256": sha256_file(args.certificate),
              "account_id": uid, "eligible_test_accounts": count,
              "candidate_source_id": candidate[0], "control_source_id": control[0],
              "graph_unchanged": True,
              "max_output_difference": maximum}
    destination = args.certificate.with_name(args.certificate.stem + "_verification.json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
