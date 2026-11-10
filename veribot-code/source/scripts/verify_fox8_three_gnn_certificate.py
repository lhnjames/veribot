"""Read Fox8 raw bytes again and check a three-GNN source certificate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
from verifiable_bot.evidence_v2 import SourceRecord


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", nargs="?", type=Path,
                        default=ROOT / "research/certificates/fox8_three_gnn_evidence_path.json")
    args = parser.parse_args()
    cert = json.loads(args.certificate.read_text(encoding="utf-8"))
    reports = cert["model_reports"]
    names = {r["model_adapter"] for r in reports}
    if names != {"HypBot-native-graph", "EvidenceGated-relation-mean",
                 "GatedRelNet-native-relations"} or len(reports) != 3:
        raise ValueError("certificate must contain the three distinct graph models")
    source_ids = {sid for report in reports for sid in report["source_ids"]}
    locations = {report["source_locations"][sid] for report in reports
                 for sid in report["source_ids"]}
    if len(source_ids) != 1 or len(locations) != 1:
        raise ValueError("this certificate must use one shared raw source")
    source_id = next(iter(source_ids))
    location = next(iter(locations))
    match = re.fullmatch(r"(.+)#line=(\d+);tweet=(\d+);user\.([A-Za-z_]+)", location)
    if match is None:
        raise ValueError("unsupported or malformed raw locator")
    raw, line_number, tweet_index, field = match.groups()
    raw_path = Path(raw)
    if not raw_path.is_file() or sha256_file(raw_path) != cert["raw_source_sha256"]:
        raise ValueError("raw source bytes differ from certificate")
    relation_index = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
    if sha256_file(relation_index) != cert["native_relation_source_sha256"]:
        raise ValueError("native relation source index differs from certificate")
    record = None
    opener = gzip.open if raw_path.suffix == ".gz" else open
    with opener(raw_path, "rt", encoding="utf-8") as stream:
        for current, line in enumerate(stream, 1):
            if current == int(line_number):
                record = json.loads(line)
                break
    if record is None:
        raise ValueError("raw locator line is absent")
    if str(record["user_id"]) != cert["account_id"]:
        raise ValueError("raw locator account differs")
    value = record["user_tweets"][int(tweet_index)]["user"][field]
    rebuilt = SourceRecord(source_id, cert["account_id"], "profile_field",
                           {field: value}, location)
    for report in reports:
        if report["source_hashes"][source_id] != rebuilt.sha256:
            raise ValueError("independent raw value differs from certificate")
        if report["recomputed_fact"] != value or report["operation"] != "field":
            raise ValueError("recomputed claim differs from raw field")
        name = report["model_adapter"]
        checkpoint_path = ROOT / {
            "HypBot-native-graph": "output/hypbot_fox8_native_graph_paired/seed0/model.pt",
            "EvidenceGated-relation-mean": "output/v2_fox8_full/seed0/model.pt",
            "GatedRelNet-native-relations": "output/fox8_rel_evidence_gnn/seed0/model.pt",
        }[name]
        if report["model_sha256"] != sha256_file(checkpoint_path):
            raise ValueError("frozen checkpoint bytes differ")
        facts = report["changed_target_facts"]
        if len(facts) != 1 or facts[0]["name"] != field or field not in report["fact_names"]:
            raise ValueError("raw field does not map to the reported named input")
        if name == "HypBot-native-graph":
            manifest = json.loads((checkpoint_path.parent / "test_predictions.json").read_text())
            if report["graph_sha256_before"] != manifest["manifest"]["native_graph"]["graph_sha256"] \
                    or report["graph_sha256_after"] != report["graph_sha256_before"]:
                raise ValueError("profile edit unexpectedly changed native graph")
            # Training split scaler has no saved sidecar in this checkpoint.
            # The exact scaled value is checked by the generating script's
            # full loader replay and saved-prediction comparison.
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            column = report["fact_names"].index(field)
            mean = float(np.asarray(checkpoint["scaler_mean"])[column])
            scale = float(np.asarray(checkpoint["scaler_scale"])[column])
            expected_before = float(np.clip((float(value) - mean) / scale, -3, 3))
            expected_after = float(np.clip((0.0 - mean) / scale, -3, 3))
            if abs(facts[0]["before"] - expected_before) > 1e-6 or \
                    abs(facts[0]["after"] - expected_after) > 1e-6:
                raise ValueError("raw source does not reproduce named model input")
            if name == "GatedRelNet-native-relations" and (
                    report["graph_sha256_before"] != checkpoint["native_graph_sha256"]
                    or report["graph_sha256_after"] != report["graph_sha256_before"]):
                raise ValueError("profile edit unexpectedly changed typed native graph")
        if abs(report["delta_logit"] -
               (report["after"]["logit"] - report["before"]["logit"])) > 1e-8:
            raise ValueError("logit response arithmetic differs")
        if report["before"]["forward_replay_max_error"] > 1e-6 or \
                report["after"]["forward_replay_max_error"] > 1e-6 or \
                report["saved_prediction_replay_error"] > 1e-6:
            raise ValueError("forward or saved prediction replay failed")
    print(json.dumps({"status": "independent_raw_locator_and_checkpoint_bytes_verified",
                      "account_id": cert["account_id"], "field": field,
                      "raw_value": value, "models": len(reports)}))


if __name__ == "__main__":
    main()
