"""Independently rebuild raw BotSim memberships, inputs and two model outputs."""

from __future__ import annotations

import argparse
import ast
import csv
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
from train_aaai_veribot_botsim import load_inputs, USERS, FEATURES
from train_aaai_veribot_fox8 import sha256_file, tensor_hash
from verifiable_bot.aaai_dense_relnet import DenseGatedRelNet
from verifiable_bot.evidence_v2 import SourceRecord
from verifiable_bot.aaai_provenance import SCHEMA_VERSION, verify_code_manifest
from verifiable_bot.aaai_veribot_core import VeriBotGraphInput

RUN = ROOT / "output/aaai_veribot_botsim/seed0"
AUDIT = ROOT / "research/certificates/aaai_veribot_botsim_all_features_audit.json"


def raw_lists(records: dict) -> tuple[list[str], list[str], list[list[str]]]:
    with USERS.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [str(row["user_id"]) for row in rows]
    memberships = []
    for row in rows:
        parsed = ast.literal_eval(row["subreddit"] or "[]")
        if not isinstance(parsed, list) or len(set(parsed)) != len(parsed):
            raise ValueError("raw subreddit list is invalid")
        memberships.append(parsed)
    subreddits = sorted({sub for values in memberships for sub in values})
    for sid, record in records.items():
        uid = record["account_id"]
        row_index = ids.index(uid)
        subreddit = record["payload"]["subreddit"]
        position = memberships[row_index].index(subreddit)
        location = f"{USERS.resolve()}#row={row_index + 2};subreddit[{position}]"
        expected = SourceRecord(sid, uid, "subreddit_membership",
                                {"subreddit": subreddit}, location)
        if (record["raw_location"] != location or
                record["sha256"] != expected.sha256 or
                record["kind"] != expected.kind):
            raise ValueError(f"raw membership differs: {sid}")
    return ids, subreddits, memberships


def graph_from_lists(ids, subreddits, memberships, removed):
    sub_index = {sub: i for i, sub in enumerate(subreddits)}
    membership = torch.zeros((len(ids), len(subreddits)), dtype=torch.float32)
    pairs = []
    for i, values in enumerate(memberships):
        for sub in values:
            if (ids[i], sub) in removed:
                continue
            j = sub_index[sub]
            membership[i, j] = 1
            pairs.append((i, j))
    edge = torch.tensor(sorted(pairs), dtype=torch.long).T.contiguous()
    shared = (membership @ membership.T) >= 1
    shared.fill_diagonal_(False)
    adjacency = shared.float()
    adjacency /= adjacency.sum(dim=1, keepdim=True).clamp(min=1)
    graph = {
        ("user", "posted_in", "subreddit"): edge,
        ("subreddit", "rev_posted_in", "user"): edge.flip(0).contiguous(),
    }
    return graph, {"shared_subreddit": adjacency}


def direct_logits(hgt, relnet, x_dict, graph, relation, target):
    with torch.inference_mode():
        h = float(hgt(x_dict, graph)["bot_logit"][target])
        r = float(relnet(x_dict["user"], relation)[target])
    return {"hgt_logit": h, "relnet_logit": r, "fused_logit": (h + r) / 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", nargs="?", type=Path,
                        default=ROOT / "research/certificates/aaai_veribot_botsim_membership.json")
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = json.loads(args.certificate.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("certificate schema version differs")
    verify_code_manifest(ROOT, report.get("code_sha256"))
    audited = json.loads(AUDIT.read_text(encoding="utf-8"))
    if (audited["status"] != "verified_raw_to_all_named_features_and_frozen_input"
            or report["raw_users_sha256"] != sha256_file(USERS)
            or report["feature_file_sha256"] != sha256_file(FEATURES)
            or report["checkpoint_sha256"] != sha256_file(args.run / "model.pt")
            or report["source_audit_sha256"] != sha256_file(AUDIT)):
        raise ValueError("source, audit or checkpoint fingerprint differs")
    x_dict, _, split, ids, _, _, metadata, contract = load_inputs()
    records = report["source_records"]
    raw_ids, subreddits, memberships = raw_lists(records)
    if raw_ids != ids or subreddits != contract["graph_meta"]["subreddit_names"]:
        raise ValueError("independent raw account or entity inventory differs")
    cert = report["certificate"]
    account_id = cert["account_id"]
    target = ids.index(account_id)
    if target not in set(split["test"]):
        raise ValueError("certified account is outside fixed test split")
    candidate = tuple(cert["claim"]["source_ids"])
    control = tuple(cert["controls"][0]["source_ids"])
    if len(candidate) != len(control) or len(candidate) != 1 or set(candidate) & set(control):
        raise ValueError("membership source budgets differ")
    if (cert["claim"]["operation"] != "field" or
            cert["claim"]["recomputed_value"] !=
            records[candidate[0]]["payload"]["subreddit"] or
            cert["source_locations"][candidate[0]] !=
            records[candidate[0]]["raw_location"]):
        raise ValueError("raw membership claim differs")
    frame = pd.read_csv(FEATURES)
    raw_features = frame.to_numpy(dtype=np.float32)
    n_sub_col = list(frame.columns).index("n_subreddits")
    if raw_features[target, n_sub_col] != len(memberships[target]):
        raise ValueError("membership list and model node fact differ")
    saved = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(saved["input_contract"]["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(saved["input_contract"]["scaler_scale"], dtype=np.float64)
    scaler.n_features_in_ = len(frame.columns)
    hgt = VerifiableHGT(metadata, {name: value.shape[1]
                                   for name, value in x_dict.items()},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3)
    relnet = DenseGatedRelNet(x_dict["user"].shape[1], hidden_channels=128,
                              relation_names=("shared_subreddit",),
                              num_layers=2, dropout=0.3)
    hgt.load_state_dict(saved["model_state"]["hgt"])
    relnet.load_state_dict(saved["model_state"]["relnet"])
    hgt.eval()
    relnet.eval()

    def replay(removed_ids):
        excluded = {(records[sid]["account_id"], records[sid]["payload"]["subreddit"])
                    for sid in removed_ids}
        graph, relation = graph_from_lists(ids, subreddits, memberships, excluded)
        changed = raw_features.copy()
        changed[target, n_sub_col] -= len(excluded)
        user_x = torch.from_numpy(
            np.clip(scaler.transform(changed), -3, 3).astype(np.float32))
        return direct_logits(hgt, relnet,
                             {"user": user_x, "subreddit": x_dict["subreddit"]},
                             graph, relation, target), graph, relation, user_x

    before, baseline_graph, baseline_relation, baseline_x = replay(())
    if tensor_hash(baseline_x) != contract["user_input_sha256"]:
        raise ValueError("raw-rebuilt user features differ from training")
    if any(tensor_hash(edge) != contract["hgt_graph_sha256"][str(key)]
           for key, edge in baseline_graph.items()):
        raise ValueError("raw-rebuilt HGT graph differs from training")
    if tensor_hash(baseline_relation["shared_subreddit"]) != \
            contract["relnet_operator_sha256"]["shared_subreddit"]:
        raise ValueError("raw-rebuilt RelNet operator differs from training")
    after, after_graph, after_relation, after_x = replay(candidate)
    control_result, _, _, _ = replay(control)
    for state, graph_state, relation_state, user_state in (
            ("before", baseline_graph, baseline_relation, baseline_x),
            ("after", after_graph, after_relation, after_x)):
        built = VeriBotGraphInput(
            x_dict={"user": user_state, "subreddit": x_dict["subreddit"]},
            edge_index_dict=graph_state, relnet_edges=relation_state,
            account_ids=tuple(ids), user_fact_names=tuple(frame.columns),
            source_dependencies={},
            graph_rule_sha256=cert["graph_rule_sha256"])
        if built.fingerprint() != cert[state]["model_input_sha256"]:
            raise ValueError(f"{state} complete model input fingerprint differs")
    maximum = 0.0
    for state, actual in (("before", before), ("after", after)):
        for key in ("hgt_logit", "relnet_logit", "fused_logit"):
            difference = abs(actual[key] - cert[state][key])
            maximum = max(maximum, difference)
            if difference > 1e-5:
                raise ValueError(f"{state} {key} does not replay")
    for branch in ("hgt", "relnet", "fused"):
        difference = abs(control_result[f"{branch}_logit"] -
                         before[f"{branch}_logit"] -
                         cert["controls"][0][f"{branch}_delta_logit"])
        maximum = max(maximum, difference)
        if difference > 1e-5:
            raise ValueError(f"control {branch} does not replay")
    result = {"status": "verified_raw_membership_graph_feature_and_frozen_model_replay",
              "certificate_sha256": sha256_file(args.certificate),
              "account_id": account_id,
              "raw_membership_count": sum(map(len, memberships)),
              "subreddit_count": len(subreddits),
              "max_output_difference": maximum}
    destination = args.certificate.with_name(args.certificate.stem + "_verification.json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
