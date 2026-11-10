"""Run one preselected raw profile fact through three frozen Fox8 graph models.

The target is the first account in the fixed seed-0 test partition, before
either model response is inspected. This is an integration check, not a claim
that the same source is influential across the test population.
"""

from __future__ import annotations

import argparse
import json
import math
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
sys.path.insert(0, str(Path("D:/Code/SAHG/SAHG/HypBot")))

from relnet_reimplementation import GatedRelNet
from export_hypbot_native_graph_paired import native_symmetric_graph
from train_v2_detector import file_sha256, hash_bytes, load_data
from verifiable_bot.evidence_v2 import AtomicClaim
from verifiable_bot.fox8_adapter import FEATURE_FIELDS, index_sources, raw_feature_matrix
from verifiable_bot.graph_evidence_architecture import (
    EvidenceGatedForwardAdapter, GatedRelNetForwardAdapter,
    GraphEvidenceInput, HypBotForwardAdapter,
    certify_source_response,
)
from verifiable_bot.fox8_relational_graph import typed_native_edges
from verifiable_bot.v2_detector import EvidenceGatedDetector, knn_edges, relation_means

from hypbot.data import mean_adjacency
from hypbot.model import HypBot
from train import PRESETS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "research/certificates/fox8_three_gnn_evidence_path.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    raw = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/fox8_23")
    loaded = argparse.Namespace(dataset="fox8", data_dir=data_dir,
                                account_file=None, raw_source=[raw], knn_k=10)
    x, _, split, ids, _, _, hashes, scaler = load_data(loaded, 0)
    uid = ids[int(split["test"][0])]
    index = ids.index(uid)
    sources = index_sources(raw, [uid])
    source_id = f"fox8:{uid}:profile:statuses_count"
    control_id = f"fox8:{uid}:profile:friends_count"
    if source_id not in sources or control_id not in sources:
        raise ValueError("preselected profile fields missing from raw record")
    raw_features = raw_feature_matrix(sources, [uid])[0]
    frame = pd.read_csv(data_dir / "features.csv")
    if list(frame.columns) != ["user_id", *FEATURE_FIELDS] or \
            ids != frame["user_id"].astype(str).tolist():
        raise ValueError("named model inputs differ from source feature schema")
    baseline_raw = frame[list(FEATURE_FIELDS)].to_numpy(dtype=np.float32)
    if not np.allclose(raw_features, baseline_raw[index], rtol=1e-5, atol=1e-5):
        raise ValueError("raw source does not rebuild target named facts")
    claim = AtomicClaim((source_id,), "field",
                        sources[source_id].payload["statuses_count"],
                        field="statuses_count")

    native_ckpt_path = ROOT / "output/hypbot_fox8_native_graph_paired/seed0/model.pt"
    native_saved = json.loads((native_ckpt_path.parent / "test_predictions.json").read_text())
    native_manifest = native_saved["manifest"]
    native_ckpt = torch.load(native_ckpt_path, map_location="cpu", weights_only=True)
    edge, graph_meta = native_symmetric_graph(ROOT, "fox8", ids)
    if file_sha256(native_ckpt_path) != native_manifest["model_sha256"] or \
            hashes["feature_sha256"] != native_manifest["node_feature_sha256"] or \
            hashes["split_sha256"] != native_manifest["split_sha256"] or \
            graph_meta != native_manifest["native_graph"]:
        raise ValueError("native HypBot frozen input contract changed")
    hypbot = HypBot(x.shape[1], K=native_manifest["num_sectors"], gated=False,
                    **PRESETS["fox8"]["model"], graph_width=x.shape[1],
                    sparse_adj=True)
    hypbot.load_state_dict(native_ckpt["model_state"])
    hypbot.eval()
    native_adj = mean_adjacency(edge, len(ids))

    v2_ckpt_path = ROOT / "output/v2_fox8_full/seed0/model.pt"
    v2_ckpt = torch.load(v2_ckpt_path, map_location="cpu", weights_only=False)
    v2 = EvidenceGatedDetector(
        v2_ckpt["feature_dim"], v2_ckpt["graph_dim"],
        v2_ckpt["num_relations"], v2_ckpt["hidden"], v2_ckpt["dropout"])
    v2.load_state_dict(v2_ckpt["state_dict"])
    v2.eval()
    if v2_ckpt["projection"] is not None:
        raise ValueError("this integration expects all named node facts in the relation channel")
    v2_scaler = StandardScaler()
    v2_scaler.mean_ = np.asarray(v2_ckpt["scaler_mean"])
    v2_scaler.scale_ = np.asarray(v2_ckpt["scaler_scale"])
    v2_scaler.n_features_in_ = len(FEATURE_FIELDS)
    base_v2 = torch.from_numpy(np.clip(
        v2_scaler.transform(baseline_raw), -3, 3).astype(np.float32))
    if hash_bytes(base_v2.numpy().tobytes()) != v2_ckpt["hashes"]["feature_sha256"]:
        raise ValueError("v2 named fact input differs from training")
    baseline_knn, baseline_relation = knn_edges(base_v2, 10)
    if hash_bytes(baseline_knn.numpy().tobytes(), baseline_relation.numpy().tobytes()) != \
            v2_ckpt["hashes"]["graph_sha256"]:
        raise ValueError("v2 graph differs from frozen training graph")
    rel_ckpt_path = ROOT / "output/fox8_rel_evidence_gnn/seed0/model.pt"
    rel_ckpt = torch.load(rel_ckpt_path, map_location="cpu", weights_only=False)
    typed, typed_hash = typed_native_edges(
        ROOT / "data/processed/fox8_v3/interaction_event_index.json", ids)
    if (rel_ckpt["feature_sha256"] != hashes["feature_sha256"]
            or rel_ckpt["split_sha256"] != hashes["split_sha256"]
            or rel_ckpt["native_graph_sha256"] != typed_hash
            or rel_ckpt["native_event_index_sha256"] != graph_meta["source_index_sha256"]
            or tuple(rel_ckpt["feature_names"]) != tuple(FEATURE_FIELDS)):
        raise ValueError("relation GNN frozen input contract changed")
    rel_model = GatedRelNet(
        rel_ckpt["input_dim"], hidden_channels=rel_ckpt["hidden_channels"],
        relation_names=tuple(rel_ckpt["relation_names"]),
        num_layers=rel_ckpt["num_layers"], dropout=rel_ckpt["dropout"])
    rel_model.load_state_dict(rel_ckpt["model_state"])
    rel_model.eval()
    rel_scaler = StandardScaler()
    rel_scaler.mean_ = np.asarray(rel_ckpt["scaler_mean"])
    rel_scaler.scale_ = np.asarray(rel_ckpt["scaler_scale"])
    rel_scaler.n_features_in_ = len(FEATURE_FIELDS)

    def updated_x(remaining, base, fitted_scaler):
        rebuilt = raw_feature_matrix(dict(remaining), [uid])[0]
        changed = base.clone()
        changed[index] = torch.from_numpy(np.clip(
            fitted_scaler.transform(rebuilt.reshape(1, -1))[0],
            -3, 3).astype(np.float32))
        return changed

    def build_native(remaining):
        facts = updated_x(remaining, x, scaler)
        return GraphEvidenceInput(facts, tuple(FEATURE_FIELDS), native_adj,
                                  graph_meta["graph_sha256"], tuple(ids))

    def build_v2(remaining):
        facts = updated_x(remaining, base_v2, v2_scaler)
        knn, relation = knn_edges(facts, 10)
        neighbor, degree = relation_means(facts, knn, relation, 1)
        graph_hash = hash_bytes(knn.numpy().tobytes(), relation.numpy().tobytes(),
                                neighbor.numpy().tobytes(), degree.numpy().tobytes())
        return GraphEvidenceInput(facts, tuple(FEATURE_FIELDS), (neighbor, degree),
                                  graph_hash, tuple(ids))

    def build_rel(remaining):
        facts = updated_x(remaining, x, rel_scaler)
        return GraphEvidenceInput(facts, tuple(FEATURE_FIELDS), typed,
                                  typed_hash, tuple(ids))

    if not torch.equal(build_native(sources).facts, x):
        raise ValueError("native no-edit source rebuild differs from frozen facts")
    if not torch.equal(build_v2(sources).facts, base_v2):
        raise ValueError("v2 no-edit source rebuild differs from frozen facts")
    if not torch.equal(build_rel(sources).facts, x):
        raise ValueError("relation GNN no-edit source rebuild differs from frozen facts")
    reports = [
        certify_source_response(
            account_id=uid, claim=claim, sources=sources,
            build_graph=builder, adapter=adapter,
            model_sha256=file_sha256(path), control_source_groups=((control_id,),))
        for builder, adapter, path in (
            (build_native, HypBotForwardAdapter(hypbot), native_ckpt_path),
            (build_v2, EvidenceGatedForwardAdapter(v2), v2_ckpt_path),
            (build_rel, GatedRelNetForwardAdapter(rel_model), rel_ckpt_path),
        )
    ]
    v2_saved = json.loads((v2_ckpt_path.parent / "test_predictions.json").read_text())
    rel_saved = json.loads((rel_ckpt_path.parent / "test_predictions.json").read_text())
    thresholds = (float(native_manifest["threshold"]),
                  float(v2_ckpt["threshold"]), 0.5)
    for report, saved, threshold in zip(
            reports, (native_saved, v2_saved, rel_saved), thresholds):
        row = next(item for item in saved["predictions"] if item["account_id"] == uid)
        probability = float(torch.sigmoid(torch.tensor(report["before"]["logit"])))
        if abs(probability - float(row["probability"])) > 1e-6:
            raise ValueError(f"{report['model_adapter']} does not replay saved prediction")
        report["saved_prediction_probability"] = float(row["probability"])
        report["saved_prediction_replay_error"] = abs(probability - float(row["probability"]))
        report["decision_threshold"] = threshold
        report["probability_before"] = probability
        report["probability_after"] = 1 / (1 + math.exp(-report["after"]["logit"]))
        report["class_before"] = int(report["probability_before"] >= threshold)
        report["class_after"] = int(report["probability_after"] >= threshold)
    certificate = {
        "status": "three_frozen_graph_models_traced_from_one_preselected_raw_fact",
        "scope": "first Fox8 seed-0 test account only; no population inference",
        "account_id": uid,
        "raw_source_sha256": file_sha256(raw),
        "native_relation_source_sha256": graph_meta["source_index_sha256"],
        "model_reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": certificate["status"], "account_id": uid,
                      "models": [{"name": r["model_adapter"],
                                  "delta_logit": r["delta_logit"],
                                  "control_delta_logit": r["controls"][0]["delta_logit"]}
                                 for r in reports]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
