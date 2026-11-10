"""Compare target edge deletion with every other seven-source native edge."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/scripts"))
from audit_hypbot_fox8_raw_edge_response import (  # noqa: E402
    DATA, EVENT_INDEX, RAW, RAW_AUDIT, REPO, RUN,
)
from export_hypbot_native_graph_paired import native_symmetric_graph  # noqa: E402
from train_v2_detector import file_sha256, load_data  # noqa: E402

TARGET = "415566147"
TARGET_NEIGHBOR = "25073877"
SOURCE_CERT = ROOT / "research/certificates/hypbot_fox8_seven_pointer_minimality.json"
MULTIPLICITY_CERT = ROOT / "research/certificates/fox8_graph_edge_raw_source_multiplicity.json"
OUT = ROOT / "research/certificates/hypbot_fox8_seven_source_pair_control.json"


def main() -> None:
    sys.path.insert(0, str(REPO.resolve()))
    from hypbot.data import mean_adjacency
    from hypbot.model import HypBot
    from train import PRESETS

    torch.set_num_threads(4)
    source_cert = json.loads(SOURCE_CERT.read_text(encoding="utf-8"))
    multiplicity = json.loads(MULTIPLICITY_CERT.read_text(encoding="utf-8"))
    event_index = json.loads(EVENT_INDEX.read_text(encoding="utf-8"))
    raw_audit = json.loads(RAW_AUDIT.read_text(encoding="utf-8"))
    frozen = json.loads((RUN / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = frozen["manifest"]
    checkpoint_path = RUN / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    args = argparse.Namespace(dataset="fox8", data_dir=DATA, account_file=None,
                              raw_source=[RAW], knn_k=10)
    x, _, split, ids, _, _, hashes, _ = load_data(args, 0)
    edge, graph_meta = native_symmetric_graph(ROOT, "fox8", ids)
    if (source_cert["raw_source_sha256"] != file_sha256(RAW)
            or source_cert["source_index_sha256"] != file_sha256(EVENT_INDEX)
            or source_cert["model_sha256"] != file_sha256(checkpoint_path)
            or source_cert["graph_sha256"] != graph_meta["graph_sha256"]
            or multiplicity["graph_sha256"] != graph_meta["graph_sha256"]
            or multiplicity["raw_source_sha256"] != file_sha256(RAW)
            or raw_audit["raw_source_sha256"] != file_sha256(RAW)
            or raw_audit["event_index_sha256"] != file_sha256(EVENT_INDEX)
            or hashes["feature_sha256"] != manifest["node_feature_sha256"]
            or hashes["split_sha256"] != manifest["split_sha256"]
            or checkpoint["split_sha256"] != hashes["split_sha256"]
            or graph_meta != manifest["native_graph"]):
        raise ValueError("Frozen source/model contract differs")
    index = {uid: i for i, uid in enumerate(ids)}
    target = index[TARGET]
    if target not in set(map(int, split["test"])):
        raise ValueError("Target is not in saved test split")
    pair_events = defaultdict(list)
    for event in event_index["events"]:
        a, b = event["source_id"], event["target_id"]
        if a in index and b in index and a != b:
            pair_events[tuple(sorted((a, b)))].append(event)
    seven_pairs = sorted(pair for pair, events in pair_events.items()
                         if sum(len(e["locators"]) for e in events) == 7)
    if len(seven_pairs) != 18 or tuple(sorted((TARGET, TARGET_NEIGHBOR))) not in seven_pairs:
        raise ValueError("Seven-source comparison population differs")
    seven_from_multiplicity = {tuple(r["undirected_graph_pair"])
                               for r in multiplicity["all_multi_pointer_pairs"]
                               if r["raw_tweet_pointer_count"] == 7}
    if set(seven_pairs) != seven_from_multiplicity:
        raise ValueError("Seven-source pairs differ from independent graph audit")
    neighbors = defaultdict(set)
    for u, v in edge.T.tolist():
        neighbors[int(u)].add(int(v))
    one_hop = neighbors[target]
    two_hop = {node for neighbor in one_hop for node in neighbors[neighbor]}
    model = HypBot(x.shape[1], K=manifest["num_sectors"], gated=False,
                   **PRESETS["fox8"]["model"], graph_width=x.shape[1],
                   sparse_adj=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        baseline = model(x, mean_adjacency(edge, len(ids))).cpu().numpy()
        baseline_graph_embedding = model.aggregate(x, mean_adjacency(edge, len(ids)))
        original_radius_std = float(model.enc_graph(baseline_graph_embedding)[0].std())
        original_node_readout, _ = model._readout(model.enc_node, model.warp_node,
                                                 model.sectors_node, x)

        def fixed_scale_logits(graph_embedding):
            radius, unit = model.enc_graph(graph_embedding)
            gamma = model.warp_graph(unit)
            entropy, alignment, _ = model.sectors_graph(unit, gamma)
            r_norm = radius / (original_radius_std + 1e-6)
            h_norm = entropy / (math.log(model.K) + 1e-6)
            graph_readout = torch.stack([
                r_norm, h_norm, alignment, r_norm * alignment,
                gamma * alignment], dim=-1)
            return model.head(torch.cat((original_node_readout, graph_readout),
                                        dim=-1)).squeeze(-1).cpu().numpy()

        if np.max(np.abs(fixed_scale_logits(baseline_graph_embedding) - baseline)) > 1e-6:
            raise ValueError("Fixed-scale baseline differs")
    original_logit = float(baseline[target])
    if abs(original_logit - source_cert["frozen_target_logit_by_graph_state"]["pair_present"]) > 1e-6:
        raise ValueError("Original target logit differs")
    threshold = float(manifest["threshold"])
    original_class = int(1 / (1 + np.exp(-original_logit)) >= threshold)
    rows = []
    for pair in seven_pairs:
        vertices = tuple(sorted((index[pair[0]], index[pair[1]])))
        keep = torch.as_tensor([
            tuple(sorted((int(u), int(v)))) != vertices
            for u, v in edge.T.tolist()], dtype=torch.bool)
        if int((~keep).sum()) != 2:
            raise ValueError("Pair is not one symmetric graph edge")
        with torch.no_grad():
            new_edge = edge[:, keep]
            graph_embedding = model.aggregate(x, mean_adjacency(new_edge, len(ids)))
            edited = float(model.forward_features(x, graph_embedding)[0][target].item())
            fixed = float(fixed_scale_logits(graph_embedding)[target])
            local_embedding_delta = float(torch.max(torch.abs(
                graph_embedding[target] - baseline_graph_embedding[target])).item())
        probability = float(1 / (1 + np.exp(-edited)))
        fixed_probability = float(1 / (1 + np.exp(-fixed)))
        dist = min(0 if index[u] == target else 1 if index[u] in one_hop
                   else 2 if index[u] in two_hop else 3 for u in pair)
        rows.append({
            "undirected_graph_pair": list(pair),
            "source_pointer_count": 7,
            "event_record_count": len(pair_events[pair]),
            "target_incident": TARGET in pair,
            "minimum_endpoint_graph_distance_from_target_capped_at_3": dist,
            "edited_target_logit": edited,
            "edited_target_probability": probability,
            "edited_target_class": int(probability >= threshold),
            "target_logit_delta": edited - original_logit,
            "fixed_population_scale_target_logit": fixed,
            "fixed_population_scale_target_class": int(fixed_probability >= threshold),
            "target_graph_embedding_max_abs_change": local_embedding_delta,
        })
    own = next(r for r in rows if r["target_incident"])
    other = [r for r in rows if not r["target_incident"]]
    if (len(other) != 17
            or abs(own["edited_target_logit"] -
                   source_cert["frozen_target_logit_by_graph_state"]["pair_absent"]) > 1e-6
            or own["edited_target_class"] == original_class):
        raise ValueError("Own seven-source edge does not replay expected class flip")
    report = {
        "scope": "all 18 Fox8 native graph pairs with exactly seven independently audited raw tweet pointers, evaluated for target 415566147",
        "operation": "clear all seven target-user pointers supporting one pair, retain node facts, remove its two symmetric model arcs, evaluate frozen GNN",
        "not_an_observed_source_correction": True,
        "official_paper_hypbot_graph_configuration": False,
        "target_account_id": TARGET,
        "raw_source_sha256": file_sha256(RAW),
        "event_index_sha256": file_sha256(EVENT_INDEX),
        "full_raw_audit_sha256": file_sha256(RAW_AUDIT),
        "graph_sha256": graph_meta["graph_sha256"],
        "model_sha256": file_sha256(checkpoint_path),
        "source_certificate_sha256": file_sha256(SOURCE_CERT),
        "multiplicity_certificate_sha256": file_sha256(MULTIPLICITY_CERT),
        "original_target_logit": original_logit,
        "original_target_class": original_class,
        "threshold": threshold,
        "seven_pointer_pair_count": len(rows),
        "other_seven_pointer_pair_count": len(other),
        "other_pair_target_class_flips": sum(r["edited_target_class"] != original_class
                                             for r in other),
        "other_pair_fixed_scale_target_class_flips": sum(
            r["fixed_population_scale_target_class"] != original_class
            for r in other),
        "other_pair_abs_target_logit_delta_max": max(
            abs(r["target_logit_delta"]) for r in other),
        "own_pair_abs_target_logit_delta": abs(own["target_logit_delta"]),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
