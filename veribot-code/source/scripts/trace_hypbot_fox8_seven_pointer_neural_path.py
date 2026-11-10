"""Trace one source-backed native graph edit through frozen HypBot layers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/scripts"))
sys.path.insert(0, str(ROOT / "source"))
from audit_hypbot_fox8_raw_edge_response import (  # noqa: E402
    DATA, EVENT_INDEX, RAW, REPO, RUN,
)
from export_hypbot_native_graph_paired import native_symmetric_graph  # noqa: E402
from train_v2_detector import file_sha256, load_data  # noqa: E402


SOURCE = ROOT / "research/certificates/hypbot_fox8_seven_pointer_minimality.json"
RESPONSE = ROOT / "research/certificates/hypbot_fox8_account_relation_source_response.json"
OUT = ROOT / "research/certificates/hypbot_fox8_seven_pointer_neural_path.json"
FEATURE_LABELS = (
    "normalized_radius", "normalized_sector_entropy", "max_alignment",
    "normalized_radius_times_alignment", "warped_alignment",
)


def norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value).item())


def main() -> None:
    sys.path.insert(0, str(REPO.resolve()))
    from hypbot.data import mean_adjacency
    from hypbot.model import HypBot
    from train import PRESETS

    torch.set_num_threads(4)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    response = json.loads(RESPONSE.read_text(encoding="utf-8"))
    frozen = json.loads((RUN / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = frozen["manifest"]
    checkpoint_path = RUN / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    args = argparse.Namespace(dataset="fox8", data_dir=DATA, account_file=None,
                              raw_source=[RAW], knn_k=10)
    x, _, split, ids, _, _, hashes, _ = load_data(args, 0)
    edge, graph_meta = native_symmetric_graph(ROOT, "fox8", ids)
    if (source["model_sha256"] != file_sha256(checkpoint_path) or
            response["model_sha256"] != file_sha256(checkpoint_path) or
            source["raw_source_sha256"] != file_sha256(RAW) or
            source["source_index_sha256"] != file_sha256(EVENT_INDEX) or
            source["graph_sha256"] != graph_meta["graph_sha256"] or
            graph_meta != manifest["native_graph"] or
            hashes["feature_sha256"] != manifest["node_feature_sha256"] or
            hashes["split_sha256"] != manifest["split_sha256"] or
            source["source_response_sha256"] != file_sha256(RESPONSE)):
        raise ValueError("Frozen source/model/graph contract changed")
    uid = source["account_id"]
    neighbor = source["neighbor_id"]
    index = {name: i for i, name in enumerate(ids)}
    target = index[uid]
    if target not in set(map(int, split["test"])):
        raise ValueError("Source account no longer belongs to saved test partition")
    pair = tuple(sorted((target, index[neighbor])))
    keep = torch.as_tensor([
        tuple(sorted((int(u), int(v)))) != pair
        for u, v in edge.T.tolist()], dtype=torch.bool)
    if int((~keep).sum()) != 2 or source["source_pointers_supporting_pair"] != 7:
        raise ValueError("Seven original pointers no longer support one symmetric pair")

    model = HypBot(x.shape[1], K=manifest["num_sectors"], gated=False,
                   **PRESETS["fox8"]["model"], graph_width=x.shape[1],
                   sparse_adj=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    adj_old = mean_adjacency(edge, len(ids))
    adj_new = mean_adjacency(edge[:, keep], len(ids))

    with torch.no_grad():
        h1_old = model.sage1_act(model.sage1(x, adj_old))
        h1_new = model.sage1_act(model.sage1(x, adj_new))
        h2_old = model.sage2_act(model.sage2(h1_old, adj_old))
        h2_new = model.sage2_act(model.sage2(h1_new, adj_new))
        direct_old = model(x, adj_old)
        direct_new = model(x, adj_new)
        node_features_old, _ = model._readout(
            model.enc_node, model.warp_node, model.sectors_node, x)
        node_features_new, _ = model._readout(
            model.enc_node, model.warp_node, model.sectors_node, x)
        graph_features_old, _ = model._readout(
            model.enc_graph, model.warp_graph, model.sectors_graph, h2_old)
        graph_features_new, _ = model._readout(
            model.enc_graph, model.warp_graph, model.sectors_graph, h2_new)
        radius_old, direction_old = model.enc_graph(h2_old)
        radius_new, direction_new = model.enc_graph(h2_new)
        global_std_old = radius_old.detach().std() + 1e-6
        global_std_new = radius_new.detach().std() + 1e-6

        def graph_readout_at_target(local: str, scale: str) -> torch.Tensor:
            radius, direction = ((radius_old, direction_old) if local == "old"
                                 else (radius_new, direction_new))
            std = global_std_old if scale == "old" else global_std_new
            gamma = model.warp_graph(direction[target:target+1])
            entropy, alignment, _ = model.sectors_graph(
                direction[target:target+1], gamma)
            k = model.K
            r_normalized = radius[target:target+1] / std
            h_normalized = entropy / (math.log(k) + 1e-6)
            return torch.stack((r_normalized, h_normalized, alignment,
                                r_normalized * alignment, gamma * alignment), dim=-1)

        states = {}
        for local in ("old", "new"):
            for scale in ("old", "new"):
                graph_features = graph_readout_at_target(local, scale)
                features = torch.cat((node_features_old[target:target+1],
                                      graph_features), dim=-1)
                logit = float(model.head(features).squeeze().item())
                probability = float(torch.sigmoid(torch.tensor(logit)).item())
                states[f"local_{local}_scale_{scale}"] = {
                    "logit": logit, "probability": probability,
                    "class": int(probability >= manifest["threshold"]),
                    "graph_readout": [float(v) for v in graph_features[0].tolist()],
                }

    old = states["local_old_scale_old"]["logit"]
    local_only = states["local_new_scale_old"]["logit"]
    scale_only = states["local_old_scale_new"]["logit"]
    both = states["local_new_scale_new"]["logit"]
    expected_old = source["frozen_target_logit_by_graph_state"]["pair_present"]
    expected_new = source["frozen_target_logit_by_graph_state"]["pair_absent"]
    maximum_error = max(
        abs(float(direct_old[target])-old),
        abs(float(direct_new[target])-both),
        abs(old-expected_old), abs(both-expected_new),
        float(torch.max(torch.abs(node_features_old-node_features_new)).item()),
        float(torch.max(torch.abs(graph_features_old[target]-
                                      graph_readout_at_target("old", "old")[0])).item()),
        float(torch.max(torch.abs(graph_features_new[target]-
                                      graph_readout_at_target("new", "new")[0])).item()),
    )
    if maximum_error > 1e-6 or states["local_old_scale_old"]["class"] != 0 or \
            states["local_new_scale_new"]["class"] != 1:
        raise ValueError("Manual neural-path states do not replay frozen model")
    report = {
        "status": "seven_source_pointer_hypbot_native_graph_neural_path_traced",
        "scope": "one previously selected Fox8 class-changing account in frozen native-graph HypBot variant",
        "source_operation": source["source_operation"],
        "account_id": uid,
        "neighbor_id": neighbor,
        "raw_source_pointer_count": 7,
        "removed_symmetric_graph_arcs": 2,
        "node_input_features_unchanged": True,
        "model_sha256": file_sha256(checkpoint_path),
        "graph_sha256": graph_meta["graph_sha256"],
        "node_feature_sha256": hashes["feature_sha256"],
        "source_certificate_sha256": file_sha256(SOURCE),
        "source_response_sha256": file_sha256(RESPONSE),
        "frozen_threshold": float(manifest["threshold"]),
        "target_sage1_embedding_l2_change": norm(h1_new[target]-h1_old[target]),
        "target_sage2_embedding_l2_change": norm(h2_new[target]-h2_old[target]),
        "target_node_readout_max_abs_change": float(torch.max(torch.abs(
            node_features_new[target]-node_features_old[target])).item()),
        "graph_radius_global_std_old": float(global_std_old),
        "graph_radius_global_std_new": float(global_std_new),
        "graph_readout_feature_order": list(FEATURE_LABELS),
        "four_path_states": states,
        "local_only_logit_change": local_only-old,
        "global_scale_only_logit_change": scale_only-old,
        "local_scale_interaction_logit": both-local_only-scale_only+old,
        "total_logit_change": both-old,
        "maximum_manual_to_frozen_logit_or_readout_error": maximum_error,
        "labels_loaded_by_data_loader_but_not_used_for_selection_or_analysis": True,
        "source_edit_is_known_correct_record_repair": False,
        "external_behavior_semantics_claimed": False,
        "interpretation": (
            "The seven raw pointers support one model edge pair. Removing every "
            "pointer changes the target's two GraphSAGE representations, graph "
            "channel readout, and final class. The 2x2 local-embedding/global-radius-"
            "scale states separate these two paths for this frozen operation. This "
            "does not establish a unique social-behavior cause or generalize to "
            "other GNN weights."),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print({"target": uid,
           "states": {name: (round(value["probability"], 6), value["class"])
                      for name, value in states.items()},
           "maximum_error": maximum_error})
    print(OUT)


if __name__ == "__main__":
    main()
