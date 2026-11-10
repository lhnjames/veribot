"""The AAAI VeriBot HGT/RelNet decision path with source-edit certificates.

This module keeps the submitted architecture's two graph branches and
logit-average fusion. It deliberately does not use attention, the auxiliary
HGT evidence head, or generated text as proof that an input influenced the
classification. A source claim is certified only after full input rebuilding
and branch-specific frozen forward comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Callable, Mapping, Protocol, Sequence

import torch
from torch.nn import functional as F

from .evidence_v2 import AtomicClaim, SourceRecord, recompute_claim


EdgeType = tuple[str, str, str]


def _tensor_hash(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    h = sha256()
    h.update(str(cpu.dtype).encode())
    h.update(str(tuple(cpu.shape)).encode())
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def _hash_collection(items: Mapping[object, torch.Tensor]) -> dict[str, str]:
    return {str(key): _tensor_hash(items[key]) for key in sorted(items, key=str)}


@dataclass(frozen=True)
class VeriBotGraphInput:
    """A fixed node inventory and two projections of the same source graph."""

    x_dict: Mapping[str, torch.Tensor]
    edge_index_dict: Mapping[EdgeType, torch.Tensor]
    relnet_edges: Mapping[str, torch.Tensor]
    account_ids: tuple[str, ...]
    user_fact_names: tuple[str, ...]
    source_dependencies: Mapping[str, Sequence[str]]
    graph_rule_sha256: str

    def validate(self) -> None:
        user = self.x_dict.get("user")
        if user is None or user.ndim != 2:
            raise ValueError("HGT requires a 2D user fact tensor")
        if user.shape != (len(self.account_ids), len(self.user_fact_names)):
            raise ValueError("every user input column must have a fact name")
        if len(set(self.account_ids)) != len(self.account_ids) or \
                len(set(self.user_fact_names)) != len(self.user_fact_names):
            raise ValueError("account IDs and fact names must be unique")
        if not all(self.user_fact_names) or not self.graph_rule_sha256:
            raise ValueError("named facts and graph rule fingerprint are required")
        if not self.edge_index_dict or not self.relnet_edges:
            raise ValueError("both original graph branches need edges")
        for name, values in self.x_dict.items():
            if values.ndim != 2 or not bool(torch.isfinite(values).all()):
                raise ValueError(f"node type {name} has invalid features")
        for key, edge in self.edge_index_dict.items():
            if len(key) != 3 or edge.dtype != torch.long or edge.ndim != 2 or edge.shape[0] != 2:
                raise ValueError(f"invalid HGT edge type {key}")
            source_type, _, target_type = key
            if source_type not in self.x_dict or target_type not in self.x_dict:
                raise ValueError(f"HGT edge {key} has a missing node type")
            if edge.numel() and (int(edge.min()) < 0 or
                                 int(edge[0].max()) >= len(self.x_dict[source_type]) or
                                 int(edge[1].max()) >= len(self.x_dict[target_type])):
                raise ValueError(f"HGT edge {key} leaves node inventory")
        for name, edge in self.relnet_edges.items():
            if edge.dtype == torch.long and edge.ndim == 2 and edge.shape[0] == 2:
                if edge.numel() and (int(edge.min()) < 0 or
                                     int(edge.max()) >= len(self.account_ids)):
                    raise ValueError(f"RelNet edge {name} leaves user inventory")
            elif edge.is_floating_point() and edge.shape == (
                    len(self.account_ids), len(self.account_ids)):
                if not bool(torch.isfinite(edge).all()) or bool((edge < 0).any()):
                    raise ValueError(f"RelNet adjacency {name} has invalid values")
                row_sums = edge.sum(dim=1)
                if bool(((row_sums > 1.00001) |
                         ((row_sums > 0) & (row_sums < 0.99999))).any()):
                    raise ValueError(f"RelNet adjacency {name} is not mean-normalized")
            else:
                raise ValueError(f"invalid RelNet relation representation {name}")

    def fingerprint(self) -> str:
        payload = {"x": _hash_collection(self.x_dict),
                   "hgt_edges": _hash_collection(self.edge_index_dict),
                   "relnet_edges": _hash_collection(self.relnet_edges),
                   "account_ids": self.account_ids,
                   "user_fact_names": self.user_fact_names,
                   "graph_rule_sha256": self.graph_rule_sha256}
        return sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class HGTBranch(Protocol):
    lin_dict: torch.nn.ModuleDict
    convs: Sequence[torch.nn.Module]
    classifier: torch.nn.Module
    dropout: float

    def eval(self): ...
    def __call__(self, x_dict, edge_index_dict): ...


class RelNetBranch(Protocol):
    input: torch.nn.Module
    layers: Sequence[torch.nn.Module]
    classifier: torch.nn.Module

    def eval(self): ...
    def __call__(self, user_x, edge_by_relation): ...


class GNNBranchAdapter(Protocol):
    """A model-specific forward path over the shared source-derived input."""

    name: str
    model_sha256: str

    def trace(self, built: VeriBotGraphInput, target: int) -> dict: ...


class HGTForwardAdapter:
    name = "hgt"

    def __init__(self, model: HGTBranch, model_sha256: str):
        self.model = model.eval()
        self.model_sha256 = model_sha256

    def trace(self, built: VeriBotGraphInput, target: int) -> dict:
        with torch.inference_mode():
            direct = self.model(built.x_dict, built.edge_index_dict)
            logits = direct["bot_logit"]
            embedding = direct["user_emb"]
            if logits.shape != (len(built.account_ids),) or \
                    embedding.shape[0] != len(built.account_ids):
                raise ValueError("HGT outputs do not match the account inventory")
            hidden = {node_type: F.gelu(self.model.lin_dict[node_type](values))
                      for node_type, values in built.x_dict.items()}
            stages = {"input_projection": _tensor_hash(hidden["user"][target])}
            values = {"input_projection": hidden["user"][target].detach().cpu().tolist()}
            for layer_number, conv in enumerate(self.model.convs, 1):
                previous = hidden
                updated = conv(hidden, built.edge_index_dict)
                hidden = {
                    node_type: F.dropout(F.gelu(current + previous[node_type]),
                                         p=self.model.dropout, training=self.model.training)
                    for node_type, current in updated.items() if current is not None
                }
                stage_name = f"hgt_layer_{layer_number}"
                stages[stage_name] = _tensor_hash(hidden["user"][target])
                values[stage_name] = hidden["user"][target].detach().cpu().tolist()
            replayed = self.model.classifier(hidden["user"]).squeeze(-1)
            error = max(float((replayed - logits).abs().max()),
                        float((hidden["user"] - embedding).abs().max()))
            if error > 1e-6:
                raise ValueError("HGT internal path differs from direct model output")
        return {"logit": float(logits[target]),
                "embedding_sha256": _tensor_hash(embedding[target]),
                "embedding_values": embedding[target].detach().cpu().tolist(),
                "stage_sha256": stages, "stage_values": values,
                "forward_replay_max_error": error}


class RelNetForwardAdapter:
    name = "relnet"

    def __init__(self, model: RelNetBranch, model_sha256: str):
        self.model = model.eval()
        self.model_sha256 = model_sha256

    def trace(self, built: VeriBotGraphInput, target: int) -> dict:
        with torch.inference_mode():
            user_x = built.x_dict["user"]
            hidden = F.gelu(self.model.input(user_x))
            stages = {"input_projection": _tensor_hash(hidden[target])}
            values = {"input_projection": hidden[target].detach().cpu().tolist()}
            for layer_number, layer in enumerate(self.model.layers, 1):
                hidden = layer(hidden, built.relnet_edges)
                stage_name = f"relation_layer_{layer_number}"
                stages[stage_name] = _tensor_hash(hidden[target])
                values[stage_name] = hidden[target].detach().cpu().tolist()
            replayed = self.model.classifier(hidden).squeeze(-1)
            direct = self.model(user_x, built.relnet_edges)
            error = float((replayed - direct).abs().max())
            if error > 1e-6 or direct.shape != (len(built.account_ids),):
                raise ValueError("RelNet internal path differs from direct model output")
        return {"logit": float(direct[target]),
                "stage_sha256": stages, "stage_values": values,
                "forward_replay_max_error": error}


class VeriBotDecision:
    """Fuse two independently traced graph branches by the original logit mean."""

    def __init__(self, hgt: HGTBranch, relnet: RelNetBranch,
                 *, hgt_sha256: str, relnet_sha256: str):
        self.hgt_adapter: GNNBranchAdapter = HGTForwardAdapter(hgt, hgt_sha256)
        self.relnet_adapter: GNNBranchAdapter = RelNetForwardAdapter(relnet, relnet_sha256)
        self.hgt = self.hgt_adapter.model
        self.relnet = self.relnet_adapter.model
        self.hgt_sha256 = hgt_sha256
        self.relnet_sha256 = relnet_sha256

    def trace(self, built: VeriBotGraphInput, account_id: str) -> dict:
        built.validate()
        target = built.account_ids.index(account_id)
        hgt = self.hgt_adapter.trace(built, target)
        relnet = self.relnet_adapter.trace(built, target)
        fused_logit = float(0.5 * (
            torch.tensor(hgt["logit"], dtype=torch.float32) +
            torch.tensor(relnet["logit"], dtype=torch.float32)))
        return {
            "account_id": account_id,
            "hgt_logit": hgt["logit"],
            "relnet_logit": relnet["logit"],
            "fused_logit": fused_logit,
            "hgt_user_embedding_sha256": hgt["embedding_sha256"],
            "hgt_user_embedding_values": hgt["embedding_values"],
            "hgt_stage_sha256": hgt["stage_sha256"],
            "hgt_stage_values": hgt["stage_values"],
            "hgt_forward_replay_max_error": hgt["forward_replay_max_error"],
            "relnet_stage_sha256": relnet["stage_sha256"],
            "relnet_stage_values": relnet["stage_values"],
            "relnet_forward_replay_max_error": relnet["forward_replay_max_error"],
            "model_input_sha256": built.fingerprint(),
        }


GraphBuilder = Callable[[Mapping[str, SourceRecord]], VeriBotGraphInput]


def certify_aaai_source_response(
    *, account_id: str, claim: AtomicClaim,
    sources: Mapping[str, SourceRecord], build_input: GraphBuilder,
    decision: VeriBotDecision,
    control_source_groups: Sequence[Sequence[str]] = (),
) -> dict:
    """Verify source facts, all graph inputs, both branches, and their fusion."""
    actual = recompute_claim(claim, sources)
    if isinstance(actual, (int, float)) and isinstance(claim.expected, (int, float)):
        agrees = abs(float(actual) - float(claim.expected)) <= claim.tolerance
    else:
        agrees = actual == claim.expected
    if not agrees:
        raise ValueError("claim does not match its cited raw source")
    baseline = build_input(sources)
    before = decision.trace(baseline, account_id)
    candidate = frozenset(claim.source_ids)
    if any(not baseline.source_dependencies.get(sid) for sid in candidate):
        raise ValueError("cited source has no declared path into either classifier")
    edited = build_input({sid: row for sid, row in sources.items() if sid not in candidate})
    if (baseline.account_ids != edited.account_ids or
            baseline.user_fact_names != edited.user_fact_names or
            set(baseline.x_dict) != set(edited.x_dict) or
            set(baseline.edge_index_dict) != set(edited.edge_index_dict) or
            set(baseline.relnet_edges) != set(edited.relnet_edges)):
        raise ValueError("source operation changed model schema or account inventory")
    after = decision.trace(edited, account_id)
    if before["model_input_sha256"] == after["model_input_sha256"]:
        raise ValueError("source operation left both classifier inputs unchanged")
    controls = []
    for group in control_source_groups:
        group = tuple(group)
        if len(group) != len(candidate) or len(set(group)) != len(group) or \
                set(group) & candidate:
            raise ValueError("control must use distinct sources with equal count")
        control_input = build_input({sid: row for sid, row in sources.items()
                                     if sid not in group})
        traced = decision.trace(control_input, account_id)
        controls.append({"source_ids": list(group),
                         "hgt_delta_logit": traced["hgt_logit"] - before["hgt_logit"],
                         "relnet_delta_logit": traced["relnet_logit"] - before["relnet_logit"],
                         "fused_delta_logit": traced["fused_logit"] - before["fused_logit"]})
    return {
        "account_id": account_id,
        "claim": {"source_ids": list(candidate), "operation": claim.operation,
                  "recomputed_value": actual},
        "source_locations": {sid: sources[sid].raw_location for sid in candidate},
        "source_hashes": {sid: sources[sid].sha256 for sid in candidate},
        "input_dependencies": {sid: list(baseline.source_dependencies[sid])
                               for sid in candidate},
        "hgt_sha256": decision.hgt_sha256,
        "relnet_sha256": decision.relnet_sha256,
        "graph_rule_sha256": baseline.graph_rule_sha256,
        "before": before,
        "after": after,
        "hgt_delta_logit": after["hgt_logit"] - before["hgt_logit"],
        "relnet_delta_logit": after["relnet_logit"] - before["relnet_logit"],
        "fused_delta_logit": after["fused_logit"] - before["fused_logit"],
        "hgt_user_embedding_l2_change": math.dist(
            before["hgt_user_embedding_values"], after["hgt_user_embedding_values"]),
        "hgt_stage_l2_change": {
            key: math.dist(before["hgt_stage_values"][key],
                           after["hgt_stage_values"][key])
            for key in before["hgt_stage_values"]},
        "relnet_stage_l2_change": {
            key: math.dist(before["relnet_stage_values"][key],
                           after["relnet_stage_values"][key])
            for key in before["relnet_stage_values"]},
        "controls": controls,
        "claim_scope": "source validity and frozen HGT/RelNet/fusion dependence",
    }
