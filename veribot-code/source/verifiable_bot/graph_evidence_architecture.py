"""One source-to-forward contract for distinct frozen graph classifiers.

The classifier receives only named facts and a graph built by the caller. A
trace is a numerical path through the actual forward computation, not an
attention-based or language-generated explanation. Source validity is checked
before the model is queried; model dependence is measured after a complete
input rebuild. The builder remains responsible for raw-record and graph rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from torch.nn import functional as F

from .evidence_v2 import AtomicClaim, SourceRecord, recompute_claim


def tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    h = sha256()
    h.update(str(cpu.dtype).encode())
    h.update(str(tuple(cpu.shape)).encode())
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class GraphEvidenceInput:
    """All node columns must be named, source-computable facts."""

    facts: torch.Tensor
    fact_names: tuple[str, ...]
    graph: Any
    graph_sha256: str
    account_ids: tuple[str, ...]

    def validate(self) -> None:
        if self.facts.ndim != 2 or self.facts.shape != (len(self.account_ids), len(self.fact_names)):
            raise ValueError("fact names and account IDs must cover every model input cell")
        if len(set(self.fact_names)) != len(self.fact_names) or not all(self.fact_names):
            raise ValueError("fact names must be unique and nonempty")
        if len(set(self.account_ids)) != len(self.account_ids):
            raise ValueError("account IDs must be unique")
        if not bool(torch.isfinite(self.facts).all()):
            raise ValueError("model facts contain nonfinite values")
        if not self.graph_sha256:
            raise ValueError("graph fingerprint is required")


class GraphForwardAdapter(Protocol):
    name: str

    def trace(self, built: GraphEvidenceInput, target: int) -> dict: ...


def _target(value: torch.Tensor, index: int) -> dict:
    item = value.detach()[index].cpu()
    return {"shape": list(item.shape), "sha256": tensor_sha256(item),
            "values": item.reshape(-1).tolist()}


class HypBotForwardAdapter:
    """Replay the two GraphSAGE layers, both readouts, and classifier head."""

    name = "HypBot-native-graph"

    def __init__(self, model: torch.nn.Module):
        self.model = model.eval()

    def trace(self, built: GraphEvidenceInput, target: int) -> dict:
        built.validate()
        if not 0 <= target < len(built.account_ids):
            raise IndexError(target)
        with torch.inference_mode():
            x, adj = built.facts, built.graph
            h1 = self.model.sage1_act(self.model.sage1(x, adj))
            h2 = self.model.sage2_act(self.model.sage2(h1, adj))
            own, _ = self.model._readout(
                self.model.enc_node, self.model.warp_node,
                self.model.sectors_node, x)
            graph, _ = self.model._readout(
                self.model.enc_graph, self.model.warp_graph,
                self.model.sectors_graph, h2)
            reconstructed = self.model.head(torch.cat((own, graph), dim=-1)).squeeze(-1)
            direct = self.model(x, adj)
            error = float((reconstructed - direct).abs().max())
            if error > 1e-6:
                raise ValueError(f"HypBot forward trace differs from direct inference: {error}")
        return {
            "adapter": self.name,
            "logit": float(direct[target]),
            "forward_replay_max_error": error,
            "stages": {
                "named_facts": _target(x, target),
                "sage1": _target(h1, target),
                "sage2": _target(h2, target),
                "node_readout": _target(own, target),
                "graph_readout": _target(graph, target),
            },
        }


class EvidenceGatedForwardAdapter:
    """Replay the relation gate and additive head of EvidenceGatedDetector."""

    name = "EvidenceGated-relation-mean"

    def __init__(self, model: torch.nn.Module):
        self.model = model.eval()

    def trace(self, built: GraphEvidenceInput, target: int) -> dict:
        built.validate()
        if not 0 <= target < len(built.account_ids):
            raise IndexError(target)
        neighbor, degree = built.graph
        with torch.inference_mode():
            x = built.facts
            direct, components = self.model(x, neighbor, degree, return_components=True)
            own = self.model.node(x)
            relation = self.model.relation(neighbor)
            replay = components["node_logit"] + components["graph_gate"] * components["graph_logit"]
            error = float((replay - direct).abs().max())
            if error > 1e-6:
                raise ValueError(f"relation-gated forward trace differs from inference: {error}")
        return {
            "adapter": self.name,
            "logit": float(direct[target]),
            "forward_replay_max_error": error,
            "stages": {
                "named_facts": _target(x, target),
                "node_encoder": _target(own, target),
                "relation_encoder": _target(relation, target),
                "relation_weights": _target(components["relation_weights"], target),
                "node_logit": float(components["node_logit"][target]),
                "graph_logit": float(components["graph_logit"][target]),
                "graph_gate": float(components["graph_gate"][target]),
            },
        }


class GatedRelNetForwardAdapter:
    """Replay relation-specific message layers and the classifier head."""

    name = "GatedRelNet-native-relations"

    def __init__(self, model: torch.nn.Module):
        self.model = model.eval()

    def trace(self, built: GraphEvidenceInput, target: int) -> dict:
        built.validate()
        if not 0 <= target < len(built.account_ids):
            raise IndexError(target)
        with torch.inference_mode():
            h = F.gelu(self.model.input(built.facts))
            stages = {"named_facts": _target(built.facts, target),
                      "input_projection": _target(h, target)}
            for layer_number, layer in enumerate(self.model.layers, 1):
                h = layer(h, built.graph)
                stages[f"relation_layer_{layer_number}"] = _target(h, target)
            replay = self.model.classifier(h).squeeze(-1)
            direct = self.model(built.facts, built.graph)
            error = float((replay - direct).abs().max())
            if error > 1e-6:
                raise ValueError(f"relation GNN trace differs from direct inference: {error}")
        return {"adapter": self.name, "logit": float(direct[target]),
                "forward_replay_max_error": error, "stages": stages}


BuildGraph = Callable[[Mapping[str, SourceRecord]], GraphEvidenceInput]


def certify_source_response(
    *, account_id: str, claim: AtomicClaim,
    sources: Mapping[str, SourceRecord], build_graph: BuildGraph,
    adapter: GraphForwardAdapter, model_sha256: str,
    control_source_groups: Sequence[Sequence[str]] = (),
) -> dict:
    """Check a source fact and replay its effect through the actual graph model.

    A changed fact or hidden state alone is not labeled a changed decision.
    Every control removes the same number of source records as the candidate.
    """
    if not claim.source_ids:
        raise ValueError("claim requires at least one raw source")
    actual = recompute_claim(claim, sources)
    if isinstance(actual, (int, float)) and isinstance(claim.expected, (int, float)):
        matches = abs(float(actual) - float(claim.expected)) <= claim.tolerance
    else:
        matches = actual == claim.expected
    if not matches:
        raise ValueError(f"source claim mismatch: {actual!r} != {claim.expected!r}")
    baseline = build_graph(sources)
    edited = build_graph({k: v for k, v in sources.items() if k not in claim.source_ids})
    baseline.validate()
    edited.validate()
    if baseline.account_ids != edited.account_ids or baseline.fact_names != edited.fact_names:
        raise ValueError("source edit changed model input identity or feature order")
    target = baseline.account_ids.index(account_id)
    before = adapter.trace(baseline, target)
    after = adapter.trace(edited, target)
    changed_facts = [
        {"name": name, "before": float(baseline.facts[target, column]),
         "after": float(edited.facts[target, column])}
        for column, name in enumerate(baseline.fact_names)
        if float(baseline.facts[target, column]) != float(edited.facts[target, column])
    ]
    if (tensor_sha256(baseline.facts) == tensor_sha256(edited.facts)
            and baseline.graph_sha256 == edited.graph_sha256):
        raise ValueError("source edit did not alter a model input")
    controls = []
    for group in control_source_groups:
        group = tuple(group)
        if (len(group) != len(claim.source_ids) or len(set(group)) != len(group)
                or any(sid in claim.source_ids for sid in group)):
            raise ValueError("control must have equal source count and be distinct")
        control = build_graph({k: v for k, v in sources.items() if k not in group})
        control.validate()
        if control.account_ids != baseline.account_ids or control.fact_names != baseline.fact_names:
            raise ValueError("control changed model input identity or feature order")
        response = adapter.trace(control, target)
        controls.append({"source_ids": list(group), "logit": response["logit"],
                         "delta_logit": response["logit"] - before["logit"]})
    return {
        "account_id": account_id,
        "model_adapter": adapter.name,
        "model_sha256": model_sha256,
        "source_ids": list(claim.source_ids),
        "source_hashes": {sid: sources[sid].sha256 for sid in claim.source_ids},
        "source_locations": {sid: sources[sid].raw_location for sid in claim.source_ids},
        "operation": claim.operation,
        "recomputed_fact": actual,
        "fact_names": list(baseline.fact_names),
        "changed_target_facts": changed_facts,
        "facts_sha256_before": tensor_sha256(baseline.facts),
        "facts_sha256_after": tensor_sha256(edited.facts),
        "graph_sha256_before": baseline.graph_sha256,
        "graph_sha256_after": edited.graph_sha256,
        "before": before,
        "after": after,
        "delta_logit": after["logit"] - before["logit"],
        "controls": controls,
        "interpretation": "source validity and frozen-model numerical dependence only",
    }
