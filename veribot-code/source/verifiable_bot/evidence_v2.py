"""Source-grounded evidence certificates for frozen graph classifiers.

The model adapter is responsible for rebuilding *all* input dependencies after
editing a source record. This module verifies claims and measures the local
model response; it never infers a real-world cause from an intervention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Callable, Mapping, Protocol, Sequence


def digest(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":"), default=str).encode("utf-8")
    return sha256(serialized).hexdigest()


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    account_id: str
    kind: str
    payload: Mapping[str, Any]
    raw_location: str

    @property
    def sha256(self) -> str:
        return digest({"source_id": self.source_id, "account_id": self.account_id,
                       "kind": self.kind, "payload": self.payload})


@dataclass(frozen=True)
class AtomicClaim:
    """A factual predicate with an executable operation and exact sources.

    Supported operations: ``field``, ``count``, ``sum``, ``mean``, ``ratio``.
    ``field`` requires one record and a key; numeric operations require a key
    for each listed record except count. ``ratio`` requires numerator and
    denominator fields from one record and uses no silent zero fallback.
    """

    source_ids: tuple[str, ...]
    operation: str
    expected: Any
    field: str | None = None
    denominator_field: str | None = None
    tolerance: float = 1e-9


class ModelAdapter(Protocol):
    model_name: str
    model_sha256: str
    preprocessor_sha256: str

    def input_dependencies(self, source_id: str) -> Sequence[str]: ...

    def predict_logits(self, account_id: str,
                       removed_source_ids: frozenset[str] = frozenset()) -> Sequence[float]: ...

    def input_sha256(self, removed_source_ids: frozenset[str] = frozenset()) -> str: ...


@dataclass(frozen=True)
class BuiltInputs:
    """Complete model inputs rebuilt from a specified set of source records."""

    value: Any
    dependencies: Mapping[str, Sequence[str]]
    sha256: str


class RebuildingAdapter:
    """Run a frozen model on complete inputs rebuilt after each source edit.

    ``build_inputs`` must rerun the dataset's feature extraction, scaling,
    graph construction, and derived graph statistics from the supplied raw
    records. Its fingerprint is placed in every evidence certificate.
    """

    def __init__(self, *, model_name: str, model_sha256: str,
                 preprocessor_sha256: str, sources: Mapping[str, SourceRecord],
                 build_inputs: Callable[[Mapping[str, SourceRecord]], BuiltInputs],
                 predict: Callable[[str, Any], Sequence[float]]):
        self.model_name = model_name
        self.model_sha256 = model_sha256
        self.preprocessor_sha256 = preprocessor_sha256
        self.sources = dict(sources)
        self.build_inputs = build_inputs
        self.predict = predict
        self._cache: dict[frozenset[str], BuiltInputs] = {}

    def _built(self, removed_source_ids: frozenset[str]) -> BuiltInputs:
        if removed_source_ids not in self._cache:
            if not removed_source_ids <= self.sources.keys():
                raise KeyError("source edit includes an unknown record")
            selected = {sid: record for sid, record in self.sources.items()
                        if sid not in removed_source_ids}
            self._cache[removed_source_ids] = self.build_inputs(selected)
        return self._cache[removed_source_ids]

    def input_dependencies(self, source_id: str) -> Sequence[str]:
        return self._built(frozenset()).dependencies.get(source_id, ())

    def input_sha256(self, removed_source_ids: frozenset[str] = frozenset()) -> str:
        return self._built(removed_source_ids).sha256

    def predict_logits(self, account_id: str,
                       removed_source_ids: frozenset[str] = frozenset()) -> Sequence[float]:
        return self.predict(account_id, self._built(removed_source_ids).value)


def recompute_claim(claim: AtomicClaim, sources: Mapping[str, SourceRecord]) -> Any:
    if not claim.source_ids or len(claim.source_ids) != len(set(claim.source_ids)):
        raise ValueError("claim requires unique, nonempty source IDs")
    records = [sources[source_id] for source_id in claim.source_ids]
    if claim.operation == "count":
        return len(records)
    if claim.operation == "field":
        if len(records) != 1 or claim.field is None:
            raise ValueError("field claim needs one source and one field name")
        return records[0].payload[claim.field]
    if claim.operation == "ratio":
        if len(records) != 1 or claim.field is None or claim.denominator_field is None:
            raise ValueError("ratio claim needs one source and two field names")
        numerator = float(records[0].payload[claim.field])
        denominator = float(records[0].payload[claim.denominator_field])
        if denominator == 0:
            raise ZeroDivisionError("ratio denominator is zero")
        return numerator / denominator
    if claim.operation in ("sum", "mean"):
        if claim.field is None:
            raise ValueError("numeric claim needs a field name")
        values = [float(record.payload[claim.field]) for record in records]
        return sum(values) if claim.operation == "sum" else sum(values) / len(values)
    raise ValueError(f"unsupported operation: {claim.operation}")


def _agrees(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if isinstance(actual, bool) or isinstance(expected, bool):
            return actual == expected
        return abs(float(actual) - float(expected)) <= tolerance
    return actual == expected


def _margin(logits: Sequence[float], reference_class: int) -> float:
    values = [float(x) for x in logits]
    if len(values) < 2 or not 0 <= reference_class < len(values):
        raise ValueError("model must return at least two class logits")
    import math
    if not all(math.isfinite(x) for x in values):
        raise ValueError("model returned non-finite logits")
    return values[reference_class] - max(value for k, value in enumerate(values)
                                         if k != reference_class)


def audit_candidate(account_id: str, claim: AtomicClaim,
                    sources: Mapping[str, SourceRecord], adapter: ModelAdapter,
                    control_source_ids: Sequence[str] = (),
                    control_protocol: str = "none") -> dict:
    """Check the claim, then probe the model with source-level removal.

    Controls are selected outside this function; their selection rule is
    recorded, not inferred. They are re-probed with the same model and account.
    The function returns an explicit failure rather than silently treating a
    mismatch as verified evidence.
    """
    try:
        if any(sources[sid].account_id != account_id for sid in claim.source_ids):
            raise ValueError("claim cites a source owned by another account")
        actual = recompute_claim(claim, sources)
        claim_valid = _agrees(actual, claim.expected, claim.tolerance)
        if not claim_valid:
            raise ValueError(f"claim mismatch: expected {claim.expected!r}, got {actual!r}")
        dependencies = {sid: list(adapter.input_dependencies(sid)) for sid in claim.source_ids}
        if any(not paths for paths in dependencies.values()):
            raise ValueError("at least one cited source has no model input mapping")
        original = list(adapter.predict_logits(account_id))
        label = max(range(len(original)), key=original.__getitem__)
        edited = list(adapter.predict_logits(account_id, frozenset(claim.source_ids)))
        input_before = adapter.input_sha256()
        input_after = adapter.input_sha256(frozenset(claim.source_ids))
        if input_before == input_after:
            raise ValueError("source edit did not change model input fingerprint")
        original_margin, edited_margin = _margin(original, label), _margin(edited, label)
        delta = original_margin - edited_margin
        controls = []
        for sid in control_source_ids:
            if sid in claim.source_ids:
                raise ValueError("control source overlaps candidate")
            if sources[sid].account_id != account_id:
                raise ValueError("control source belongs to another account")
            if not adapter.input_dependencies(sid):
                raise ValueError("control source has no model input mapping")
            control_logits = list(adapter.predict_logits(account_id, frozenset((sid,))))
            controls.append({"source_id": sid,
                             "delta_margin": original_margin - _margin(control_logits, label)})
        graph = {
            "nodes": [
                *({"id": sid, "type": "Source", "sha256": sources[sid].sha256,
                   "raw_location": sources[sid].raw_location} for sid in claim.source_ids),
                {"id": "computed_value", "type": "Artifact", "value": actual,
                 "operation": claim.operation},
                {"id": "claim", "type": "Claim", "expected": claim.expected},
                {"id": "model_input", "type": "Input", "dependencies": dependencies},
                {"id": "model_response", "type": "Probe", "delta_margin": delta},
            ],
            "edges": [
                *({"from": sid, "to": "computed_value", "type": "compute"}
                  for sid in claim.source_ids),
                *({"from": sid, "to": "model_input", "type": "map-to-input"}
                  for sid in claim.source_ids),
                {"from": "computed_value", "to": "claim", "type": "ground"},
                {"from": "model_input", "to": "model_response", "type": "probe"},
            ],
        }
        return {
            "status": "checked", "account_id": account_id,
            "claim": asdict(claim), "recomputed_value": actual,
            "source_hashes": {sid: sources[sid].sha256 for sid in claim.source_ids},
            "model": adapter.model_name, "model_sha256": adapter.model_sha256,
            "preprocessor_sha256": adapter.preprocessor_sha256,
            "input_dependencies": dependencies, "reference_class": label,
            "logits_before": original, "logits_after": edited,
            "input_sha256_before": input_before, "input_sha256_after": input_after,
            "delta_margin": delta,
            "source_edit": {"operation": "remove", "source_ids": list(claim.source_ids)},
            "prediction_flipped": max(range(len(edited)), key=edited.__getitem__) != label,
            "control_protocol": control_protocol,
            "control_probes": controls, "evidence_graph": graph,
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {"status": "rejected", "account_id": account_id,
                "claim": asdict(claim), "reason": str(exc)}
