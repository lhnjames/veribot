from __future__ import annotations

import json

from .schema import AccountRecord, EvidencePrediction
from .verifier import EvidenceVerifier


WEIGHTS = {"grounding": .35, "semantic": .25, "counterfactual": .20, "label": .15, "format": .05}


def score_prediction(record: AccountRecord, raw: str, semantic_judge=None, counterfactual_judge=None) -> dict[str, float]:
    try:
        obj = json.loads(raw)
        pred = EvidencePrediction.model_validate(obj)
        format_score = 1.0 if set(EvidencePrediction.model_fields).issubset(obj) else 0.5
    except Exception:
        return {**{k: 0.0 for k in WEIGHTS}, "total": -1.0}
    result = EvidenceVerifier(semantic_judge, counterfactual_judge).verify(record, pred)
    parts = {
        "grounding": float(result.anchor_grounded and result.graph_consistent),
        "semantic": float(result.semantically_supported is True),
        "counterfactual": float(result.counterfactually_used is True),
        "label": float(pred.label == record.label),
        "format": format_score,
    }
    hallucination = 1.0 if result.missing_anchors else 0.0
    contradiction = 1.0 if result.semantically_supported is False else 0.0
    parts["total"] = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS) - hallucination - contradiction
    return parts
