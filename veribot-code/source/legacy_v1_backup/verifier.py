from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .schema import AccountRecord, EvidencePrediction, VerificationResult


SemanticJudge = Callable[[str, list[str]], bool]
CounterfactualJudge = Callable[[AccountRecord, EvidencePrediction], bool]


@dataclass
class EvidenceVerifier:
    semantic_judge: SemanticJudge | None = None
    counterfactual_judge: CounterfactualJudge | None = None

    def verify(self, record: AccountRecord, prediction: EvidencePrediction) -> VerificationResult:
        available = record.anchor_map()
        missing = [a for a in prediction.anchors if a not in available]
        grounded = bool(prediction.anchors) and not missing
        graph_anchors = [a for a in prediction.anchors if a.startswith("edge:")]
        graph_consistent = all(a in available for a in graph_anchors)
        if prediction.evidence_status == "insufficient" and not prediction.anchors:
            return VerificationResult(anchor_grounded=False, graph_consistent=True,
                                      status="insufficient", details={"reason": "honest abstention"})
        if not grounded or not graph_consistent:
            return VerificationResult(anchor_grounded=grounded, graph_consistent=graph_consistent,
                                      missing_anchors=missing, status="rejected")
        values = [json.dumps(available[a], ensure_ascii=False) for a in prediction.anchors]
        semantic = self.semantic_judge(prediction.core_claim, values) if self.semantic_judge else None
        counterfactual = self.counterfactual_judge(record, prediction) if self.counterfactual_judge else None
        verified = semantic is True and counterfactual is True
        return VerificationResult(anchor_grounded=True, graph_consistent=True,
                                  semantically_supported=semantic, counterfactually_used=counterfactual,
                                  status="verified" if verified else "insufficient",
                                  details={"requires_semantic": semantic is None,
                                           "requires_counterfactual": counterfactual is None})


def grounding_reward(record: AccountRecord, raw_prediction: str) -> float:
    try:
        prediction = EvidencePrediction.model_validate_json(raw_prediction)
    except Exception:
        return -1.0
    if prediction.evidence_status == "insufficient" and not prediction.anchors:
        return 0.25
    result = EvidenceVerifier().verify(record, prediction)
    return 1.0 if result.anchor_grounded and result.graph_consistent else -1.0
