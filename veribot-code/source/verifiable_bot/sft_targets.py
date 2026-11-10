"""Shared helper: serialize an SFT target JSON while tracking the character span of each
top-level field's value, so train_sft.py can apply the per-section loss weights
(lambda_select=1, lambda_y, lambda_c, lambda_s) from spec section 5 as a per-token weighted CE
loss over one ordinary autoregressive sequence, instead of four separate loss heads.
"""
from __future__ import annotations

import json


def serialize_target_with_spans(
    target: dict, lambda_y: float, lambda_c: float, lambda_s: float
) -> tuple[str, list[tuple[int, int, float]]]:
    parts: list[str] = []
    spans: list[tuple[int, int, float]] = []

    def add(piece: str, weight: float | None = None) -> None:
        start = sum(len(p) for p in parts)
        parts.append(piece)
        if weight is not None:
            spans.append((start, start + len(piece), weight))

    add('{"evidence_ids":')
    add(json.dumps(target["evidence_ids"], ensure_ascii=False), weight=1.0)
    add(',"agreement":')
    add(json.dumps(target["agreement"], ensure_ascii=False), weight=lambda_y)
    add(',"claims":')
    add(json.dumps(target["claims"], ensure_ascii=False), weight=lambda_c)
    add(',"evidence_status":')
    add(json.dumps(target["evidence_status"], ensure_ascii=False), weight=lambda_s)
    add("}")
    return "".join(parts), spans
