"""Optional API language organizer for fixed, source-linked evidence items.

The caller supplies the graph prediction and selected items. The API returns
only their order and a connector from a fixed vocabulary; this module renders
the original claims, operations, and source IDs without rewriting them.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


CONNECTORS = {"", "Also, ", "Separately, ", "In addition, "}


@dataclass(frozen=True)
class EvidenceItem:
    item_id: str
    claim: str
    operation: str
    source_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "EvidenceItem":
        item = cls(
            item_id=str(row["item_id"]),
            claim=str(row["claim"]),
            operation=str(row["operation"]),
            source_ids=tuple(str(v) for v in row["source_ids"]),
        )
        if not item.item_id or not item.claim or not item.operation or not item.source_ids:
            raise ValueError("Every evidence item needs an ID, claim, operation, and source IDs")
        if len(set(item.source_ids)) != len(item.source_ids):
            raise ValueError(f"Duplicate source ID in {item.item_id}")
        return item


def _validated_order(raw: str, item_ids: list[str]) -> tuple[list[str], list[str]]:
    result = json.loads(raw)
    order = result["order"]
    connectors = result["connectors"]
    if not isinstance(order, list) or sorted(order) != sorted(item_ids):
        raise ValueError("API order must contain every item exactly once")
    if not isinstance(connectors, list) or len(connectors) != len(order):
        raise ValueError("API connectors must align with items")
    if any(value not in CONNECTORS for value in connectors):
        raise ValueError("API connector is outside the fixed vocabulary")
    return order, connectors


def organize_evidence(
    prediction: str,
    evidence: Sequence[EvidenceItem],
    source_index: Mapping[str, Any],
    *,
    client: Any | None = None,
    model: str | None = None,
    use_api: bool = True,
) -> dict[str, Any]:
    """Render immutable account claims; an API may choose display order.

    The function never asks the API to classify, verify, or write factual text.
    It accepts any OpenAI-compatible client for direct integration.
    """
    if prediction not in {"bot", "human"}:
        raise ValueError("prediction must be the already fixed 'bot' or 'human' label")
    ids = [item.item_id for item in evidence]
    if len(set(ids)) != len(ids):
        raise ValueError("Evidence item IDs must be unique")
    for item in evidence:
        missing = set(item.source_ids).difference(source_index)
        if missing:
            raise ValueError(f"Unresolved source ID(s) in {item.item_id}: {sorted(missing)}")

    order = ids.copy()
    connectors = [""] + ["Also, "] * max(0, len(ids) - 1)
    if evidence and use_api:
        if client is None:
            from openai import OpenAI

            api_key = os.environ.get("EVIDENCE_LLM_API_KEY")
            if not api_key:
                raise ValueError("EVIDENCE_LLM_API_KEY is required for API organization")
            client = OpenAI(
                api_key=api_key,
                base_url=os.environ.get("EVIDENCE_LLM_BASE_URL") or None,
            )
        selected_model = model or os.environ.get("EVIDENCE_LLM_MODEL")
        if not selected_model:
            raise ValueError("EVIDENCE_LLM_MODEL is required for API organization")
        request = {
            "prediction_fixed": prediction,
            "items": [
                {"item_id": item.item_id, "claim": item.claim,
                 "operation": item.operation, "source_ids": list(item.source_ids)}
                for item in evidence
            ],
        }
        response = client.chat.completions.create(
            model=selected_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": (
                    "You organize an account report. The prediction and every evidence "
                    "item are fixed. Return JSON with 'order' containing every item_id "
                    "exactly once and 'connectors' of equal length. Each connector "
                    "must be exactly one of: '', 'Also, ', 'Separately, ', "
                    "'In addition, '. Do not classify, verify, rewrite claims, or add facts."
                )},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
            ],
        )
        raw = response.choices[0].message.content or ""
        order, connectors = _validated_order(raw, ids)

    by_id = {item.item_id: item for item in evidence}
    sentences = []
    for position, item_id in enumerate(order):
        item = by_id[item_id]
        citation = ", ".join(item.source_ids)
        sentences.append(f"{connectors[position]}{item.claim} [{citation}]")
    return {
        "prediction": prediction,
        "ordered_item_ids": order,
        "report": " ".join(sentences),
        "evidence": [
            {"item_id": item_id, "operation": by_id[item_id].operation,
             "source_ids": list(by_id[item_id].source_ids)}
            for item_id in order
        ],
    }


def main() -> None:
    import sys

    payload = json.load(sys.stdin)
    items = [EvidenceItem.from_mapping(row) for row in payload["evidence"]]
    result = organize_evidence(payload["prediction"], items, payload["source_index"])
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
