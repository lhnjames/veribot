"""R_format: fully programmatic, no model calls. Section 6.2(1) of the design spec."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from .prompting import PATTERN_BANNED_WORDS
from .schema import EvidenceOutput

MIN_EVIDENCE = 1
MAX_EVIDENCE = 8
MAX_CLAIMS = 8


@dataclass
class FormatCheck:
    ok: bool
    output: EvidenceOutput | None
    reason: str = ""


TIER_PATTERN = re.compile(
    r"\b(very high|very low|high-intensity burst|low-level burst|moderate burst|"
    r"extreme imbalance|strong imbalance|mild imbalance|mostly incoming|mostly outgoing|"
    r"mixed-direction|moderate|high|low)\b",
    re.IGNORECASE,
)


def check_format(
    raw_text: str, unit_texts: dict[str, str], hidden_metadata: dict[str, dict] | None = None,
) -> FormatCheck:
    """unit_texts maps every citable unit_id to its rendered text (e.g. units.unit_map(units)) --
    needed both for the existing "does this id exist" check and for the fabricated-number check
    below, which needs the actual source text, not just the id."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return FormatCheck(False, None, f"not valid json: {exc}")

    try:
        output = EvidenceOutput.model_validate(obj)
    except ValidationError as exc:
        return FormatCheck(False, None, f"schema mismatch: {exc}")

    valid_unit_ids = set(unit_texts)
    if not (MIN_EVIDENCE <= len(output.evidence_ids) <= MAX_EVIDENCE):
        return FormatCheck(False, None, "evidence_ids count out of budget")
    if len(output.evidence_ids) != len(set(output.evidence_ids)):
        return FormatCheck(False, None, "duplicate evidence_ids")
    unknown = [uid for uid in output.evidence_ids if uid not in valid_unit_ids]
    if unknown:
        return FormatCheck(False, None, f"evidence_ids reference unknown units: {unknown}")

    if not (1 <= len(output.claims) <= MAX_CLAIMS):
        return FormatCheck(False, None, "claims count out of budget")

    evidence_set = set(output.evidence_ids)
    for claim in output.claims:
        if not claim.source_ids:
            return FormatCheck(False, None, "a claim has no source_ids")
        if len(claim.source_ids) != len(set(claim.source_ids)):
            return FormatCheck(False, None, "duplicate source_ids within a claim")
        if not claim.claim.strip():
            return FormatCheck(False, None, "a claim has empty text")
        if not claim.pattern.strip():
            return FormatCheck(False, None, "a claim has empty pattern")
        if not set(claim.source_ids).issubset(evidence_set):
            return FormatCheck(False, None, "claim cites source_ids outside evidence_ids")
        # v3 model-visible numeric evidence is qualitative. Any Arabic digit in generated prose is
        # therefore either a fabricated raw value or an unwanted copy from verbatim text. Unit ids
        # are structural and remain allowed in source_ids; only claim/pattern prose is checked.
        if re.search(r"\d", claim.claim) or re.search(r"\d", claim.pattern):
            return FormatCheck(False, None, "claim or pattern contains an Arabic digit")

        # Programmatic category consistency: every tier phrase repeated by a claim must be one of
        # the authoritative descriptors attached to its cited hidden raw-value sidecar entries.
        if hidden_metadata is not None:
            stated = {match.group(1).lower() for match in TIER_PATTERN.finditer(claim.claim)}
            if stated:
                available = set()
                for source_id in claim.source_ids:
                    meta = hidden_metadata.get(source_id, {})
                    for key in ("bucket", "descriptor"):
                        if meta.get(key):
                            available.add(str(meta[key]).lower())
                    available.update(
                        match.group(1).lower() for match in TIER_PATTERN.finditer(unit_texts[source_id])
                    )
                mismatched = sorted(stated - available)
                if mismatched:
                    return FormatCheck(
                        False, None,
                        f"claim tier description disagrees with hidden source metadata: {mismatched}",
                    )
        # Same anti-leakage principle as units.py's BANNED_WORDS check on OBS units: pattern must
        # name a behavioral shape, not smuggle in the label/conclusion it's meant to stay silent on.
        lowered_pattern = claim.pattern.lower()
        leaked = [w for w in PATTERN_BANNED_WORDS if re.search(rf"\b{re.escape(w)}\b", lowered_pattern)]
        if leaked:
            return FormatCheck(False, None, f"claim pattern leaks label words: {leaked}")

    return FormatCheck(True, output)
