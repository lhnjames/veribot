"""Exact text-pair provenance diagnostic for the blind annotation packet."""

from __future__ import annotations

import difflib
import hashlib
import itertools
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "research/annotation_pilot_v1/rater_materials/blind_packet.jsonl"
PILOT = ROOT / "research/annotation_pilot_v1/claude_blind_pilot_12.json"
OUT = ROOT / "research/annotation_pilot_v1/literal_wording_overlap_audit.json"


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"\d+", " ", value)
    return " ".join(re.findall(r"[#@]?\w+", value))


def score_pair(a: str, b: str) -> tuple[float, float]:
    at, bt = set(a.split()), set(b.split())
    shared = at & bt
    jaccard = len(shared) / len(at | bt) if at or bt else 0.0
    sequence = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    return sequence, jaccard


def main() -> None:
    rows = [json.loads(line) for line in PACKET.read_text(
        encoding="utf-8").splitlines()]
    if len(rows) != 120:
        raise ValueError("blind packet population differs")
    results = []
    for row in rows:
        if row["concept"] != "templated_text_in_shown_sample":
            continue
        candidates = [(item["ref"], normalize(item["text"]))
                      for item in row["items"]
                      if not item["text"].lstrip().casefold().startswith("rt @")]
        valid = [(ref, value) for ref, value in candidates
                 if len(value) >= 25 and len(value.split()) >= 4]
        pairs = []
        for (ref_a, a), (ref_b, b) in itertools.combinations(valid, 2):
            sequence, jaccard = score_pair(a, b)
            if sequence >= 0.88 and jaccard >= 0.70:
                pairs.append({"refs": [ref_a, ref_b],
                              "sequence_ratio": sequence,
                              "token_jaccard": jaccard,
                              "normalized_sha256": [hashlib.sha256(x.encode()).hexdigest()
                                                    for x in (a, b)]})
        results.append({"blind_id": row["blind_id"], "dataset": row["dataset"],
                        "observable_high_wording_overlap": bool(pairs),
                        "support_pairs": pairs[:10],
                        "pair_count": len(pairs),
                        "eligible_shown_messages": len(valid)})
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    prior = {row["blind_id"]: row for row in pilot["judgments"]}
    pilot_comparison = []
    for row in results:
        if row["blind_id"] in prior:
            model = prior[row["blind_id"]]
            pilot_comparison.append({"blind_id": row["blind_id"],
                "literal_overlap": row["observable_high_wording_overlap"],
                "claude_model_only_answer": model["answer"],
                "same_binary_answer": ((model["answer"] == "yes") ==
                                       row["observable_high_wording_overlap"])
                if model["answer"] != "uncertain" else None})
    OUT.write_text(json.dumps({
        "scope": "shown_text_only_not_behavioral_semantic_gold",
        "packet_sha256": hashlib.sha256(PACKET.read_bytes()).hexdigest(),
        "rule": "exclude RT @; NFKC/casefold; remove URLs and digits; >=25 normalized characters and >=4 tokens each; sequence ratio >=0.88 and token Jaccard >=0.70",
        "account_labels_or_private_provenance_read": False,
        "tasks": len(results),
        "counts": {str(key): count for key, count in Counter(
            (row["dataset"], row["observable_high_wording_overlap"])
            for row in results).items()},
        "pilot_comparison": pilot_comparison,
        "results": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(results), "high_overlap": sum(
        row["observable_high_wording_overlap"] for row in results),
        "model_pilot_agreements": sum(
            row["same_binary_answer"] is True for row in pilot_comparison),
        "model_pilot_comparisons": len(pilot_comparison)},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
