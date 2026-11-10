"""Validate blind annotation packets and completed independent rater files."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from .semantic_annotation import CONCEPTS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def validate_packet(packet: list[dict], provenance: list[dict]) -> dict:
    ids = [row.get("blind_id") for row in packet]
    source = {row.get("blind_id"): row for row in provenance}
    if not ids or len(set(ids)) != len(ids) or len(source) != len(provenance) or \
            set(ids) != set(source):
        raise ValueError("packet and source mapping have duplicate or missing blind IDs")
    concept_counts = {}
    for row in packet:
        if set(row) != {"blind_id", "dataset", "concept", "question", "items"}:
            raise ValueError("blind row contains unexpected metadata")
        if row["dataset"] not in ("twibot20", "botsim24") or not row["question"]:
            raise ValueError("blind row has invalid dataset or question")
        if row["concept"] not in CONCEPTS or \
                row["dataset"] not in CONCEPTS[row["concept"]]["applies_to"] or \
                row["question"] != CONCEPTS[row["concept"]]["question"]:
            raise ValueError("blind row concept does not match fixed annotation schema")
        refs = []
        for item in row["items"]:
            if set(item) != {"ref", "text", "truncated"} or not item["text"].strip() or \
                    not isinstance(item["truncated"], bool) or len(item["text"]) > 1800:
                raise ValueError("blind item contains extra metadata, empty text or invalid crop")
            refs.append(item["ref"])
        if len(refs) != len(set(refs)) or not refs:
            raise ValueError("blind item refs are duplicated or missing")
        mapping = source[row["blind_id"]]
        if set(mapping) != {"blind_id", "dataset", "account_id", "source_records"} or \
                mapping["dataset"] != row["dataset"] or \
                {item["ref"] for item in mapping["source_records"]} != set(refs):
            raise ValueError("private source mapping does not match blind refs")
        concept_counts[row["concept"]] = concept_counts.get(row["concept"], 0) + 1
    return {"tasks": len(packet), "concept_counts": concept_counts,
            "unique_accounts_in_private_mapping": len({row["account_id"] for row in provenance}),
            "blind_metadata_excludes_account_ids_labels_and_model_scores": True}


def _rater_rows(path: Path, packet: dict[str, dict], rater: str) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(packet):
        raise ValueError(f"rater {rater} has missing or duplicate task rows")
    by_id = {}
    for row in rows:
        uid = row["blind_id"]
        if uid not in packet or uid in by_id or row["concept"] != packet[uid]["concept"] or \
                row["annotator_id"] != rater:
            raise ValueError(f"rater {rater} task identity differs from packet")
        judgment = row["judgment_yes_no_uncertain"].strip().lower()
        if judgment and judgment not in ("yes", "no", "uncertain"):
            raise ValueError(f"rater {rater} used an invalid judgment")
        confidence = row["confidence_low_medium_high"].strip().lower()
        if confidence and confidence not in ("low", "medium", "high"):
            raise ValueError(f"rater {rater} used an invalid confidence")
        if row["minutes"].strip() and float(row["minutes"]) < 0:
            raise ValueError(f"rater {rater} used negative time")
        cited = [value.strip() for value in row["support_refs"].replace(";", ",").split(",")
                 if value.strip()]
        available = {item["ref"] for item in packet[uid]["items"]}
        if any(value not in available for value in cited):
            raise ValueError(f"rater {rater} cited a ref absent from packet")
        if judgment == "yes" and packet[uid]["concept"] == "templated_text_in_shown_sample" and \
                len(set(cited)) < 2:
            raise ValueError(f"rater {rater} templated-text yes needs two refs")
        if judgment in ("yes", "no") and packet[uid]["concept"] == "reply_addresses_parent" and \
                set(cited) != {"PARENT", "REPLY"}:
            raise ValueError(f"rater {rater} reply judgment must cite parent and reply")
        if judgment and (not confidence or not row["minutes"].strip()):
            raise ValueError(f"rater {rater} completed judgment lacks confidence or time")
        by_id[uid] = row
    return by_id


def validate_responses(packet: list[dict], rater_a: Path, rater_b: Path) -> dict:
    by_id = {row["blind_id"]: row for row in packet}
    a = _rater_rows(rater_a, by_id, "A")
    b = _rater_rows(rater_b, by_id, "B")
    missing = {rater: sum(not rows[uid]["judgment_yes_no_uncertain"].strip() for uid in by_id)
               for rater, rows in (("A", a), ("B", b))}
    if any(missing.values()):
        return {"status": "incomplete", "missing_judgments": missing,
                "agreement_not_computed": True}
    ordered = list(by_id)
    ya = [a[uid]["judgment_yes_no_uncertain"].strip().lower() for uid in ordered]
    yb = [b[uid]["judgment_yes_no_uncertain"].strip().lower() for uid in ordered]
    return {"status": "complete", "tasks": len(ordered),
            "exact_agreement": sum(x == y for x, y in zip(ya, yb)) / len(ordered),
            "cohen_kappa_three_categories": float(cohen_kappa_score(
                ya, yb, labels=["yes", "no", "uncertain"])),
            "disagreement_blind_ids": [uid for uid, x, y in zip(ordered, ya, yb) if x != y],
            "time_minutes": {"A": sum(float(a[uid]["minutes"]) for uid in ordered),
                             "B": sum(float(b[uid]["minutes"]) for uid in ordered)}}
