"""Locate per-fact reference fallback and its frozen decision response."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from lightgbm import Booster
import numpy as np
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[2]
FACTS = ROOT / "data/processed/twibot20_v3/facts_split_visible_train_dev.npz"
REFERENCE = ROOT / ("output/twibot20_compact20_ab_group_tree_dev/"
                    "reference_split_visible_train_dev.json.gz")
CARDS = ROOT / "research/certificates/twibot20_split_visible_graph_ab_dev_decisions.json"
OUT = ROOT / "research/certificates/twibot20_reference_fallback_decisions.json"
ASSIGNMENTS = ROOT / "research/certificates/twibot20_reference_context_assignments.jsonl"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = json.loads(CARDS.read_text(encoding="utf-8"))
    if (report["fact_cache_sha256"] != sha(FACTS) or
            report["reference_artifact_sha256"] != sha(REFERENCE)):
        raise ValueError("cards do not bind these facts and reference")
    cards = report["decision_cards"]
    with np.load(FACTS, allow_pickle=False) as saved:
        dev = np.flatnonzero(saved["split"].astype(str) == "dev")
        ids = saved["ids"].astype(str)[dev]
        contexts = saved["contexts"].astype(str)[dev]
        all_names = saved["names"].astype(str).tolist()
        train = saved["split"].astype(str) == "train"
        train_contexts = saved["contexts"].astype(str)[train]
        train_values = saved["values"].astype(float)[train]
    with gzip.open(REFERENCE, "rt", encoding="utf-8") as stream:
        stored = json.load(stream)
    payload = stored["payload"]
    reference_version = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if reference_version != stored["version"]:
        raise ValueError("reference version checksum mismatch")
    model_path = Path(report["frozen_model_path"])
    if sha(model_path) != report["frozen_model_sha256"]:
        raise ValueError("frozen model changed")
    booster = Booster(model_file=str(model_path))
    if len(cards) != len(dev) or len(cards) != report["dev_n"]:
        raise ValueError("dev card count mismatch")
    names = [row["fact_name"] for row in cards[0]["fact_terms"]]
    if len(names) != 20 or len(set(names)) != 20:
        raise ValueError("card vocabulary differs")
    key_set = set(payload["reference"])
    fallback_by_fact: Counter[str] = Counter()
    fallback_by_context: Counter[str] = Counter()
    missing_key_by_fact: Counter[str] = Counter()
    missing_key_by_context: Counter[str] = Counter()
    accounts_with_missing_key = 0
    unavailable_missing_key_cells = 0
    affected = []
    assignments = []
    max_logit_replay_error = 0.0
    for k, (uid, context, card) in enumerate(zip(ids, contexts, cards, strict=True)):
        if (card["account_id"] != uid or
                card["reference_version"] != report["reference_version_with_support_rule"]):
            raise ValueError("account card or reference version differs")
        terms = card["fact_terms"]
        if [term["fact_name"] for term in terms] != names:
            raise ValueError("fact order differs across cards")
        x = np.array([[
            *(term["upper_model_input"] for term in terms),
            *(term["lower_model_input"] for term in terms),
        ]], dtype=float)
        full_logit = float(booster.predict(x, raw_score=True)[0])
        max_logit_replay_error = max(max_logit_replay_error,
                                     abs(full_logit - card["full_logit"]))
        fallback = []
        unavailable_without_context_reference = []
        for j, name in enumerate(names):
            index = all_names.index(name)
            if f"{context}:{index}" in key_set:
                continue
            if f"all:{index}" not in key_set:
                raise ValueError("neither stratum nor pooled reference exists")
            missing_key_by_fact[name] += 1
            missing_key_by_context[context] += 1
            if bool(terms[j]["missing"]):
                if (terms[j]["upper_tail_score"] != 0 or
                        terms[j]["lower_tail_score"] != 0 or
                        terms[j]["reliability"] != 0):
                    raise ValueError("unavailable fact unexpectedly has reference evidence")
                unavailable_without_context_reference.append(name)
                unavailable_missing_key_cells += 1
                continue
            fallback.append(name)
            fallback_by_fact[name] += 1
            fallback_by_context[context] += 1
        if fallback or unavailable_without_context_reference:
            accounts_with_missing_key += 1
        assignments.append({
            "account_id": uid,
            "declared_context": context,
            "fallback_to_all_facts": fallback,
            "unavailable_facts_with_no_context_reference_lookup":
                unavailable_without_context_reference,
            "effective_context_for_other_available_facts": context,
            "reference_version": reference_version,
        })
        if not fallback:
            continue
        masked = x.copy()
        for name in fallback:
            j = names.index(name)
            masked[0, [j, j + len(names)]] = 0.0
        masked_logit = float(booster.predict(masked, raw_score=True)[0])
        threshold = float(card["threshold"])
        affected.append({
            "account_id": uid,
            "declared_context": context,
            "fallback_facts": fallback,
            "effective_reference_context_for_fallback_facts": "all",
            "full_logit": full_logit,
            "threshold_logit": float(card["threshold_logit"]),
            "original_prediction": int(card["prediction"]),
            "evaluation_label": int(card["evaluation_label"]),
            "fallback_facts_masked_logit": masked_logit,
            "fallback_facts_masked_prediction": int(expit(masked_logit) >= threshold),
            "conditional_mask_logit_effect": full_logit - masked_logit,
            "class_changed_when_fallback_facts_masked":
                bool((expit(full_logit) >= threshold) !=
                     (expit(masked_logit) >= threshold)),
        })
    if max_logit_replay_error > 1e-9:
        raise ValueError("stored decision cards do not replay")
    support = {}
    for context in missing_key_by_context:
        selected = train_contexts == context
        support[context] = {
            "train_accounts_in_declared_context": int(selected.sum()),
            "facts": {},
        }
        for name in missing_key_by_fact:
            j = all_names.index(name)
            observed = int(np.isfinite(train_values[selected, j]).sum())
            if observed >= int(payload["min_context_n"]):
                raise ValueError("fallback occurred despite sufficient context support")
            support[context]["facts"][name] = {
                "observed_training_values_in_context": observed,
                "minimum_context_n": int(payload["min_context_n"]),
                "pooled_observed_training_values": len(payload["reference"][f"all:{j}"]),
            }
    ASSIGNMENTS.write_text("".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in assignments), encoding="utf-8")
    result = {
        "scope": "TwiBot-20 split-visible AB group-tree dev reference-fallback audit",
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
            for path in (FACTS, REFERENCE, CARDS, model_path)
        },
        "reference_version": reference_version,
        "decision_evidence_version_with_support_rule": report[
            "reference_version_with_support_rule"],
        "dev_accounts": len(cards),
        "candidate_fact_cells": len(cards) * len(names),
        "accounts_with_missing_context_reference_key": accounts_with_missing_key,
        "missing_context_reference_key_cells": sum(missing_key_by_fact.values()),
        "unavailable_cells_no_reference_lookup": unavailable_missing_key_cells,
        "accounts_with_any_fallback": len(affected),
        "fallback_fact_cells": sum(fallback_by_fact.values()),
        "fallback_cells_by_fact": dict(fallback_by_fact),
        "fallback_cells_by_declared_context": dict(fallback_by_context),
        "reference_support_by_fallback_context": support,
        "all_card_reference_assignments": str(ASSIGNMENTS.relative_to(ROOT)).replace("\\", "/"),
        "all_card_reference_assignments_sha256": sha(ASSIGNMENTS),
        "max_full_logit_replay_error": max_logit_replay_error,
        "affected_accounts_with_nonzero_fallback_fact_mask_effect": sum(
            abs(row["conditional_mask_logit_effect"]) > 1e-9 for row in affected),
        "affected_accounts_changing_class_on_fallback_fact_mask": sum(
            row["class_changed_when_fallback_facts_masked"] for row in affected),
        "operation_note": "Encoded fact masking diagnoses the frozen function; it is not a source edit.",
        "affected": affected,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
