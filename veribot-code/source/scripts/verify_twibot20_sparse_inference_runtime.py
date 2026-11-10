"""Replay all official TwiBot-20 test decisions through the label-free runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import ijson
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.twibot20_sparse_runtime import TwiBot20SparseRuntime, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--base-source-audit", type=Path, required=True)
    parser.add_argument("--active-source-audit", type=Path, required=True)
    parser.add_argument("--active-cache", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--selected-report", type=Path, required=True)
    parser.add_argument("--saved-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = TwiBot20SparseRuntime(
        archive=args.zip, base_facts=args.base_facts,
        base_source_audit=args.base_source_audit,
        active_source_audit=args.active_source_audit,
        active_cache=args.active_cache, screen=args.screen,
        selection_lock=args.selection_lock, artifact=args.artifact,
        reference=args.reference, selected_report=args.selected_report)
    with np.load(args.base_facts, allow_pickle=False) as cached:
        ids = cached["ids"].astype(str)
        parts = cached["split"].astype(str)
        positions = cached["positions"].astype(int)
        contexts = cached["contexts"].astype(str)
        values = cached["values"].astype(float)
        names = cached["names"].astype(str).tolist()
    if names != list(runtime.reference.names):
        raise ValueError("official source fact vocabulary differs")
    test = np.flatnonzero(parts == "test")
    tweets = []
    with zipfile.ZipFile(args.zip) as zipped:
        with zipped.open("Twibot-20/test.json") as stream:
            for position, row in enumerate(ijson.items(stream, "item")):
                i = int(test[position])
                if (ids[i] != str(row["ID"]) or positions[i] != position
                        or contexts[i] != str(row.get("domain") or "unknown")):
                    raise ValueError("official test source or context differs")
                tweets.append([str(text or "") for text in row.get("tweet") or []])
    if len(test) != 1183 or len(tweets) != len(test):
        raise ValueError("official test population differs")
    predictions = runtime.predict(
        values[test], contexts[test], tweets, ids[test],
        source_rows=[("Twibot-20/test.json", int(positions[i])) for i in test])
    saved = json.loads(args.saved_predictions.read_text(encoding="utf-8"))
    if len(saved["predictions"]) != len(predictions):
        raise ValueError("saved test prediction count differs")
    with np.load(args.active_cache, allow_pickle=False) as checked:
        if not np.array_equal(checked["ids"].astype(str), ids):
            raise ValueError("independent active-unigram population differs")
        independent_lexical = checked["values"][test].astype(float)
    max_probability = max_logit = max_lexical = max_structured = 0.0
    decision_mismatch = 0
    located_unigram_items = 0
    for k, (actual, expected) in enumerate(zip(predictions, saved["predictions"])):
        if actual["account_id"] != expected["account_id"]:
            raise ValueError("saved test account order differs")
        max_probability = max(max_probability, abs(
            actual["probability"] - expected["probability"]))
        max_lexical = max(max_lexical, float(np.max(np.abs(
            np.asarray(actual["active_unigram_values"]) - independent_lexical[k]))))
        located_unigram_items += len(actual["active_unigram_source_locations"])
        actual_structured = sum(actual["structured_contributions"])
        actual_lexical = sum(actual["active_unigram_contributions"])
        max_structured = max(max_structured, abs(
            actual_structured - expected["group_contributions"]["structured_evidence"]),
            abs(actual_lexical - expected["group_contributions"]["unigram_evidence"]))
        max_logit = max(max_logit, abs(actual["logit"] - actual["intercept"]
                                       - actual_structured - actual_lexical))
        decision_mismatch += int(actual["prediction"] !=
                                 int(expected["probability"] >= actual["threshold"]))
    if (max_probability > 1e-10 or max_lexical > 1e-10
            or max_structured > 1e-10 or max_logit > 1e-10
            or decision_mismatch):
        raise ValueError("frozen test prediction does not follow audited evidence")
    result = {
        "dataset": "twibot20",
        "status": "all_official_test_predictions_replayed_by_label_free_evidence_runtime",
        "test_accounts": len(test),
        "classification_labels_read_from_source": False,
        "source_archive_sha256": sha256(args.zip),
        "source_fact_cache_sha256": sha256(args.base_facts),
        "source_audit_sha256": sha256(args.base_source_audit),
        "active_unigram_audit_sha256": sha256(args.active_source_audit),
        "model_artifact_sha256": sha256(args.artifact),
        "saved_predictions_sha256": sha256(args.saved_predictions),
        "max_probability_difference": max_probability,
        "max_logit_reconstruction_difference": max_logit,
        "max_independent_unigram_value_difference": max_lexical,
        "max_saved_group_contribution_difference": max_structured,
        "decision_mismatches": decision_mismatch,
        "nonzero_unigram_items_with_tweet_source_locations": located_unigram_items,
        "all_evidence_masked_logit": runtime.intercept,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
