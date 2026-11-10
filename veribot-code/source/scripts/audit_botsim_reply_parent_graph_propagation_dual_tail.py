"""Trace one controlled raw reply-parent edit through the full graph and model.

The edit is selected by source structure and split membership, before model
scores are inspected. It is an in-memory sensitivity probe, not a correction.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import sys

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from gate1_fox8_botsim_schema_dev import _missing_means, _source_facts
from screen_schema_v2_dual_tail_neutral_dev import _context, _design
from audit_botsim_twelve_independent_raw_truth import ACTIVITY, USERS, NATIVE_DIR
from verifiable_bot.botsim_native_facts import FACT_NAMES, extract
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v2.json"
LOCK = ROOT / "output/schema_v2_dual_tail_neutral_candidate/selection_lock.json"
TEST = ROOT / "output/schema_v2_dual_tail_neutral_candidate/fixed_test_report.json"
OUT = ROOT / "research/certificates/botsim_dual_tail_reply_parent_source_edit.json"


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    tested = json.loads(TEST.read_text(encoding="utf-8"))
    if (lock["schema_sha256"] != sha256_file(SCHEMA)
            or tested["selection_lock_sha256"] != sha256_file(LOCK)):
        raise ValueError("Frozen model records differ")
    with USERS.open("r", encoding="utf-8-sig", newline="") as stream:
        users = list(csv.DictReader(stream))
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    ids, original_native, original_trace, original_counts = extract(users, activity)
    with np.load(NATIVE_DIR / "native_graph_facts.npz",
                 allow_pickle=False) as cached:
        if (cached["ids"].astype(str).tolist() != ids
                or cached["names"].astype(str).tolist() != list(FACT_NAMES)
                or not np.array_equal(cached["values"], original_native)):
            raise ValueError("Fresh raw graph differs from released cache")
    source, _, model_ids, fact_names, split, manifest = _source_facts("botsim", 0)
    frozen = lock["runs"]["botsim"][0]
    if (ids != model_ids or manifest["split_sha256"] != frozen["split_sha256"]
            or manifest["files_sha256"] != frozen["source_files_sha256"]):
        raise ValueError("Model source or split differs")
    test_ids = {ids[int(index)] for index in split["test"]}
    pair_count = Counter((row["source_account_id"], row["target_account_id"])
                         for row in original_trace)
    eligible = sorted((row for row in original_trace
                       if (row["source_account_id"] in test_ids
                           and row["target_account_id"] in test_ids
                           and pair_count[(row["source_account_id"],
                                           row["target_account_id"])] == 1)),
                      key=lambda row: (row["source_account_id"],
                                       row["target_account_id"],
                                       row["comment_id"]))
    if not eligible:
        raise ValueError("No unique reply edge joins two test accounts")
    chosen = eligible[0]
    author = chosen["source_account_id"]
    target = chosen["target_account_id"]
    kind = chosen["activity_kind"]
    matches = [item for item in activity[author][kind]
               if str(item["comment_id"]) == chosen["comment_id"]]
    if len(matches) != 1 or str(matches[0]["parent_id"]) != chosen["parent_id"]:
        raise ValueError("Raw comment locator ambiguous")
    original_parent = matches[0]["parent_id"]
    missing_parent = "t3_veribot_audit_unresolved_parent"
    if any(missing_parent in str(row) for row in original_trace):
        raise ValueError("Synthetic missing parent collides with existing trace")
    matches[0]["parent_id"] = missing_parent
    try:
        edited_ids, edited_native, edited_trace, edited_counts = extract(users, activity)
    finally:
        matches[0]["parent_id"] = original_parent
    if edited_ids != ids or len(edited_trace) != len(original_trace)-1:
        raise ValueError("Edited graph did not remove exactly one reply")
    changed_rows = np.flatnonzero(np.any(edited_native != original_native, axis=1))
    changed_ids = {ids[int(i)] for i in changed_rows}
    if changed_ids != {author, target}:
        raise ValueError(f"Graph edit did not propagate to expected accounts: {changed_ids}")
    selected_names = [row["fact"] for row in schema["schema"]["botsim"]["concepts"]]
    source = _missing_means("botsim", source, fact_names, ids)
    facts = source[:, [fact_names.index(name) for name in selected_names]]
    positions = [ids.index(author), ids.index(target)]
    original = facts[positions].copy()
    edited = original.copy()
    for name in FACT_NAMES:
        j = selected_names.index(name)
        native_j = FACT_NAMES.index(name)
        if not np.array_equal(original[:, j], original_native[positions, native_j]):
            raise ValueError("Model native fact differs from raw graph")
        edited[:, j] = edited_native[positions, native_j]
    reference_path = ROOT / frozen["reference_path"]
    model_paths = [ROOT / path for path in frozen["model_paths"]]
    if (sha256_file(reference_path) != frozen["reference_sha256"]
            or [sha256_file(path) for path in model_paths] !=
                frozen["model_sha256"]):
        raise ValueError("Frozen model artifacts differ")
    reference = LabelBlindReference.load(reference_path)
    context = _context(facts, selected_names, frozen)
    context_rows = None if context is None else context[positions]
    old_x = _design(reference.transform(original, context_rows))
    new_x = _design(reference.transform(edited, context_rows))
    native_columns = [2*selected_names.index(name)+k
                      for name in FACT_NAMES for k in (0, 1)]
    unrelated = [j for j in range(old_x.shape[1]) if j not in native_columns]
    if not np.array_equal(old_x[:, unrelated], new_x[:, unrelated]):
        raise ValueError("Graph edit changed unrelated encoded evidence")
    original_scores = []
    edited_scores = []
    for fold, path in enumerate(model_paths):
        model = joblib.load(path)
        zero = frozen["zero_input_logit_by_fold"][fold]
        original_scores.append(np.asarray(model.booster_.predict(
            old_x, raw_score=True))-zero)
        edited_scores.append(np.asarray(model.booster_.predict(
            new_x, raw_score=True))-zero)
    original_score = np.mean(original_scores, axis=0)
    edited_score = np.mean(edited_scores, axis=0)
    matrix_path = ROOT / tested["results"]["botsim"][0]["group_matrix_path"]
    if sha256_file(matrix_path) != tested["results"]["botsim"][0][
            "group_matrix_sha256"]:
        raise ValueError("Saved model score matrix differs")
    with np.load(matrix_path, allow_pickle=False) as matrix:
        saved_index = {str(uid): j for j, uid in enumerate(matrix["ids"])}
        max_readback = max(abs(float(original_score[j]) -
                               float(matrix["score"][saved_index[uid]]))
                           for j, uid in enumerate((author, target)))
    if max_readback > 1e-12:
        raise ValueError("Raw graph baseline does not replay saved score")
    account_changes = []
    for j, uid in enumerate((author, target)):
        row = ids.index(uid)
        account_changes.append({
            "account_id": uid,
            "role": "comment_author" if uid == author else "parent_author",
            "native_fact_changes": {
                name: [float(original_native[row, k]),
                       float(edited_native[row, k])]
                for k, name in enumerate(FACT_NAMES)
                if original_native[row, k] != edited_native[row, k]
            },
            "original_score_relative_to_origin": float(original_score[j]),
            "edited_score_relative_to_origin": float(edited_score[j]),
            "score_delta_edited_minus_original": float(
                edited_score[j]-original_score[j]),
            "score_delta_by_frozen_fold": [float(
                edited_scores[fold][j]-original_scores[fold][j])
                for fold in range(len(model_paths))],
            "original_class": int(original_score[j] > 0),
            "edited_class": int(edited_score[j] > 0),
        })
    report = {
        "scope": "single_in_memory_raw_parent_edit_full_graph_rebuild_to_frozen_model",
        "selection_rule": "lexicographically_first_unique_directed_reply_pair_with_both_accounts_in_seed0_test; model scores not used to select",
        "eligible_reply_records": len(eligible),
        "raw_comment_locator": {
            "source_account_id": author,
            "activity_kind": kind,
            "comment_id": chosen["comment_id"],
            "original_parent_id": original_parent,
            "synthetic_unresolved_parent_id": missing_parent,
            "resolved_original_target_account_id": target,
        },
        "original_graph_counts": original_counts,
        "edited_graph_counts": edited_counts,
        "accounts_whose_native_facts_changed": sorted(changed_ids),
        "original_score_max_saved_readback_error": float(max_readback),
        "account_changes": account_changes,
        "raw_files_mutated": False,
        "natural_record_error_or_correct_value_established": False,
        "real_world_causality_established": False,
        "selection_lock_sha256": sha256_file(LOCK),
        "fixed_test_report_sha256": sha256_file(TEST),
        "raw_users_sha256": sha256_file(USERS),
        "raw_activity_sha256": sha256_file(ACTIVITY),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(OUT)
    print(account_changes)


if __name__ == "__main__":
    main()
