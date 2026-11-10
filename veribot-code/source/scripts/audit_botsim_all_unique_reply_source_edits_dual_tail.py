"""Audit all uniquely supported BotSim test-to-test reply edits.

Each in-memory parent-ID edit removes one resolvable cross-account reply.
Local graph count updates are checked against three full graph rebuilds.
This is a model sensitivity audit, not a natural correction experiment.
"""

from __future__ import annotations

from collections import Counter, defaultdict
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
from verifiable_bot.botsim_native_facts import FACT_NAMES, _object_id, extract
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v2.json"
LOCK = ROOT / "output/schema_v2_dual_tail_neutral_candidate/selection_lock.json"
TEST = ROOT / "output/schema_v2_dual_tail_neutral_candidate/fixed_test_report.json"
OUT = ROOT / "research/certificates/botsim_dual_tail_all_unique_reply_source_edits.json"


def local_edit_values(source: str, target: str, outgoing: dict, incoming: dict,
                      out_count: Counter, in_count: Counter) -> tuple[np.ndarray, np.ndarray]:
    reciprocal = int(source in outgoing[target])
    source_counts = [
        len(outgoing[source])-1,
        len(incoming[source]),
        len(outgoing[source] & incoming[source])-reciprocal,
        out_count[source]-1,
        in_count[source],
    ]
    target_counts = [
        len(outgoing[target]),
        len(incoming[target])-1,
        len(outgoing[target] & incoming[target])-reciprocal,
        out_count[target],
        in_count[target]-1,
    ]
    if min(source_counts + target_counts) < 0:
        raise ValueError("Local graph edit produced negative count")
    return (np.log1p(np.asarray(source_counts, dtype=np.float32)),
            np.log1p(np.asarray(target_counts, dtype=np.float32)))


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    tested = json.loads(TEST.read_text(encoding="utf-8"))
    if (lock["schema_sha256"] != sha256_file(SCHEMA)
            or tested["selection_lock_sha256"] != sha256_file(LOCK)):
        raise ValueError("Frozen model lock differs")
    with USERS.open("r", encoding="utf-8-sig", newline="") as stream:
        users = list(csv.DictReader(stream))
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    ids, native, trace, graph_counts = extract(users, activity)
    with np.load(NATIVE_DIR / "native_graph_facts.npz",
                 allow_pickle=False) as cache:
        if (cache["ids"].astype(str).tolist() != ids
                or cache["names"].astype(str).tolist() != list(FACT_NAMES)
                or not np.array_equal(cache["values"], native)):
            raise ValueError("Raw graph differs from saved factual inputs")
    source, _, model_ids, fact_names, split, manifest = _source_facts("botsim", 0)
    frozen = lock["runs"]["botsim"][0]
    if (model_ids != ids or manifest["split_sha256"] != frozen["split_sha256"]
            or manifest["files_sha256"] != frozen["source_files_sha256"]):
        raise ValueError("Model and raw graph populations differ")
    test_ids = {ids[int(i)] for i in split["test"]}
    pair_count = Counter((row["source_account_id"], row["target_account_id"])
                         for row in trace)
    eligible = sorted((row for row in trace
                       if (row["source_account_id"] in test_ids
                           and row["target_account_id"] in test_ids
                           and pair_count[(row["source_account_id"],
                                           row["target_account_id"])] == 1)),
                      key=lambda row: (row["source_account_id"],
                                       row["target_account_id"],
                                       row["comment_id"]))
    if len(eligible) != 627:
        raise ValueError("Eligible source-only edit population differs")
    missing_parent = "t3_veribot_audit_unresolved_parent"
    existing_object_ids = set()
    raw_locators = Counter()
    for uid, record in activity.items():
        for kind, key in (("posts", "submission_id"),
                          ("comment_1", "comment_id"),
                          ("comment_2", "comment_id")):
            for item in record.get(kind, []):
                existing_object_ids.add(_object_id(item[key]))
                if kind != "posts":
                    raw_locators[(uid, kind, str(item["comment_id"]),
                                  str(item.get("parent_id")))] += 1
    if _object_id(missing_parent) in existing_object_ids or any(
            raw_locators[(row["source_account_id"], row["activity_kind"],
                          row["comment_id"], row["parent_id"])] != 1
            for row in eligible):
        raise ValueError("Synthetic parent or raw comment locator is ambiguous")
    outgoing = defaultdict(set)
    incoming = defaultdict(set)
    out_count = Counter()
    in_count = Counter()
    for row in trace:
        source_id = row["source_account_id"]
        target_id = row["target_account_id"]
        outgoing[source_id].add(target_id)
        incoming[target_id].add(source_id)
        out_count[source_id] += 1
        in_count[target_id] += 1
    index = {uid: j for j, uid in enumerate(ids)}
    expected_rows = []
    for row in eligible:
        source_id = row["source_account_id"]
        target_id = row["target_account_id"]
        source_native, target_native = local_edit_values(
            source_id, target_id, outgoing, incoming, out_count, in_count)
        expected_rows.append((source_native, target_native))

    checked_rebuilds = []
    for event_index in sorted({0, len(eligible)//2, len(eligible)-1}):
        row = eligible[event_index]
        author = row["source_account_id"]
        matches = [item for item in activity[author][row["activity_kind"]]
                   if str(item["comment_id"]) == row["comment_id"]]
        if len(matches) != 1 or str(matches[0]["parent_id"]) != row["parent_id"]:
            raise ValueError("Raw comment edit locator differs")
        old_parent = matches[0]["parent_id"]
        matches[0]["parent_id"] = missing_parent
        try:
            changed_ids, changed, changed_trace, _ = extract(users, activity)
        finally:
            matches[0]["parent_id"] = old_parent
        expected = native.copy()
        expected[index[author]] = expected_rows[event_index][0]
        expected[index[row["target_account_id"]]] = expected_rows[event_index][1]
        if (changed_ids != ids or len(changed_trace) != len(trace)-1
                or not np.array_equal(changed, expected)):
            raise ValueError(f"Local graph formula differs from full rebuild {event_index}")
        checked_rebuilds.append(event_index)

    source = _missing_means("botsim", source, fact_names, ids)
    selected_names = [item["fact"] for item in schema["schema"]["botsim"]["concepts"]]
    facts = source[:, [fact_names.index(name) for name in selected_names]]
    reference_path = ROOT / frozen["reference_path"]
    model_paths = [ROOT / item for item in frozen["model_paths"]]
    if (sha256_file(reference_path) != frozen["reference_sha256"]
            or [sha256_file(path) for path in model_paths] !=
                frozen["model_sha256"]):
        raise ValueError("Frozen model artifacts differ")
    positions = []
    edited_rows = []
    for event_index, row in enumerate(eligible):
        for role, uid in enumerate((row["source_account_id"],
                                    row["target_account_id"])):
            account = index[uid]
            changed_row = facts[account].copy()
            for k, name in enumerate(FACT_NAMES):
                j = selected_names.index(name)
                if changed_row[j] != native[account, k]:
                    raise ValueError("Model native input differs from raw graph")
                changed_row[j] = expected_rows[event_index][role][k]
            positions.append(account)
            edited_rows.append(changed_row)
    original_rows = facts[positions]
    edited_rows = np.asarray(edited_rows)
    context = _context(facts, selected_names, frozen)
    selected_context = None if context is None else context[positions]
    reference = LabelBlindReference.load(reference_path)
    x_old = _design(reference.transform(original_rows, selected_context))
    x_new = _design(reference.transform(edited_rows, selected_context))
    native_columns = [2*selected_names.index(name)+k
                      for name in FACT_NAMES for k in (0, 1)]
    unrelated = [j for j in range(x_old.shape[1]) if j not in native_columns]
    if not np.array_equal(x_old[:, unrelated], x_new[:, unrelated]):
        raise ValueError("Raw graph edit altered non-graph encoded evidence")
    fold_original = []
    fold_edited = []
    for fold, path in enumerate(model_paths):
        model = joblib.load(path)
        zero = frozen["zero_input_logit_by_fold"][fold]
        fold_original.append(np.asarray(model.booster_.predict(
            x_old, raw_score=True))-zero)
        fold_edited.append(np.asarray(model.booster_.predict(
            x_new, raw_score=True))-zero)
    original_score = np.mean(fold_original, axis=0)
    edited_score = np.mean(fold_edited, axis=0)
    matrix_path = ROOT / tested["results"]["botsim"][0]["group_matrix_path"]
    if sha256_file(matrix_path) != tested["results"]["botsim"][0][
            "group_matrix_sha256"]:
        raise ValueError("Saved model score matrix differs")
    with np.load(matrix_path, allow_pickle=False) as matrix:
        lookup = {str(uid): j for j, uid in enumerate(matrix["ids"])}
        saved = np.asarray([matrix["score"][lookup[ids[pos]]]
                            for pos in positions])
    max_readback = float(np.max(np.abs(original_score-saved)))
    if max_readback > 1e-12:
        raise ValueError("Original source scores do not replay saved decisions")
    delta = edited_score-original_score
    class_change = (edited_score > 0) != (original_score > 0)
    tol = 1e-10
    author_response = np.abs(delta[0::2]) > tol
    parent_response = np.abs(delta[1::2]) > tol
    author_class = class_change[0::2]
    parent_class = class_change[1::2]
    if len(author_response) != len(eligible):
        raise ValueError("Event-response pairing differs")
    matrix_out = OUT.with_name("botsim_dual_tail_all_unique_reply_source_edits.npz")
    np.savez_compressed(
        matrix_out,
        comment_ids=np.asarray([row["comment_id"] for row in eligible]),
        author_ids=np.asarray([row["source_account_id"] for row in eligible]),
        parent_author_ids=np.asarray([row["target_account_id"] for row in eligible]),
        original_scores=original_score.reshape(-1, 2),
        edited_scores=edited_score.reshape(-1, 2),
        fold_score_deltas=np.column_stack([
            new-old for old, new in zip(fold_original, fold_edited, strict=True)
        ]).reshape(-1, 2, len(model_paths)),
        original_native=native[positions].reshape(-1, 2, len(FACT_NAMES)),
        edited_native=np.asarray([item for pair in expected_rows for item in pair])
                     .reshape(-1, 2, len(FACT_NAMES)))
    report = {
        "scope": "all_unique_pair_seed0_test_reply_parent_unresolution_edits",
        "selection": "unique directed pair with both endpoints in seed0 fixed test; raw-source structure only, no labels or model scores",
        "events": len(eligible),
        "endpoint_account_evaluations": len(positions),
        "full_graph_rebuilds_checked_at_event_indices": checked_rebuilds,
        "full_graph_rebuild_checks_passed": True,
        "all_raw_comment_locators_unique_verified": True,
        "synthetic_parent_id_absent_from_raw_objects_verified": True,
        "source_facts_changed_at_both_endpoints_for_every_event": bool(all(
            np.any(expected_rows[j][0] != native[index[row["source_account_id"]]])
            and np.any(expected_rows[j][1] != native[index[row["target_account_id"]]])
            for j, row in enumerate(eligible))),
        "max_original_score_readback_error": max_readback,
        "author_score_response_events": int(np.count_nonzero(author_response)),
        "parent_author_score_response_events": int(np.count_nonzero(parent_response)),
        "distinct_comment_authors_eligible": len(set(
            row["source_account_id"] for row in eligible)),
        "distinct_parent_authors_eligible": len(set(
            row["target_account_id"] for row in eligible)),
        "distinct_comment_authors_with_score_response": len(set(
            row["source_account_id"] for row, responsive in
            zip(eligible, author_response, strict=True) if responsive)),
        "distinct_parent_authors_with_score_response": len(set(
            row["target_account_id"] for row, responsive in
            zip(eligible, parent_response, strict=True) if responsive)),
        "both_endpoint_score_response_events": int(np.count_nonzero(
            author_response & parent_response)),
        "neither_endpoint_score_response_events": int(np.count_nonzero(
            ~author_response & ~parent_response)),
        "author_class_change_events": int(np.count_nonzero(author_class)),
        "parent_author_class_change_events": int(np.count_nonzero(parent_class)),
        "max_absolute_score_response_by_role": {
            "comment_author": float(np.max(np.abs(delta[0::2]))),
            "parent_author": float(np.max(np.abs(delta[1::2]))),
        },
        "account_matrix": str(matrix_out.relative_to(ROOT)),
        "account_matrix_sha256": sha256_file(matrix_out),
        "raw_files_mutated": False,
        "natural_record_error_or_correct_value_established": False,
        "real_world_causality_established": False,
        "original_graph_counts": graph_counts,
        "selection_lock_sha256": sha256_file(LOCK),
        "fixed_test_report_sha256": sha256_file(TEST),
        "raw_users_sha256": sha256_file(USERS),
        "raw_activity_sha256": sha256_file(ACTIVITY),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(OUT)
    print({k: report[k] for k in (
        "events", "author_score_response_events",
        "parent_author_score_response_events",
        "neither_endpoint_score_response_events",
        "author_class_change_events", "parent_author_class_change_events")})


if __name__ == "__main__":
    main()
