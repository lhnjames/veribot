"""Compare a coherent raw description edit with encoded group masking.

Removing a description here is a controlled sensitivity probe, not a claim
that the source record is erroneous or that editing it benefits auditing.
"""

from __future__ import annotations

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
from audit_botsim_twelve_independent_raw_truth import (
    ACTIVITY, BASE_NAMES, USERS, raw_account,
)
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v2.json"
LOCK = ROOT / "output/schema_v2_dual_tail_neutral_candidate/selection_lock.json"
TEST = ROOT / "output/schema_v2_dual_tail_neutral_candidate/fixed_test_report.json"
OUT = ROOT / "research/certificates/botsim_dual_tail_description_source_edit.json"


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    tested = json.loads(TEST.read_text(encoding="utf-8"))
    if (lock["schema_sha256"] != sha256_file(SCHEMA)
            or tested["selection_lock_sha256"] != sha256_file(LOCK)):
        raise ValueError("Frozen candidate source differs")
    spec = schema["schema"]["botsim"]
    names = [item["fact"] for item in spec["concepts"]]
    field_indices = [names.index("has_description"),
                     names.index("description_len")]
    with USERS.open("r", encoding="utf-8-sig", newline="") as stream:
        source_rows = {row["user_id"]: row for row in csv.DictReader(stream)}
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    runs = []
    for seed, frozen in enumerate(lock["runs"]["botsim"]):
        source, _, ids, fact_names, split, manifest = _source_facts("botsim", seed)
        if (manifest["split_sha256"] != frozen["split_sha256"]
                or manifest["files_sha256"] != frozen["source_files_sha256"]
                or set(source_rows) != set(ids) or set(activity) != set(ids)):
            raise ValueError("Raw source, account population or split differs")
        source = _missing_means("botsim", source, fact_names, ids)
        facts = source[:, [fact_names.index(name) for name in names]]
        test = split["test"]
        original = facts[test].copy()
        edited = original.copy()
        eligible = []
        for position, index in enumerate(test):
            uid = ids[int(index)]
            raw_row = source_rows[uid]
            original_fields = dict(zip(BASE_NAMES,
                                       raw_account(raw_row, activity[uid]), strict=True))
            for name, j in zip(("has_description", "description_len"),
                               field_indices, strict=True):
                if abs(float(original_fields[name])-original[position, j]) > 1e-6:
                    raise ValueError("Independent raw description differs from model")
            changed_row = dict(raw_row)
            changed_row["description"] = ""
            changed_fields = dict(zip(BASE_NAMES,
                                      raw_account(changed_row, activity[uid]), strict=True))
            for name, j in zip(("has_description", "description_len"),
                               field_indices, strict=True):
                edited[position, j] = float(changed_fields[name])
            eligible.append(bool(original_fields["has_description"]))
        eligible = np.asarray(eligible, dtype=bool)
        if not np.array_equal(edited[~eligible], original[~eligible],
                              equal_nan=True):
            raise ValueError("Empty descriptions unexpectedly changed")
        reference_path = ROOT / frozen["reference_path"]
        model_paths = [ROOT / item for item in frozen["model_paths"]]
        if (sha256_file(reference_path) != frozen["reference_sha256"]
                or [sha256_file(path) for path in model_paths] !=
                    frozen["model_sha256"]):
            raise ValueError("Frozen reference or models differ")
        reference = LabelBlindReference.load(reference_path)
        context = _context(facts, names, frozen)
        original_x = _design(reference.transform(
            original, None if context is None else context[test]))
        edited_x = _design(reference.transform(
            edited, None if context is None else context[test]))
        affected = [2*j+k for j in field_indices for k in (0, 1)]
        other = [j for j in range(original_x.shape[1]) if j not in affected]
        if not np.array_equal(original_x[:, other], edited_x[:, other]):
            raise ValueError("Description edit changed unrelated model inputs")
        original_scores = []
        edited_scores = []
        for fold, path in enumerate(model_paths):
            model = joblib.load(path)
            zero = frozen["zero_input_logit_by_fold"][fold]
            original_scores.append(np.asarray(model.booster_.predict(
                original_x, raw_score=True))-zero)
            edited_scores.append(np.asarray(model.booster_.predict(
                edited_x, raw_score=True))-zero)
        original_score = np.mean(original_scores, axis=0)
        edited_score = np.mean(edited_scores, axis=0)
        fixed_run = tested["results"]["botsim"][seed]
        matrix_path = ROOT / fixed_run["group_matrix_path"]
        if sha256_file(matrix_path) != fixed_run["group_matrix_sha256"]:
            raise ValueError("Saved group-mask result differs")
        with np.load(matrix_path, allow_pickle=False) as matrix:
            group_names = matrix["group_names"].astype(str).tolist()
            if (group_names.count("presentation") != 1
                    or matrix["ids"].astype(str).tolist() !=
                        [ids[int(i)] for i in test]):
                raise ValueError("Presentation group or account order differs")
            saved_error = float(np.max(np.abs(
                original_score-matrix["score"])))
            masked_score = matrix["without_group_scores"][:,
                         group_names.index("presentation")].copy()
        if saved_error > 1e-12:
            raise ValueError("Original source does not replay saved model")
        source_edit_change = (edited_score > 0) != (original_score > 0)
        group_mask_change = (masked_score > 0) != (original_score > 0)
        comparison = (edited_score > 0) != (masked_score > 0)
        path = ROOT / fixed_run["prediction_path"]
        if sha256_file(path) != fixed_run["prediction_sha256"]:
            raise ValueError("Fixed test prediction file differs")
        npz_path = OUT.with_name(f"botsim_dual_tail_description_source_edit_seed{seed}.npz")
        np.savez_compressed(
            npz_path, ids=np.asarray([ids[int(i)] for i in test]),
            description_present=eligible,
            original_description_length=original[:, field_indices[1]],
            original_score=original_score, edited_score=edited_score,
            encoded_group_mask_score=masked_score,
            source_edit_class_change=source_edit_change,
            group_mask_class_change=group_mask_change)
        runs.append({
            "seed": seed, "test_accounts": len(test),
            "source_rows_with_nonempty_description": int(np.count_nonzero(eligible)),
            "max_original_score_readback_error": saved_error,
            "source_edit_changes_class": int(np.count_nonzero(
                source_edit_change & eligible)),
            "encoded_group_mask_changes_class": int(np.count_nonzero(
                group_mask_change & eligible)),
            "source_edit_vs_group_mask_class_disagreement": int(np.count_nonzero(
                comparison & eligible)),
            "source_edit_vs_group_mask_score_difference": {
                "max_absolute": float(np.max(np.abs(
                    edited_score[eligible]-masked_score[eligible]))),
                "mean_absolute": float(np.mean(np.abs(
                    edited_score[eligible]-masked_score[eligible]))),
            },
            "account_matrix": str(npz_path.relative_to(ROOT)),
            "account_matrix_sha256": sha256_file(npz_path),
        })
        print(runs[-1], flush=True)
    report = {
        "scope": "controlled_description_removal_vs_encoded_group_mask_on_frozen_dual_tail",
        "source_edit": "Set raw Users.csv description to empty in memory; recompute both has_description and description_len; keep frozen reference and model; do not mutate source files.",
        "natural_record_error_or_correct_value_established": False,
        "real_world_behavioral_causality_established": False,
        "selection_lock_sha256": sha256_file(LOCK),
        "fixed_test_report_sha256": sha256_file(TEST),
        "raw_users_sha256": sha256_file(USERS),
        "raw_activity_sha256": sha256_file(ACTIVITY),
        "runs": runs,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
