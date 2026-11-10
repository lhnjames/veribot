"""Independently replay source-derived Human normality and fixed predictions."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output/v3_mgtab_targeted_20_concept_ab_group_tree"
FOLDER = ROOT / "output/v3_mgtab_positive_human_evidence"
OUT = ROOT / "research/certificates/mgtab_positive_human_evidence_source_replay.json"
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    report_path = FOLDER / "fixed_test_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    lock_path = FOLDER / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads((BASE / "evidence_cards_manifest.json")
                          .read_text(encoding="utf-8"))
    if (report["selection_lock_sha256"] != sha256(lock_path) or
            lock["dev_screen_sha256"] != sha256(FOLDER / "dev_only.json") or
            lock["test_metric_or_prediction_used_to_create_lock"] is not False):
        raise ValueError("dev selection lock differs")
    runs = []
    for seed in (0, 1, 2):
        run_dir = FOLDER / f"seed{seed}"
        model_path = run_dir / "model.joblib"
        evidence_path = run_dir / "test_evidence.npz"
        previous_path = BASE / f"seed{seed}/test_evidence.npz"
        card_path = BASE / f"seed{seed}/test_evidence_cards.jsonl.gz"
        saved = report["runs"][seed]
        if (saved["model_sha256"] != sha256(model_path) or
                saved["evidence_sha256"] != sha256(evidence_path) or
                manifest["runs"][seed]["sha256"] != sha256(card_path)):
            raise ValueError("frozen model, matrix, or source card changed")
        with np.load(evidence_path, allow_pickle=False) as arr:
            ids = arr["ids"].astype(str).tolist()
            x = arr["model_input"].copy()
            labels = arr["label"].astype(int)
            probability = arr["probability"].copy()
            threshold = float(arr["threshold"])
            groups = arr["source_group_names"].astype(str).tolist()
            signs = arr["direction_signs"].astype(int).tolist()
        with np.load(previous_path, allow_pickle=False) as arr:
            old_ids = arr["ids"].astype(str).tolist()
            old_x = arr["evidence_matrix"].copy()
            old_labels = arr["label"].astype(int)
        with gzip.open(card_path, "rt", encoding="utf-8") as stream:
            cards = [json.loads(line) for line in stream]
        if (ids != old_ids or len(ids) != len(cards) or len(ids) != len(labels) or
                groups != list(cards[0]["group_effects"]) or
                not np.array_equal(labels, old_labels) or
                x.shape != (len(ids), 50) or
                np.max(np.abs(x[:, :40]-old_x)) > 1e-12 or
                signs[40:] != [-1]*10):
            raise ValueError("source concept population or sign contract differs")
        located = 0
        for j, card in enumerate(cards):
            if card["account_id"] != ids[j] or card["label"] != labels[j]:
                raise ValueError("source card account or label differs")
            for k, group in enumerate(groups):
                evidence = card["evidence"][2*k:2*k+2]
                if any(not row["concept"].startswith(group+":") for row in evidence):
                    raise ValueError("source concept group differs")
                present = any(row["winning_fact"] is not None for row in evidence)
                if any((row["winning_fact"] is None) !=
                       (row["source_locator"] is None) for row in evidence):
                    raise ValueError("human evidence lacks source locator")
                located += sum(row["source_locator"] is not None for row in evidence)
                expected = float(present) * np.exp(-max(old_x[j, 2*k],
                                                          old_x[j, 2*k+1]))
                if abs(expected-x[j, 40+k]) > 1e-12:
                    raise ValueError("source-derived human normality differs")
        model = joblib.load(model_path)
        if model.get_params()["monotone_constraints"] != signs:
            raise ValueError("saved monotone direction contract differs")
        source_cols = [[2*k, 2*k+1, 20+2*k, 21+2*k, 40+k]
                       for k in range(10)]
        one_group, counts = all_trees_one_group(model, source_cols)
        if not one_group:
            raise ValueError("tree crossed source groups")
        replay = model.predict_proba(x)[:, 1]
        max_error = float(np.max(np.abs(replay-probability)))
        if max_error > 1e-12:
            raise ValueError("model probabilities differ")
        zero = float(model.predict_proba(np.zeros((1, 50)))[:, 1][0])
        if abs(zero-saved["zero_evidence_probability"]) > 1e-12:
            raise ValueError("zero-evidence prediction differs")
        full_pred = probability >= threshold
        no_normality = x.copy()
        no_normality[:, 40:] = 0
        no_normality_pred = model.predict_proba(no_normality)[:, 1] >= threshold
        lost = int(np.count_nonzero((full_pred == labels) &
                                    (no_normality_pred != labels)))
        fixed = int(np.count_nonzero((full_pred != labels) &
                                     (no_normality_pred == labels)))
        changes = int(np.count_nonzero(full_pred != no_normality_pred))
        if changes != lost + fixed:
            raise ValueError("normality decision counts differ")
        mask_counts = {}
        for k, group in enumerate(groups):
            masked = x.copy()
            masked[:, source_cols[k]] = 0
            mask_counts[group] = int(np.count_nonzero(
                (model.predict_proba(masked)[:, 1] >= threshold) != full_pred))
        if mask_counts != saved["group_mask_changed_classes"]:
            raise ValueError("source group masks differ")
        runs.append({
            "seed": seed, "accounts": len(ids), "source_locator_concepts": located,
            "model_sha256": sha256(model_path),
            "new_evidence_sha256": sha256(evidence_path),
            "original_source_cards_sha256": sha256(card_path),
            "max_replay_probability_error": max_error,
            "all_zero_probability": zero,
            "all_zero_class": int(zero >= threshold),
            "human_normality_only_mask_changes": changes,
            "human_to_bot_when_human_normality_removed": int(np.count_nonzero(
                (~full_pred) & no_normality_pred)),
            "correct_lost_when_human_normality_removed": lost,
            "errors_fixed_when_human_normality_removed": fixed,
            "net_correct_supported_by_human_normality": lost-fixed,
            "source_group_masks": mask_counts,
            "source_group_tree_counts": counts,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "independent_mgtab_source_present_human_evidence_replay",
        "source_tier": "published profile and averaged tweet tensors plus typed relation tensors",
        "prior_source_audit_manifest_sha256": sha256(BASE / "evidence_cards_manifest.json"),
        "locked_test_report_sha256": sha256(report_path),
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    for r in runs:
        print(r["seed"], "n", r["accounts"], "source", r["source_locator_concepts"],
              "normality_changed", r["human_normality_only_mask_changes"],
              "net_correct", r["net_correct_supported_by_human_normality"])
    print(OUT)


if __name__ == "__main__":
    main()
