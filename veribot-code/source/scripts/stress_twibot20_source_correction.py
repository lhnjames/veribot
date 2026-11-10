"""Frozen-model source correction test for injected TwiBot-20 verified-field errors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import ijson
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.twibot20_facts import _boolean
from verifiable_bot.twibot20_source_certificate import sha256
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


SALT = "veribot-v3-verified-measurement-v1:"
FIELD = "verified"
FRACTION = 0.20


def select_indices(ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    eligible = np.flatnonzero(np.isfinite(values) & np.isin(values, [0.0, 1.0]))
    scored = sorted(eligible, key=lambda i: hashlib.sha256(
        (SALT + str(ids[i])).encode("utf-8")).digest())
    return np.sort(np.asarray(scored[:round(FRACTION * len(scored))], dtype=int))


def source_values(archive: Path, selected_ids: set[str]) -> dict[str, tuple[float, int]]:
    found = {}
    with zipfile.ZipFile(archive) as zipped, zipped.open("Twibot-20/test.json") as stream:
        for ordinal, row in enumerate(ijson.items(stream, "item")):
            uid = str(row["ID"])
            if uid in selected_ids:
                if uid in found:
                    raise ValueError("duplicate selected account in official test member")
                found[uid] = (_boolean((row.get("profile") or {}).get(FIELD)), ordinal)
    if set(found) != selected_ids:
        raise ValueError("selected account missing from official test member")
    return found


def counts(y: np.ndarray, probability: np.ndarray, threshold: float,
           selected: np.ndarray) -> dict:
    error = (probability >= threshold) != y
    return {"all_errors": int(error.sum()),
            "selected_errors": int(error[selected].sum()),
            "other_errors": int(error.sum() - error[selected].sum())}


def run_seed(facts: Path, model_dir: Path, screen_dir: Path, seed: int,
             archive_sha: str, raw: dict[str, tuple[float, int]]) -> dict:
    with np.load(facts, allow_pickle=False) as cache:
        all_x = cache["values"].astype(float)
        all_y = cache["labels"].astype(int)
        all_ids = cache["ids"].astype(str)
        all_split = cache["split"].astype(str)
        all_contexts = cache["contexts"].astype(str)
        names = cache["names"].astype(str).tolist()
    test = np.flatnonzero(all_split == "test")
    x, y, ids, contexts = (all_x[test].copy(), all_y[test], all_ids[test],
                            all_contexts[test])
    selected = select_indices(ids, x[:, names.index(FIELD)])
    selected_ids = {str(ids[i]) for i in selected}
    if selected_ids != set(raw):
        raise ValueError("selected account set differs from source extraction")
    for i in selected:
        if x[i, names.index(FIELD)] != raw[str(ids[i])][0]:
            raise ValueError("original fact cache disagrees with raw source")
    report = json.loads((model_dir / f"seed{seed}" / "report.json").read_text(encoding="utf-8"))
    manifest = report["manifest"]
    if manifest["facts_sha256"] != sha256(facts) or manifest["archive_sha256"] != archive_sha:
        raise ValueError("frozen model source differs from current inputs")
    ref_path = screen_dir / f"seed{seed}" / "reference.json.gz"
    if manifest["reference_artifact_sha256"] != sha256(ref_path):
        raise ValueError("frozen reference artifact differs")
    reference = LabelBlindReference.load(ref_path)
    model = MonotoneSplineReasoner.from_manifest(report["reasoner"])
    if model.reference_version != reference.version:
        raise ValueError("frozen model and reference differ")
    threshold = float(manifest["threshold"])
    clean = model.predict_proba(reference.transform(x, contexts=contexts))
    saved = json.loads((model_dir / f"seed{seed}" / "test_predictions.json").read_text(
        encoding="utf-8"))["predictions"]
    if [str(ids[i]) for i in range(len(ids))] != [row["account_id"] for row in saved] or \
            np.max(np.abs(clean - np.array([row["probability"] for row in saved]))) > 1e-12:
        raise ValueError("clean scores differ from frozen test output")
    corrupted_x = x.copy()
    corrupted_x[selected, names.index(FIELD)] = 1.0 - corrupted_x[selected, names.index(FIELD)]
    corrupted = model.predict_proba(reference.transform(corrupted_x, contexts=contexts))
    corrected_x = corrupted_x.copy()
    for i in selected:
        corrected_x[i, names.index(FIELD)] = raw[str(ids[i])][0]
    corrected = model.predict_proba(reference.transform(corrected_x, contexts=contexts))
    if not np.array_equal(corrected_x, x, equal_nan=True) or \
            np.max(np.abs(corrected - clean)) > 1e-12:
        raise ValueError("source correction failed to restore the clean fact inputs")
    corrupted_error = (corrupted >= threshold) != y
    corrected_error = (corrected >= threshold) != y
    removed = int(np.sum(corrupted_error[selected] & ~corrected_error[selected]))
    introduced = int(np.sum(~corrupted_error[selected] & corrected_error[selected]))
    selected_hash = hashlib.sha256(json.dumps(sorted(selected_ids)).encode()).hexdigest()
    return {
        "seed": seed, "field": FIELD, "selected_accounts": len(selected),
        "selected_id_sha256": selected_hash, "threshold": threshold,
        "clean": counts(y, clean, threshold, selected),
        "corrupted": counts(y, corrupted, threshold, selected),
        "source_corrected": counts(y, corrected, threshold, selected),
        "correction_removed_errors": removed,
        "correction_introduced_errors": introduced,
        "direct_account_label_same_account_count": {
            "actions": len(selected),
            "all_errors": int(corrupted_error.sum() - corrupted_error[selected].sum()),
            "comparison_unit": "account actions, not measured human time",
        },
        "rows": [{"account_id": str(ids[i]), "source_locator":
                   f"Twibot-20/test.json[{raw[str(ids[i])][1]}].profile.{FIELD}",
                  "source_value": raw[str(ids[i])][0],
                  "injected_value": float(corrupted_x[i, names.index(FIELD)]),
                  "label_for_evaluation": int(y[i]),
                  "clean_probability": float(clean[i]),
                  "corrupted_probability": float(corrupted[i]),
                  "corrected_probability": float(corrected[i])}
                 for i in selected],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    archive_sha = sha256(args.zip)
    with np.load(args.facts, allow_pickle=False) as cache:
        mask = cache["split"].astype(str) == "test"
        ids = cache["ids"][mask].astype(str)
        values = cache["values"][mask, cache["names"].astype(str).tolist().index(FIELD)]
    selected = select_indices(ids, values)
    raw = source_values(args.zip, {str(ids[i]) for i in selected})
    runs = [run_seed(args.facts, args.model_dir, args.screen_dir, seed, archive_sha, raw)
            for seed in args.seeds]
    report = {
        "protocol": "research/VeriBot_v3_TwiBot20来源纠错压力测试协议_2026-11-01.md",
        "dataset": "twibot20", "source_archive_sha256": archive_sha,
        "fact_cache_sha256": sha256(args.facts),
        "injection": {"field": FIELD, "fraction": FRACTION, "selection_salt": SALT,
                      "selection_uses_class_labels": False},
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runs": [
        {key: value for key, value in row.items() if key != "rows"} for row in runs
    ]}, indent=2))


if __name__ == "__main__":
    main()
