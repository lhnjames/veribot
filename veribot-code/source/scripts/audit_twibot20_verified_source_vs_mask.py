"""Audit all-test verified source toggles against encoded evidence masking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stress_twibot20_source_correction import source_values
from train_v3_twibot20 import sha256
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--selected-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.selected_dir / "report.json").read_text(encoding="utf-8"))
    manifest = report["manifest"]
    paths = (("source_archive_sha256", args.zip),
             ("base_facts_sha256", args.facts),
             ("model_artifact_sha256", args.artifact),
             ("reference_artifact_sha256", args.reference))
    for key, path in paths:
        if sha256(path) != manifest[key]:
            raise ValueError(f"Frozen artifact/source mismatch: {key}")
    with np.load(args.facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        splits = cache["split"].astype(str)
        positions = cache["positions"].astype(int)
        labels = cache["labels"].astype(int)
        values = cache["values"].astype(float)
        contexts = cache["contexts"].astype(str)
        names = cache["names"].astype(str).tolist()
    test = np.flatnonzero(splits == "test")
    ids_test, y_test, x, context_test = ids[test], labels[test], values[test].copy(), contexts[test]
    positions_test = positions[test]
    raw = source_values(args.zip, set(ids_test.tolist()))
    j = names.index("verified")
    if len(raw) != len(test):
        raise ValueError("Official test record count differs")
    for i, uid in enumerate(ids_test):
        value, position = raw[uid]
        if not np.isclose(x[i, j], value, equal_nan=True) or positions_test[i] != position:
            raise ValueError(f"Fact or position mismatch: {uid}")
    with np.load(args.artifact, allow_pickle=False) as model:
        coefficient = model["coefficient"].astype(float)
        intercept = float(model["intercept"][0])
        threshold = float(model["threshold"][0])
    if threshold != manifest["threshold"] or len(coefficient[:44]) != 2 * len(names):
        raise ValueError("Frozen decision contract differs")
    reference = LabelBlindReference.load(args.reference)
    if reference.version != manifest["reference_version"] or reference.names != tuple(names):
        raise ValueError("Frozen reference differs")
    saved = json.loads((args.selected_dir / "test_predictions.json").read_text(
        encoding="utf-8"))["predictions"]
    if [row["account_id"] for row in saved] != ids_test.tolist():
        raise ValueError("Frozen prediction account order differs")
    lexical = np.asarray([row["group_contributions"]["unigram_evidence"]
                          for row in saved], dtype=float)

    def encoded(facts: np.ndarray) -> np.ndarray:
        evidence = reference.transform(facts, contexts=context_test)
        return np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                                np.log1p(evidence.lower) * evidence.reliability))

    original = encoded(x)
    base_logit = intercept + original @ coefficient[:44] + lexical
    p_base = expit(base_logit)
    saved_p = np.asarray([row["probability"] for row in saved], dtype=float)
    saved_structured = np.asarray([row["group_contributions"]["structured_evidence"]
                                   for row in saved], dtype=float)
    replay_error = float(np.max(np.abs(p_base - saved_p)))
    structured_error = float(np.max(np.abs(original @ coefficient[:44] - saved_structured)))
    if replay_error > 1e-12 or structured_error > 1e-12:
        raise ValueError("Original scores do not replay frozen predictions")
    eligible = np.flatnonzero(np.isin(x[:, j], [0.0, 1.0]))
    edited = x.copy()
    edited[eligible, j] = 1.0 - edited[eligible, j]
    changed = encoded(edited)
    other = np.ones(original.shape[1], dtype=bool)
    other[[j, j + len(names)]] = False
    if not np.allclose(original[:, other], changed[:, other], rtol=0, atol=0):
        raise ValueError("Verified source toggle changed an unrelated encoded coordinate")
    source_logit = intercept + changed @ coefficient[:44] + lexical
    mask_logit = base_logit - (original[:, j] * coefficient[j]
                               + original[:, j + len(names)] * coefficient[j + len(names)])
    p_source, p_mask = expit(source_logit), expit(mask_logit)
    base_class, source_class, mask_class = (p_base >= threshold,
                                           p_source >= threshold, p_mask >= threshold)
    source_changes = np.flatnonzero((base_class != source_class) & np.isin(np.arange(len(test)), eligible))
    mask_changes = np.flatnonzero((base_class != mask_class) & np.isin(np.arange(len(test)), eligible))
    disagreement = np.flatnonzero((source_class != mask_class) & np.isin(np.arange(len(test)), eligible))
    by_source_value = {}
    for value in (0.0, 1.0):
        group = eligible[x[eligible, j] == value]
        by_source_value[str(int(value))] = {
            "count": len(group),
            "source_toggle_class_changes": int(np.count_nonzero(base_class[group] != source_class[group])),
            "encoded_mask_class_changes": int(np.count_nonzero(base_class[group] != mask_class[group])),
            "toggle_vs_mask_class_disagreements": int(np.count_nonzero(source_class[group] != mask_class[group])),
        }
    rows = [{
        "account_id": str(ids_test[i]),
        "source_locator": f"Twibot-20/test.json[{raw[str(ids_test[i])][1]}].profile.verified",
        "source_value": int(x[i, j]),
        "toggled_value": int(edited[i, j]),
        "label_for_evaluation": int(y_test[i]),
        "original_logit": float(base_logit[i]),
        "source_toggled_logit": float(source_logit[i]),
        "encoded_mask_logit": float(mask_logit[i]),
        "original_probability": float(p_base[i]),
        "source_toggled_probability": float(p_source[i]),
        "encoded_mask_probability": float(p_mask[i]),
        "original_class": int(base_class[i]),
        "source_toggled_class": int(source_class[i]),
        "encoded_mask_class": int(mask_class[i]),
        "verified_upper_encoded": float(original[i, j]),
        "verified_lower_encoded": float(original[i, j + len(names)]),
        "verified_upper_after_source_toggle": float(changed[i, j]),
        "verified_lower_after_source_toggle": float(changed[i, j + len(names)]),
    } for i in eligible]
    result = {
        "scope": "TwiBot-20 official test; all finite binary profile.verified accounts",
        "operation": "in-memory raw-source verified toggle vs removing only two encoded verified tail terms",
        "causal_scope": "controlled input sensitivity; no claim that source values were erroneous",
        "source_archive_sha256": manifest["source_archive_sha256"],
        "base_facts_sha256": manifest["base_facts_sha256"],
        "model_artifact_sha256": manifest["model_artifact_sha256"],
        "reference_artifact_sha256": manifest["reference_artifact_sha256"],
        "test_accounts": len(test),
        "eligible_accounts": len(eligible),
        "threshold": threshold,
        "original_replay_max_probability_error": replay_error,
        "original_replay_max_structured_contribution_error": structured_error,
        "source_toggle_class_changes": len(source_changes),
        "encoded_mask_class_changes": len(mask_changes),
        "toggle_vs_mask_class_disagreements": len(disagreement),
        "source_toggle_score_changes": int(np.count_nonzero(np.abs(source_logit[eligible] - base_logit[eligible]) > 1e-12)),
        "encoded_mask_score_changes": int(np.count_nonzero(np.abs(mask_logit[eligible] - base_logit[eligible]) > 1e-12)),
        "original_errors_all_test": int(np.count_nonzero(base_class != y_test)),
        "source_toggled_errors_all_test": int(np.count_nonzero(source_class != y_test)),
        "encoded_mask_errors_all_test": int(np.count_nonzero(mask_class != y_test)),
        "by_source_value": by_source_value,
        "largest_absolute_toggle_mask_logit_difference": float(np.max(np.abs(source_logit[eligible] - mask_logit[eligible]))),
        "example_disagreement": rows[next(k for k, r in enumerate(rows)
                                          if r["source_toggled_class"] != r["encoded_mask_class"])]
                                if len(disagreement) else None,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
