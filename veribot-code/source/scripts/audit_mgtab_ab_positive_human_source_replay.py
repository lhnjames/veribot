"""Independently recompute 9 Human scores and all frozen MGTAB test probabilities."""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.compact_source_population import load_source_population
from verifiable_bot.mgtab_full_atomic_runtime import AB_RELATIONS
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_mgtab_ab_group_positive_human"


def digest_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    value = np.ascontiguousarray(array, dtype=np.float64)
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((FOLDER / "inference_manifest.json")
                          .read_text(encoding="utf-8"))
    data_dir = resolve_mgtab_source_dir(ROOT)
    rows = []
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    for seed, entry in enumerate(manifest["runs"]):
        source_meta = ROOT / entry["source_metadata_path"]
        facts, names, ids = load_source_population(
            "mgtab", data_dir,
            ROOT / f"output/v3_mgtab_ab_spline_content/selected/seed{seed}",
            structural_facts=ROOT / "data/processed/mgtab_v3/structural_facts.npz",
            neighbor_facts=ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
            inference_metadata=source_meta)
        reference = LabelBlindReference.load(ROOT / entry["reference_path"])
        model = joblib.load(ROOT / entry["model_path"])
        model.set_params(n_jobs=1)
        frozen = json.loads((FOLDER / f"seed{seed}/test_predictions.json")
                            .read_text(encoding="utf-8"))
        expected = frozen["predictions"]
        positions = np.array([ids.index(item["account_id"]) for item in expected])
        if (tuple(names) != reference.names or
                reference.version != entry["reference_version"] or
                any(ids[i] != item["account_id"]
                    for i, item in zip(positions, expected, strict=True))):
            raise ValueError("published source population differs")
        with np.load(ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
                     allow_pickle=False) as cache:
            supports = cache["supports"][positions].astype(float)
        batch = reference.transform(facts[positions])
        reliability = batch.reliability.copy()
        reliability[:, 62:86] *= supports / (supports+5.0)
        high = np.log1p(batch.upper)*reliability
        low = np.log1p(batch.lower)*reliability
        base = np.concatenate((high, low), axis=1)
        groups = [list(range(20))]
        groups += [[j for j in range(20, 86)
                    if names[j].startswith(f"published_{relation}_")]
                   for relation in AB_RELATIONS]
        groups += [list(range(86, 150))]
        if (len(groups) != 9 or
                sorted(j for group in groups for j in group) !=
                list(range(150))):
            raise ValueError("source groups do not partition atomic facts")
        human_scores = np.empty((len(positions), 9), dtype=float)
        source_present_count = 0
        for k, members in enumerate(groups):
            present = batch.available[:, members].any(axis=1)
            source_present_count += int(present.sum())
            deviation = np.maximum(high[:, members].max(axis=1),
                                   low[:, members].max(axis=1))
            human_scores[:, k] = present.astype(float)*np.exp(-deviation)
        design = np.concatenate((base, human_scores), axis=1)
        p = model.predict_proba(design)[:, 1]
        saved_p = np.asarray([item["probability"] for item in expected])
        raw = np.asarray(model.booster_.predict(design, raw_score=True))
        max_p = float(np.max(np.abs(p-saved_p)))
        max_link = float(np.max(np.abs(expit(raw)-p)))
        if (design.shape != (1020, 309) or max_p > 1e-12 or
                max_link > 1e-12 or np.any(human_scores < 0) or
                np.any(human_scores > 1)):
            raise ValueError("independent Human score or prediction differs")
        row = {
            "seed": seed, "test_accounts": len(positions),
            "human_score_cells_recomputed": int(human_scores.size),
            "source_present_group_account_cells": source_present_count,
            "human_score_matrix_sha256": digest_array(human_scores),
            "model_input_matrix_sha256": digest_array(design),
            "max_saved_probability_difference": max_p,
            "max_logistic_link_difference": max_link,
            "reference_version": reference.version,
        }
        rows.append(row)
        print(row, flush=True)
    output = (ROOT /
        "research/certificates/mgtab_ab_positive_human_source_replay.json")
    output.write_text(json.dumps({
        "scope": "independent_published_source_to_nine_positive_human_scores_to_frozen_probability",
        "runs": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
