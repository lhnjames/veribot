"""Recompute current Fox8 concept winners after withholding each atomic fact.

This tests the frozen classifier's dependence on cited source facts. Withholding
one observed fact is a model-input intervention, not an edit to the raw graph.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screen_fox8_neighbor_atomic_evidence_capacity_dev import evidence_matrix
from screen_fox8_native_neighbor_profile_concepts_dev import weighted_atomic
from screen_mgtab_20_hybrid_concepts import columns
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_fox8_targeted_neighbor_20_concepts"
OUT = ROOT / "research/certificates/fox8_current_atomic_fact_withholding.json"
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def withheld(batch: EvidenceBatch, col: int) -> EvidenceBatch:
    high = batch.upper.copy()
    low = batch.lower.copy()
    reliability = batch.reliability.copy()
    available = batch.available.copy()
    high[:, col] = 0.0
    low[:, col] = 0.0
    reliability[:, col] = 0.0
    available[:, col] = False
    return EvidenceBatch(high, low, reliability, available,
                         batch.reference_version, batch.names)


def main() -> None:
    lock_path = FOLDER / "selection_lock_ab_group_tree.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    source_groups = json.loads((ROOT / "research/fox8_20_concept_groups.json")
                               .read_text(encoding="utf-8"))
    source_groups.pop(lock["selected"]["removed_group"])
    source_groups["native_neighbor_volume"] = [
        "native_interaction_neighbor_mean_statuses_count"]
    cache_path = ROOT / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
    if lock["neighbor_fact_sha256"] != sha256(cache_path):
        raise ValueError("native-neighbor source differs")
    with np.load(cache_path, allow_pickle=False) as cache:
        cache_ids = cache["ids"].astype(str).tolist()
        cache_support = cache["supports"].copy()
    cache_index = {uid: i for i, uid in enumerate(cache_ids)}
    card_manifest = json.loads((FOLDER / "evidence_cards_manifest.json")
                               .read_text(encoding="utf-8"))
    runs = []
    for seed in (0, 1, 2):
        run_dir = FOLDER / f"seed{seed}"
        model_path = run_dir / "ab_group_tree.joblib"
        reference_path = run_dir / "label_blind_reference.json.gz"
        evidence_path = run_dir / "ab_group_tree_test_evidence.npz"
        card_path = run_dir / "ab_group_tree_test_cards.jsonl.gz"
        prediction_path = run_dir / "ab_group_tree_test_predictions.json"
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        manifest = prediction["manifest"]
        if (manifest["selection_lock_sha256"] != sha256(lock_path) or
                manifest["model_sha256"] != sha256(model_path) or
                manifest["reference_sha256"] != sha256(reference_path) or
                manifest["evidence_sha256"] != sha256(evidence_path) or
                card_manifest["runs"][seed]["sha256"] != sha256(card_path)):
            raise ValueError("frozen model or source card differs")
        with np.load(evidence_path, allow_pickle=False) as saved:
            ids = saved["ids"].astype(str).tolist()
            names = saved["retained_fact_names"].astype(str).tolist()
            raw = saved["retained_raw_values"].copy()
            concept_names = saved["concept_names"].astype(str).tolist()
            concept_scores = saved["concept_scores"].copy()
            source_winners = saved["winning_fact_index"].copy()
            x = saved["model_input"].copy()
            probability = saved["probability"].copy()
            labels = saved["label"].astype(int)
        with gzip.open(card_path, "rt", encoding="utf-8") as stream:
            cards = [json.loads(line) for line in stream]
        if len(ids) != len(cards) or len(ids) != len(set(ids)):
            raise ValueError("fixed account card population differs")
        reference = LabelBlindReference.load(reference_path)
        if reference.version != lock["reference_version_by_seed"][str(seed)]:
            raise ValueError("training reference differs")
        retained = [reference.names.index(name) for name in names]
        full_raw = np.full((len(ids), len(reference.names)), np.nan)
        full_raw[:, retained] = raw
        support = cache_support[[cache_index[uid] for uid in ids]]
        atomic = weighted_atomic(reference, full_raw, support)
        atomic = columns(atomic, retained)
        concept, winners = collapse_group_tails(atomic, source_groups)
        reconstructed = evidence_matrix(concept)
        if (tuple(concept.names) != tuple(concept_names) or
                np.max(np.abs(concept.upper-concept_scores)) > 1e-12 or
                np.max(np.abs(reconstructed-x)) > 1e-12 or
                any(np.any(winners[name] != source_winners[:, k])
                    for k, name in enumerate(concept_names))):
            raise ValueError("source facts fail exact concept reconstruction")
        model = joblib.load(model_path)
        model.set_params(n_jobs=1)
        if np.max(np.abs(model.predict_proba(x)[:, 1]-probability)) > 1e-12:
            raise ValueError("frozen model prediction differs")
        threshold = float(lock["threshold_by_seed"][str(seed)])
        full_prediction = probability >= threshold
        all_after = np.empty((len(ids), len(names)), dtype=float)
        all_winners = []
        for fact_index in range(len(names)):
            altered, replacement = collapse_group_tails(
                withheld(atomic, fact_index), source_groups)
            all_after[:, fact_index] = model.predict_proba(
                evidence_matrix(altered))[:, 1]
            all_winners.append(np.column_stack([
                replacement[name] for name in concept_names]))
        fact_changes = np.abs(all_after-probability[:, None]) > 1e-10
        fact_flips = (all_after >= threshold) != full_prediction[:, None]
        source_path = run_dir / "atomic_fact_withholding_cards.jsonl.gz"
        cited_fact_entries = 0
        cited_effectful_entries = 0
        cited_flip_entries = 0
        accounts_with_effectful_cited_fact = 0
        accounts_with_flip_cited_fact = 0
        native_neighbor_cited_effect = 0
        with source_path.open("wb") as binary:
            with gzip.GzipFile(filename="", fileobj=binary,
                               mode="wb", mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8") as stream:
                    for j, (uid, card) in enumerate(zip(ids, cards, strict=True)):
                        if (card["account_id"] != uid or
                                card["label"] != labels[j] or
                                card["prediction"] != int(full_prediction[j])):
                            raise ValueError("source card account or decision differs")
                        cited = sorted(set(int(k) for k in source_winners[j] if k >= 0))
                        records = []
                        for k in cited:
                            matching = [row for row in card["evidence"]
                                        if row["winning_source_fact"] == names[k]]
                            if not matching or any(row["source_locator"] is None
                                                   for row in matching):
                                raise ValueError("cited winning fact lacks source")
                            changed_concepts = [
                                concept_names[c] for c in range(20)
                                if int(all_winners[k][j, c]) !=
                                   int(source_winners[j, c])]
                            if not changed_concepts:
                                raise ValueError("withheld winner did not leave concept")
                            changed = bool(fact_changes[j, k])
                            flip = bool(fact_flips[j, k])
                            records.append({
                                "fact": names[k],
                                "source_locator": matching[0]["source_locator"],
                                "cited_by_concepts": [row["concept"] for row in matching],
                                "replacement_concepts": changed_concepts,
                                "probability_without_fact": float(all_after[j, k]),
                                "probability_changes_gt_1e_10": changed,
                                "class_changes": flip,
                            })
                            cited_fact_entries += 1
                            cited_effectful_entries += changed
                            cited_flip_entries += flip
                            if names[k].startswith("native_") and changed:
                                native_neighbor_cited_effect += 1
                        accounts_with_effectful_cited_fact += any(
                            r["probability_changes_gt_1e_10"] for r in records)
                        accounts_with_flip_cited_fact += any(
                            r["class_changes"] for r in records)
                        stream.write(json.dumps({
                            "account_id": uid, "label": int(labels[j]),
                            "prediction": int(full_prediction[j]),
                            "full_probability": float(probability[j]),
                            "threshold": threshold,
                            "cited_fact_withholding": records,
                        }, ensure_ascii=False, separators=(",", ":")) + "\n")
        runs.append({
            "seed": seed, "accounts": len(ids),
            "retained_atomic_facts": len(names),
            "atomic_fact_withholdings": int(all_after.size),
            "cited_unique_fact_entries": cited_fact_entries,
            "cited_entries_with_probability_change": cited_effectful_entries,
            "cited_entries_with_class_change": cited_flip_entries,
            "accounts_with_effectful_cited_fact": accounts_with_effectful_cited_fact,
            "accounts_with_class_changing_cited_fact": accounts_with_flip_cited_fact,
            "native_cited_entries_with_probability_change": native_neighbor_cited_effect,
            "all_facts_with_class_changes": int(np.count_nonzero(fact_flips)),
            "model_sha256": sha256(model_path),
            "reference_sha256": sha256(reference_path),
            "source_card_sha256": sha256(card_path),
            "withholding_cards_path": str(source_path.relative_to(ROOT)).replace("\\", "/"),
            "withholding_cards_sha256": sha256(source_path),
        })
        print(json.dumps({key: runs[-1][key] for key in (
            "seed", "cited_unique_fact_entries",
            "cited_entries_with_probability_change",
            "cited_entries_with_class_change",
            "accounts_with_effectful_cited_fact")}), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "current_fox8_exact_cited_atomic_fact_withholding",
        "intervention": "withhold one atomic fact at frozen inference, recompute group-tail winner and model output; raw graph unchanged",
        "source_cards_manifest_sha256": sha256(FOLDER / "evidence_cards_manifest.json"),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
