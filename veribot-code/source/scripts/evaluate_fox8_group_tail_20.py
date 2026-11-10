"""Evaluate the validation-locked Fox8 twenty-concept source bottleneck."""

from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def mask(evidence: EvidenceBatch, columns: list[int]) -> EvidenceBatch:
    high, low = evidence.upper.copy(), evidence.lower.copy()
    high[:, columns] = 0.0
    low[:, columns] = 0.0
    return EvidenceBatch(high, low, evidence.reliability, evidence.available,
                         evidence.reference_version, evidence.names)


def source_cards(source_dir: Path, seed: int, ids: list[str]) -> tuple[list[dict], str]:
    manifest = json.loads((source_dir / "evidence_cards_manifest.json")
                          .read_text(encoding="utf-8"))
    verification = json.loads((source_dir / "verification.json")
                              .read_text(encoding="utf-8"))
    entry = next(row for row in manifest["runs"] if row["seed"] == seed)
    audit = next(row for row in verification["runs"] if row["seed"] == seed)
    path = source_dir / f"seed{seed}_evidence_cards.jsonl.gz"
    digest = sha256_file(path)
    if (digest != entry["cards_sha256"]
            or audit["status"] != "verified_all_source_to_logit_cards"
            or audit["fact_cells"] != len(ids) * 43):
        raise ValueError("original Fox8 raw-source audit differs")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        cards = [json.loads(line) for line in stream]
    if [card["account_id"] for card in cards] != ids:
        raise ValueError("original source-card account order differs")
    return cards, digest


def evaluate_seed(args: argparse.Namespace, seed: int, screen: dict, lock: dict,
                  groups: dict) -> dict:
    original_dir = args.full_model_dir / f"seed{seed}"
    original = json.loads((original_dir / "report.json").read_text(encoding="utf-8"))
    x, y, ids, names, split, source = load("fox8", args.data_dir, None, seed)
    with np.load(args.native_facts, allow_pickle=False) as cache:
        if cache["ids"].astype(str).tolist() != ids:
            raise ValueError("native interaction account order differs")
        names += cache["names"].astype(str).tolist()
        x = np.column_stack((x, cache["values"].astype(float)))
    if (source["split_sha256"] != original["manifest"]["split_sha256"]
            or source["files_sha256"] != original["manifest"]["files_sha256"]
            or sha256_file(args.native_facts) != original["manifest"]["native_facts_sha256"]
            or tuple(names) != tuple(original["reasoner"]["directions"])):
        raise ValueError("original source facts or fixed split differ")
    reference_path = original_dir / "reference.json.gz"
    if sha256_file(reference_path) != original["manifest"]["reference_artifact_sha256"]:
        raise ValueError("original reference artifact differs")
    reference = LabelBlindReference.load(reference_path)
    if reference.version != original["manifest"]["reference_version"]:
        raise ValueError("reference version differs")
    train, dev, test = split["train"], split["val"], split["test"]
    high_train, _ = collapse_group_tails(reference.transform(x[train]), groups)
    high_dev, _ = collapse_group_tails(reference.transform(x[dev]), groups)
    high_test, winners = collapse_group_tails(reference.transform(x[test]), groups)
    if len(high_train.names) != 20 or not (
            high_train.reference_version == high_dev.reference_version ==
            high_test.reference_version):
        raise ValueError("twenty-concept vocabulary differs")
    concept_groups = {name: [f"{name}:high_tail_max", f"{name}:low_tail_max"]
                      for name in groups}
    a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                            stratify=y[train], random_state=seed)
    model = MonotoneSplineReasoner(concept_groups, l2=lock["selected_l2"])
    model.fit(rows(high_train, a), y[train][a], rows(high_train, b), y[train][b])
    dev_probability = model.predict_proba(high_dev)
    threshold = best_validation_threshold(dev_probability, y[dev])
    validation = evaluate(y[dev], dev_probability, threshold)
    expected = next(row for row in screen["runs"] if row["seed"] == seed)[
        "alternatives"][str(lock["selected_l2"])]
    if (any(abs(validation[key] - expected["validation"][key]) > 1e-12
            for key in ("acc", "macro_f1", "auroc"))
            or abs(threshold - expected["threshold"]) > 1e-12):
        raise ValueError("validation result differs from selection lock")
    probability = model.predict_proba(high_test)
    design = model._design(high_test)
    per_concept = np.column_stack([
        design[:, model._slices[name]] @ model.coef[model._slices[name]]
        for name in high_test.names])
    logit = model.intercept + per_concept.sum(axis=1)
    if np.max(np.abs(expit(logit) - probability)) > 1e-12:
        raise ValueError("concepts do not reconstruct probability")
    group_contributions = model.contributions(high_test)
    group_mask_changes = {}
    for group, members in concept_groups.items():
        indices = [high_test.names.index(name) for name in members]
        after = model.predict_proba(mask(high_test, indices)) >= threshold
        group_mask_changes[group] = int(np.count_nonzero(after != (probability >= threshold)))
    intercept_probability = model.predict_proba(mask(high_test,
                                                     list(range(20))))
    if np.max(np.abs(intercept_probability - expit(model.intercept))) > 1e-12:
        raise ValueError("non-concept bypass exists")
    test_ids = [ids[int(i)] for i in test]
    original_cards, original_cards_sha256 = source_cards(
        args.source_cards_dir, seed, test_ids)
    out = args.output_dir / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    cards_path = out / "concept_evidence_cards.jsonl.gz"
    predictions = []
    with cards_path.open("wb") as binary:
        with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as stream:
                for offset, index in enumerate(test):
                    old = original_cards[offset]
                    if [term["name"] for term in old["terms"]] != names:
                        raise ValueError("audited source vocabulary differs")
                    concepts = []
                    for j, name in enumerate(high_test.names):
                        winner = int(winners[name][offset])
                        term = old["terms"][winner] if winner >= 0 else None
                        raw = float(x[int(index), winner]) if winner >= 0 else None
                        if term is not None and (term["source_value"] is None or
                                abs(term["source_value"] - raw) > 1e-12):
                            raise ValueError("winning raw source differs")
                        concepts.append({
                            "name": name, "available": bool(high_test.available[offset, j]),
                            "winning_atomic_fact": names[winner] if winner >= 0 else None,
                            "source_locator": term["source_locator"] if term else None,
                            "raw_source_value": raw,
                            "concept_value": float(high_test.upper[offset, j]),
                            "fixed_direction": int(model.sign[j]),
                            "spline_coefficients": model.coef[model._slices[name]].tolist(),
                            "logit_contribution": float(per_concept[offset, j]),
                        })
                    card = {
                        "dataset": "fox8", "seed": seed,
                        "account_id": ids[int(index)], "source_level": old["source_level"],
                        "source_hashes": old["source_hashes"],
                        "source_certificate_sha256": original_cards_sha256,
                        "concept_version": high_test.reference_version,
                        "intercept": float(model.intercept), "concepts": concepts,
                        "group_contributions": {group: float(value[offset])
                                                for group, value in group_contributions.items()},
                        "logit": float(logit[offset]),
                        "probability": float(probability[offset]),
                        "threshold": float(threshold),
                        "prediction": int(probability[offset] >= threshold),
                    }
                    stream.write(json.dumps(card, separators=(",", ":")) + "\n")
                    predictions.append({
                        "account_id": ids[int(index)], "label": int(y[index]),
                        "probability": float(probability[offset]), "split": "test",
                        "group_contributions": card["group_contributions"],
                    })
    manifest = {"dataset": "fox8", "seed": seed,
                "split_sha256": source["split_sha256"],
                "source_files_sha256": source["files_sha256"],
                "native_facts_sha256": sha256_file(args.native_facts),
                "reference_version": reference.version,
                "concept_version": high_test.reference_version,
                "concept_count": 20, "concept_names": list(high_test.names),
                "selection_lock_sha256": sha256_file(args.lock),
                "validation_screen_sha256": sha256_file(args.screen),
                "threshold": float(threshold),
                "prior_test_results_on_same_population_already_observed": True}
    report = {"manifest": manifest, "reasoner": model.manifest(),
              "validation": validation, "test": evaluate(y[test], probability, threshold),
              "group_mask_prediction_changes": group_mask_changes,
              "all_evidence_mask_probability": float(expit(model.intercept)),
              "test_accounts": len(test), "cards_sha256": sha256_file(cards_path)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "test_predictions.json").write_text(json.dumps({
        "manifest": manifest, "predictions": predictions}, indent=2) + "\n", encoding="utf-8")
    return {"seed": seed, "validation": validation, "test": report["test"],
            "group_mask_prediction_changes": group_mask_changes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--native-facts", required=True, type=Path)
    parser.add_argument("--full-model-dir", required=True, type=Path)
    parser.add_argument("--source-cards-dir", required=True, type=Path)
    parser.add_argument("--groups-json", required=True, type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if (screen["dataset"] != "fox8" or lock["dataset"] != "fox8"
            or sha256_file(args.screen) != lock["validation_screen_sha256"]
            or sha256_file(args.groups_json) != lock["prespecified_groups_sha256"]
            or lock["selected_l2"] != 0.0001
            or screen["test_predictions_accessed"]):
        raise ValueError("Fox8 concept selection lock differs")
    groups = json.loads(args.groups_json.read_text(encoding="utf-8"))
    print(json.dumps([evaluate_seed(args, seed, screen, lock, groups)
                      for seed in (0, 1, 2)], indent=2))


if __name__ == "__main__":
    main()
