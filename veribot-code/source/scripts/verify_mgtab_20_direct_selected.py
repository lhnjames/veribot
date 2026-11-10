"""Independently replay 20 MGTAB facts from released tensors to all decisions."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_mgtab_sparse_evidence_cards import locator
from gate0_shallow_tree import evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_20_train_selected_concepts import rank_atomic, rank_content
from screen_mgtab_content_projection_evidence import fingerprint, transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def spline_contribution(score: float, reliability: float, sign: int,
                        knots: list[float], coefficients: list[float]) -> float:
    value = float(np.log1p(score) * reliability)
    previous = total = 0.0
    for knot, coefficient in zip(knots, coefficients):
        total += min(max(value - previous, 0.0), knot - previous) * sign * coefficient
        previous = knot
    return total + max(value - previous, 0.0) * sign * coefficients[-1]


def verify_seed(args: argparse.Namespace, seed: int, published: np.ndarray) -> dict:
    original_dir = args.selected_model_dir / f"seed{seed}"
    directory = args.output_dir / f"seed{seed}"
    original = json.loads((original_dir / "report.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    saved = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = report["manifest"]
    if saved["manifest"] != manifest:
        raise ValueError("report and saved prediction manifests differ")
    x, y, names, split = selected_facts(args, seed, original)
    train, test = split["train"], split["test"]
    if (manifest["split_sha256"] != original["manifest"]["split_sha256"]
            or manifest["source_files_sha256"] != original["manifest"]["files_sha256"]
            or manifest["concept_count"] != 20
            or sha256_file(args.lock) != manifest["selection_lock_sha256"]
            or sha256_file(original_dir / "projection.npz") !=
            manifest["projection_artifact_sha256"]):
        raise ValueError("fixed source, split or selection differs")
    normalized_scaler = StandardScaler().fit(published[train, 20:])
    normalized = normalized_scaler.transform(published[:, 20:])
    pca = PCA(n_components=64, svd_solver="randomized",
              random_state=seed).fit(normalized[train])
    if fingerprint(normalized_scaler.mean_, normalized_scaler.scale_,
                   pca.mean_, pca.components_) != manifest["projection_fingerprint"]:
        raise ValueError("train-only PCA does not refit from published tensor")
    projected = pca.transform(normalized[test])
    if np.max(np.abs(projected - x[test, 86:])) > 1e-10:
        raise ValueError("saved content projections differ from published tensor")
    reference_path = original_dir / "reference.json.gz"
    reference = LabelBlindReference.load(reference_path)
    refit = LabelBlindReference(min_context_n=reference.min_context_n,
                                support_prior=reference.support_prior).fit(x[train], names)
    if (sha256_file(reference_path) !=
            original["manifest"]["reference_artifact_sha256"]
            or refit.version != reference.version
            or reference.version != manifest["reference_version"]):
        raise ValueError("reference does not refit from released training facts")
    a, _ = train_test_split(np.arange(len(train)), test_size=0.5,
                            stratify=y[train], random_state=seed)
    atomic_train = transform_with_support(reference, x[train],
                                          split["supports"][train])
    selected = rank_atomic(atomic_train, y[train], a, list(range(86)), 4) + \
        rank_content(atomic_train, y[train], a, 16)
    if (selected != manifest["selected_fact_indices"]
            or [names[j] for j in selected] != manifest["selected_fact_names"]):
        raise ValueError("A-only fact selection does not replay")
    atomic_test = transform_with_support(reference, x[test],
                                         split["supports"][test])
    model = report["reasoner"]
    card_path = directory / "evidence_cards.jsonl.gz"
    if sha256_file(card_path) != report["cards_sha256"]:
        raise ValueError("saved evidence cards changed")
    predictions = saved["predictions"]
    largest_fact = largest_term = largest_probability = 0.0
    cards = 0
    with gzip.open(card_path, "rt", encoding="utf-8") as stream:
        for offset, line in enumerate(stream):
            if offset >= len(test):
                raise ValueError("too many test cards")
            card = json.loads(line)
            index = int(test[offset])
            prediction = predictions[offset]
            if (card["account_id"] != str(index)
                    or prediction["account_id"] != str(index)
                    or card["seed"] != seed
                    or card["source_features_sha256"] !=
                    manifest["source_features_sha256"]
                    or card["reference_version"] != reference.version
                    or card["selection_lock_sha256"] !=
                    manifest["selection_lock_sha256"]
                    or len(card["terms"]) != 20):
                raise ValueError("card account, source or vocabulary differs")
            logit = float(model["intercept"])
            contributions = {}
            for term, fact_index in zip(card["terms"], selected, strict=True):
                name = names[fact_index]
                raw = float(x[index, fact_index])
                source_locator = locator(fact_index, name, index,
                                         manifest["projection_fingerprint"])
                if (term["name"] != name or term["source_locator"] != source_locator
                        or term["available"] != bool(atomic_test.available[offset, fact_index])):
                    raise ValueError("term source locator or availability differs")
                fact_difference = abs(term["source_value"] - raw)
                largest_fact = max(largest_fact, fact_difference)
                if fact_difference > 1e-10:
                    raise ValueError("term value differs from published tensor")
                for key, expected in (("upper_tail_score", atomic_test.upper[offset, fact_index]),
                                      ("lower_tail_score", atomic_test.lower[offset, fact_index]),
                                      ("reliability", atomic_test.reliability[offset, fact_index])):
                    if abs(term[key] - expected) > 1e-12:
                        raise ValueError("reference evidence differs")
                direction = model["directions"][name]
                if (term["selected_tail"] != direction["tail"]
                        or term["fixed_direction"] != direction["sign"]
                        or term["knots"] != model["knots"][name]
                        or term["nonnegative_slopes"] != model["coefficients"][name]):
                    raise ValueError("fixed reasoner differs from card")
                score = (atomic_test.upper if direction["tail"] == "upper"
                         else atomic_test.lower)[offset, fact_index]
                value = spline_contribution(score,
                                            atomic_test.reliability[offset, fact_index],
                                            direction["sign"], model["knots"][name],
                                            model["coefficients"][name])
                term_difference = abs(value - term["logit_contribution"])
                largest_term = max(largest_term, term_difference)
                if term_difference > 1e-12:
                    raise ValueError("logit term does not replay")
                contributions[name] = value
                logit += value
            probability = float(expit(logit))
            probability_difference = max(abs(logit - card["logit"]),
                                         abs(probability - card["probability"]),
                                         abs(probability - prediction["probability"]))
            largest_probability = max(largest_probability, probability_difference)
            if (probability_difference > 1e-12 or prediction["label"] != int(y[index])
                    or card["prediction"] != int(probability >= manifest["threshold"])):
                raise ValueError("card does not reconstruct fixed decision")
            for group, members in model["groups"].items():
                value = sum(contributions[name] for name in members)
                if (abs(value - card["group_contributions"][group]) > 1e-12
                        or abs(value - prediction["group_contributions"][group]) > 1e-12):
                    raise ValueError("group contribution differs")
            cards += 1
    if cards != len(test) or cards != len(predictions):
        raise ValueError("card count differs from fixed test")
    probabilities = np.asarray([row["probability"] for row in predictions])
    metrics = evaluate(y[test], probabilities, manifest["threshold"])
    if any(abs(metrics[key] - report["test"][key]) > 1e-12
           for key in ("acc", "macro_f1", "auroc")):
        raise ValueError("test metric differs from independently read predictions")
    return {"seed": seed, "test_accounts": cards, "fact_terms": 20 * cards,
            "cards_sha256": sha256_file(card_path),
            "model_report_sha256": sha256_file(directory / "report.json"),
            "source_audit_sha256": sha256_file(args.source_audit),
            "max_source_fact_difference": largest_fact,
            "max_term_difference": largest_term,
            "max_logit_or_probability_difference": largest_probability,
            "reference_refit_from_released_train": True,
            "projection_refit_from_released_train": True,
            "status": "all_published_tensor_to_evidence_to_decision_cards_replayed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--structural-facts", required=True, type=Path)
    parser.add_argument("--neighbor-facts", required=True, type=Path)
    parser.add_argument("--selected-model-dir", required=True, type=Path)
    parser.add_argument("--source-audit", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.dataset = "mgtab"
    args.account_file = None
    args.native_graph_facts = None
    args.fox8_native_facts = None
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    original = json.loads((args.selected_model_dir / "seed0" / "report.json")
                          .read_text(encoding="utf-8"))
    if (audit["status"] != "verified_all_published_tensor_derived_fact_cells"
            or sha256_file(args.source_audit) !=
            original["manifest"]["source_audit_sha256"]):
        raise ValueError("independent published-tensor source audit differs")
    published = torch.load(args.data_dir / "features.pt", map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    if published.shape != (10199, 788):
        raise ValueError("released tensor shape differs")
    result = {"dataset": "mgtab", "scope": "all_test_20_direct_source_fact_cards",
              "source_level": "published_profile_edge_and_mean_tweet_tensors",
              "runs": [verify_seed(args, seed, published) for seed in (0, 1, 2)]}
    output = args.output_dir / "independent_verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
