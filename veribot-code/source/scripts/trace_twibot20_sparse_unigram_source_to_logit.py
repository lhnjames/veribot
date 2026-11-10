"""Verify a sparse evidence decision from official tweets to saved logit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import ijson
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_sparse_active_unigrams import tokens
from train_v3_twibot20 import sha256
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--base-source-audit", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--active-cache", type=Path, required=True)
    parser.add_argument("--active-audit", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--source-top-k", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.model_dir / "report.json").read_text(encoding="utf-8"))
    saved = json.loads((args.model_dir / "test_predictions.json").read_text(
        encoding="utf-8"))
    manifest = report["manifest"]
    audit = json.loads(args.active_audit.read_text(encoding="utf-8"))
    base_audit = json.loads(args.base_source_audit.read_text(encoding="utf-8"))
    if (manifest["source_archive_sha256"] != sha256(args.zip)
            or manifest["base_facts_sha256"] != sha256(args.base_facts)
            or manifest["base_source_audit_sha256"] != sha256(args.base_source_audit)
            or manifest["model_artifact_sha256"] != sha256(args.artifact)
            or manifest["reference_artifact_sha256"] != sha256(args.reference)
            or manifest["active_source_audit_sha256"] != sha256(args.active_audit)
            or manifest["active_source_cache_sha256"] != sha256(args.active_cache)
            or audit["status"] !=
            "verified_train_idf_and_recomputed_all_active_unigram_values"
            or base_audit["status"] !=
            "verified_all_labeled_account_fact_cells_against_official_zip"):
        raise ValueError("official source or audit artifacts differ")
    with np.load(args.base_facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        parts = cache["split"].astype(str)
        positions = cache["positions"].astype(int)
        labels = cache["labels"].astype(int)
        contexts = cache["contexts"].astype(str)
        base_values = cache["values"].astype(float)
        names = cache["names"].astype(str).tolist()
    matches = np.flatnonzero((ids == args.account_id) & (parts == "test"))
    if len(matches) != 1:
        raise ValueError("account must occur once in official test")
    index = int(matches[0])
    with np.load(args.artifact, allow_pickle=False) as model:
        vocabulary = model["vocabulary"].astype(str).tolist()
        idf = model["idf"].astype(float)
        coefficient = model["coefficient"].astype(float)
        intercept = float(model["intercept"][0])
        threshold = float(model["threshold"][0])
    with np.load(args.active_cache, allow_pickle=False) as cache:
        if cache["ids"].astype(str).tolist() != ids.tolist():
            raise ValueError("active source account order differs")
        active = cache["active_indices"].astype(int)
        audited = cache["values"][index].astype(float)
    if not np.array_equal(active, np.flatnonzero(coefficient[44:] != 0)):
        raise ValueError("active term vocabulary differs")
    with zipfile.ZipFile(args.zip) as zipped:
        with zipped.open("Twibot-20/test.json") as stream:
            row = next(value for position, value in enumerate(
                ijson.items(stream, "item")) if position == positions[index])
    if str(row["ID"]) != args.account_id:
        raise ValueError("official account locator differs")
    tweets = row.get("tweet") or []
    tokenized = [tokens(tweet) for tweet in tweets]
    counts = Counter(term for tweet in tokenized for term in tweet)
    column = {term: j for j, term in enumerate(vocabulary)}
    weighted = {column[term]: (1.0 + np.log(count)) * idf[column[term]]
                for term, count in counts.items() if term in column}
    norm = float(np.sqrt(sum(value * value for value in weighted.values())))
    recomputed = np.asarray([weighted.get(int(j), 0.0) / norm if norm else 0.0
                             for j in active])
    if not np.allclose(recomputed, audited, rtol=0, atol=1e-10):
        raise ValueError("account unigram values differ from full raw audit")
    reference = LabelBlindReference.load(args.reference)
    if reference.version != manifest["reference_version"] or reference.names != tuple(names):
        raise ValueError("structured reference differs")
    evidence = reference.transform(base_values[index:index + 1],
                                   contexts[index:index + 1])
    structured = np.r_[np.log1p(evidence.upper[0]) * evidence.reliability[0],
                       np.log1p(evidence.lower[0]) * evidence.reliability[0]]
    structured_terms = []
    for j, name in enumerate(names):
        upper = float(structured[j] * coefficient[j])
        lower = float(structured[22 + j] * coefficient[22 + j])
        structured_terms.append({
            "fact_name": name,
            "source_value": float(base_values[index, j]) if
            np.isfinite(base_values[index, j]) else None,
            "upper_tail_score": float(evidence.upper[0, j]),
            "lower_tail_score": float(evidence.lower[0, j]),
            "reliability": float(evidence.reliability[0, j]),
            "upper_coefficient": float(coefficient[j]),
            "lower_coefficient": float(coefficient[22 + j]),
            "logit_contribution": upper + lower,
        })
    unigram_terms = [{
        "term": vocabulary[int(j)],
        "train_idf": float(idf[int(j)]),
        "source_occurrences": int(counts[vocabulary[int(j)]]),
        "normalized_tfidf": float(value),
        "coefficient": float(coefficient[44 + int(j)]),
        "logit_contribution": float(value * coefficient[44 + int(j)]),
    } for j, value in zip(active, recomputed)]
    structured_group = float(sum(term["logit_contribution"]
                                 for term in structured_terms))
    unigram_group = float(sum(term["logit_contribution"]
                              for term in unigram_terms))
    logit = intercept + structured_group + unigram_group
    probability = float(expit(logit))
    recorded = next((item for item in saved["predictions"]
                     if item["account_id"] == args.account_id), None)
    if (recorded is None or recorded["label"] != int(labels[index])
            or abs(probability - recorded["probability"]) > 1e-10
            or abs(structured_group - recorded["group_contributions"][
                "structured_evidence"]) > 1e-10
            or abs(unigram_group - recorded["group_contributions"][
                "unigram_evidence"]) > 1e-10):
        raise ValueError("source evidence does not reconstruct saved decision")
    top = sorted(unigram_terms, key=lambda term: (
        -abs(term["logit_contribution"]), term["term"]))[:args.source_top_k]
    locators = []
    for term in top:
        found = [{"tweet_index": k,
                  "occurrences": values.count(term["term"]),
                  "tweet_text_sha256": hashlib.sha256(
                      str(tweets[k]).encode("utf-8")).hexdigest()}
                 for k, values in enumerate(tokenized) if term["term"] in values]
        if sum(item["occurrences"] for item in found) != term["source_occurrences"]:
            raise ValueError("top-term tweet locators differ from source count")
        locators.append({"term": term["term"], "zip_member": "Twibot-20/test.json",
                         "account_array_position": int(positions[index]),
                         "tweet_locators": found})
    result = {
        "dataset": "twibot20", "account_id": args.account_id,
        "source_archive_sha256": manifest["source_archive_sha256"],
        "base_source_audit_sha256": manifest["base_source_audit_sha256"],
        "active_source_audit_sha256": manifest["active_source_audit_sha256"],
        "reference_version": reference.version,
        "intercept": intercept,
        "structured_terms": structured_terms,
        "unigram_terms": unigram_terms,
        "top_unigram_source_locators": locators,
        "structured_group_logit_contribution": structured_group,
        "unigram_group_logit_contribution": unigram_group,
        "reconstructed_logit": logit,
        "reconstructed_probability": probability,
        "saved_probability": recorded["probability"],
        "probability_without_unigram_group": float(expit(intercept +
                                                        structured_group)),
        "threshold": threshold,
        "predicted_label": int(probability >= threshold),
        "true_label": int(labels[index]),
        "status": "official_source_to_sparse_evidence_to_logit_verified",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"account_id": args.account_id,
                      "active_unigram_terms": len(unigram_terms),
                      "top_source_terms": len(locators),
                      "probability": probability,
                      "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
