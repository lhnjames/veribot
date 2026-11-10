"""Trace a selected official test decision from tweet tokens to saved logit."""

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
from train_v3_twibot20 import sha256
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.twibot20_token_facts import tokenize
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--base-source-audit", type=Path, required=True)
    parser.add_argument("--token-facts", type=Path, required=True)
    parser.add_argument("--token-source-audit", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--source-top-k", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.model_dir / f"seed{args.seed}"
    report = json.loads((directory / "evidence_report.json").read_text(encoding="utf-8"))
    saved = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = report["manifest"]
    base_audit = json.loads(args.base_source_audit.read_text(encoding="utf-8"))
    token_audit = json.loads(args.token_source_audit.read_text(encoding="utf-8"))
    if (manifest["source_archive_sha256"] != sha256(args.zip)
            or manifest["base_facts_sha256"] != sha256(args.base_facts)
            or manifest["token_facts_sha256"] != sha256(args.token_facts)
            or manifest["token_source_audit_sha256"] !=
            sha256(args.token_source_audit)
            or manifest["selection_lock_sha256"] != sha256(args.selection_lock)
            or base_audit["status"] !=
            "verified_all_labeled_account_fact_cells_against_official_zip"
            or token_audit["status"] !=
            "verified_all_token_fact_cells_and_train_only_vocabulary"):
        raise ValueError("official sources, audits, or selection lock differ")
    with np.load(args.base_facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        positions = cache["positions"].astype(int)
        parts = cache["split"].astype(str)
        contexts = cache["contexts"].astype(str)
        labels = cache["labels"].astype(int)
        base_names = cache["names"].astype(str).tolist()
        base_values = cache["values"].astype(float)
    matches = np.flatnonzero((ids == args.account_id) & (parts == "test"))
    if len(matches) != 1:
        raise ValueError("selected account must occur once in official test")
    i = int(matches[0])
    with np.load(args.token_facts, allow_pickle=False) as cache:
        if (cache["ids"].astype(str).tolist() != ids.tolist()
                or cache["positions"].astype(int).tolist() != positions.tolist()):
            raise ValueError("token and base account order differs")
        token_names = cache["names"].astype(str).tolist()
        token_values = cache["values"][i].astype(float)
    with zipfile.ZipFile(args.zip) as zipped:
        with zipped.open("Twibot-20/test.json") as stream:
            row = next(value for position, value in enumerate(
                ijson.items(stream, "item")) if position == positions[i])
    if str(row["ID"]) != args.account_id:
        raise ValueError("official ZIP account locator differs")
    tweets = row.get("tweet") or []
    token_sets = [tokenize(text) for text in tweets]
    recomputed = np.full(len(token_names), np.nan, dtype=np.float32)
    if tweets:
        recomputed.fill(0.0)
        for j, name in enumerate(token_names):
            token = name.split(":", 1)[1]
            recomputed[j] = sum(token in values for values in token_sets) / len(tweets)
    if not np.allclose(recomputed, token_values, rtol=0, atol=1e-7,
                       equal_nan=True):
        raise ValueError("token incidence differs from official tweet sample")
    reference_path = directory / "reference.json.gz"
    if sha256(reference_path) != manifest["reference_artifact_sha256"]:
        raise ValueError("reference artifact differs")
    reference = LabelBlindReference.load(reference_path)
    reasoner = DualTailEvidenceReasoner.from_manifest(report["reasoner"])
    names = base_names + token_names
    if (tuple(names) != reference.names
            or reference.version != reasoner.reference_version):
        raise ValueError("reference or reasoner vocabulary differs")
    raw = np.concatenate((base_values[i], recomputed.astype(float)))[None, :]
    evidence = reference.transform(raw, contexts[i:i + 1])
    design = reasoner._design(evidence)
    width = len(names)
    terms = []
    for j, name in enumerate(names):
        high = float(design[0, j] * reasoner.coef[j])
        low = float(design[0, width + j] * reasoner.coef[width + j])
        terms.append({
            "fact_name": name,
            "source_value": float(raw[0, j]) if np.isfinite(raw[0, j]) else None,
            "available": bool(evidence.available[0, j]),
            "upper_tail_score": float(evidence.upper[0, j]),
            "lower_tail_score": float(evidence.lower[0, j]),
            "reliability": float(evidence.reliability[0, j]),
            "upper_direction": int(reasoner.tail_sign[j, 0]),
            "lower_direction": int(reasoner.tail_sign[j, 1]),
            "upper_nonnegative_coefficient": float(reasoner.coef[j]),
            "lower_nonnegative_coefficient": float(reasoner.coef[width + j]),
            "total_logit_contribution": high + low,
        })
    probability = float(reasoner.predict_proba(evidence)[0])
    groups = {name: float(values[0]) for name, values in
              reasoner.contributions(evidence).items()}
    saved_row = next((item for item in saved["predictions"]
                      if item["account_id"] == args.account_id), None)
    logit = float(reasoner.intercept + sum(
        term["total_logit_contribution"] for term in terms))
    if (saved_row is None or saved_row["label"] != int(labels[i])
            or abs(probability - saved_row["probability"]) > 1e-12
            or abs(logit - float(reasoner.logits(evidence)[0])) > 1e-12
            or any(abs(groups[name] - saved_row["group_contributions"][name])
                   > 1e-12 for name in groups)):
        raise ValueError("source-backed evidence does not reconstruct saved prediction")
    ranked = sorted(range(len(base_names), width), key=lambda j: (
        -abs(terms[j]["total_logit_contribution"]), names[j]))[:args.source_top_k]
    sources = []
    for j in ranked:
        token = names[j].split(":", 1)[1]
        locators = [{"tweet_index": k,
                     "tweet_text_sha256": hashlib.sha256(
                         str(tweets[k]).encode("utf-8")).hexdigest()}
                    for k, values in enumerate(token_sets) if token in values]
        if tweets and not np.isclose(len(locators) / len(tweets),
                                     float(recomputed[j - len(base_names)]),
                                     rtol=0, atol=1e-7):
            raise ValueError("source locators do not match token incidence")
        sources.append({"fact_name": names[j],
                        "zip_member": "Twibot-20/test.json",
                        "account_array_position": int(positions[i]),
                        "account_id": args.account_id,
                        "sampled_tweet_count": len(tweets),
                        "tweet_locators": locators})
    result = {
        "dataset": "twibot20", "seed": args.seed,
        "account_id": args.account_id,
        "source_archive_sha256": manifest["source_archive_sha256"],
        "base_source_audit_sha256": sha256(args.base_source_audit),
        "token_source_audit_sha256": sha256(args.token_source_audit),
        "reference_version": reference.version,
        "token_facts_recomputed_from_official_tweets": True,
        "intercept": reasoner.intercept,
        "fact_terms": terms,
        "top_token_source_locators": sources,
        "group_logit_contributions": groups,
        "reconstructed_logit": logit,
        "reconstructed_probability": probability,
        "saved_probability": saved_row["probability"],
        "threshold": manifest["threshold"],
        "predicted_label": int(probability >= manifest["threshold"]),
        "true_label": int(labels[i]),
        "status": "official_source_to_token_evidence_to_logit_verified",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"account_id": args.account_id,
                      "fact_terms": len(terms),
                      "top_token_sources": len(sources),
                      "probability": probability,
                      "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
