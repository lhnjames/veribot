"""Verify one Fox8 test decision from native tweet targets to spline logit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from verifiable_bot.fox8_adapter import FEATURE_FIELDS, profile_features, tweet_features
from verifiable_bot.fox8_native_interaction_facts import FACT_NAMES
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def account_native_source(raw: Path, event_index: Path, account_id: str,
                          known: set[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    indexed = json.loads(event_index.read_text(encoding="utf-8"))
    selected = [item for item in indexed["events"] if account_id in
                (item["source_id"], item["target_id"])]
    recorded = {(item["source_id"], item["target_id"], item["relation"]):
                item["locators"] for item in selected}
    observed: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    all_out = Counter()
    all_out_locators: list[dict] = []
    published_values = None
    account_line = None
    seen = set()
    with gzip.open(raw, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            source = str(row["user_id"])
            if source in seen:
                continue
            seen.add(source)
            if source == account_id:
                tweets = row.get("user_tweets") or []
                profile = tweets[0]["user"] if tweets else {}
                fields = {**profile_features(profile), **tweet_features(tweets)}
                published_values = np.asarray(
                    [float(fields[name]) for name in FEATURE_FIELDS],
                    dtype=np.float32)
                account_line = line_number
            for position, tweet in enumerate(row.get("user_tweets") or []):
                reply = tweet.get("in_reply_to_user_id_str") or \
                    tweet.get("in_reply_to_user_id")
                retweet = tweet.get("retweeted_status")
                specs = []
                if reply is not None:
                    specs.append(("reply", str(reply),
                                  "in_reply_to_user_id_str|in_reply_to_user_id"))
                if isinstance(retweet, dict):
                    user = retweet.get("user") or {}
                    target = user.get("id_str") or user.get("id")
                    if target is not None:
                        specs.append(("retweet", str(target),
                                      "retweeted_status.user.id_str|id"))
                for kind, target, field in specs:
                    if source != account_id and target != account_id:
                        continue
                    locator = {"line": line_number, "tweet_index": position,
                               "field": field,
                               "tweet_id": str(tweet.get("id_str") or
                                               tweet.get("id") or position)}
                    if source == account_id:
                        all_out[kind] += 1
                        all_out_locators.append({"source_id": source,
                                                 "target_id": target,
                                                 "relation": kind, **locator})
                    if target in known and target != source:
                        observed[source, target, kind].append(locator)
    if set(observed) != set(recorded) or any(
            observed[key] != recorded[key] for key in recorded):
        raise ValueError("selected account event locators differ from raw source")
    if published_values is None:
        raise ValueError("selected account absent from raw source")
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    out_count = Counter()
    in_count = Counter()
    for (source, target, kind), locators in observed.items():
        if source == account_id:
            outgoing[kind].add(target)
            out_count[kind] += len(locators)
        if target == account_id:
            incoming[kind].add(source)
            in_count[kind] += len(locators)
    values = []
    for kind in ("reply", "retweet"):
        values += [np.log1p(out_count[kind]), np.log1p(in_count[kind]),
                   np.log1p(len(outgoing[kind])), np.log1p(len(incoming[kind]))]
    values += [np.log1p(len(outgoing[kind] & incoming[kind]))
               for kind in ("reply", "retweet")]
    values += [out_count[kind] / all_out[kind] if all_out[kind] else np.nan
               for kind in ("reply", "retweet")]
    return published_values, np.asarray(values, dtype=float), {
        "account_raw_line": account_line,
        "raw_events_touching_account": [
            {"source_id": source, "target_id": target, "relation": kind,
             "locators": locators}
            for (source, target, kind), locators in sorted(observed.items())],
        "all_outgoing_target_locators_including_external": all_out_locators,
        "outgoing_target_denominators": dict(all_out),
        "internal_outgoing_target_numerators": dict(out_count),
        "selected_event_index_exactly_matches_raw": True,
    }


def trace(args: argparse.Namespace) -> dict:
    directory = args.model_dir / f"seed{args.seed}"
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    saved = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = report["manifest"]
    audit = json.loads(args.native_source_audit.read_text(encoding="utf-8"))
    published_audit = json.loads(args.published_source_audit.read_text(
        encoding="utf-8"))
    event_index = args.native_facts.with_name("interaction_event_index.json")
    if (manifest["raw_source_sha256"] != sha256_file(args.raw)
            or manifest["native_facts_sha256"] != sha256_file(args.native_facts)
            or manifest["native_event_index_sha256"] != sha256_file(event_index)
            or manifest["native_source_audit_sha256"] !=
            sha256_file(args.native_source_audit)
            or audit["status"] !=
            "verified_all_native_interaction_fact_cells_and_event_locators"
            or not published_audit["all_rows_match"]
            or published_audit["files_sha256"][str(args.raw.resolve())] !=
            sha256_file(args.raw)
            or published_audit["files_sha256"][str(
                (args.data_dir / "features.csv").resolve())] !=
            sha256_file(args.data_dir / "features.csv")):
        raise ValueError("saved model or full-population source audits differ")
    x, y, ids, names, split, source = load("fox8", args.data_dir,
                                           None, args.seed)
    if source["split_sha256"] != manifest["split_sha256"]:
        raise ValueError("test split differs")
    with np.load(args.native_facts, allow_pickle=False) as cache:
        if (cache["ids"].astype(str).tolist() != ids
                or cache["names"].astype(str).tolist() != list(FACT_NAMES)):
            raise ValueError("native fact account order or vocabulary differs")
        native = cache["values"].astype(float)
    matches = [int(i) for i in split["test"] if ids[int(i)] == args.account_id]
    if len(matches) != 1:
        raise ValueError("account must occur once in the selected test split")
    i = matches[0]
    published, direct, sources = account_native_source(
        args.raw, event_index, args.account_id, set(ids))
    if not np.allclose(published, x[i], rtol=1e-5, atol=1e-5):
        raise ValueError("published facts differ from raw account and tweets")
    if not np.allclose(direct, native[i], rtol=0, atol=1e-12,
                       equal_nan=True):
        raise ValueError("native source values differ from cached fact row")
    raw = np.concatenate((published.astype(float), direct))[None, :]
    reference_path = directory / "reference.json.gz"
    if sha256_file(reference_path) != manifest["reference_artifact_sha256"]:
        raise ValueError("reference artifact differs")
    reference = LabelBlindReference.load(reference_path)
    reasoner = MonotoneSplineReasoner.from_manifest(report["reasoner"])
    if (reference.version != manifest["reference_version"]
            or reasoner.reference_version != reference.version
            or reference.names != tuple(names + list(FACT_NAMES))):
        raise ValueError("reference or reasoner vocabulary differs")
    evidence = reference.transform(raw)
    design = reasoner._design(evidence)
    terms = []
    for j, name in enumerate(reference.names):
        sl = reasoner._slices[name]
        contribution = float(design[0, sl] @ reasoner.coef[sl])
        terms.append({
            "fact_name": name,
            "source_value": float(raw[0, j]) if np.isfinite(raw[0, j]) else None,
            "source_level": "raw_tweet_target" if j >= len(names)
                            else "raw_account_or_tweet_field",
            "available": bool(evidence.available[0, j]),
            "upper_tail_score": float(evidence.upper[0, j]),
            "lower_tail_score": float(evidence.lower[0, j]),
            "reliability": float(evidence.reliability[0, j]),
            "selected_tail": reasoner.tail[j],
            "direction": int(reasoner.sign[j]),
            "nonnegative_spline_coefficients": reasoner.coef[sl].tolist(),
            "logit_contribution": contribution,
        })
    by_group = {name: float(value[0]) for name, value in
                reasoner.contributions(evidence).items()}
    saved_row = next((item for item in saved["predictions"]
                      if item["account_id"] == args.account_id), None)
    if saved_row is None or saved_row["label"] != int(y[i]):
        raise ValueError("saved test account or label differs")
    logit = float(reasoner.intercept + sum(
        item["logit_contribution"] for item in terms))
    probability = float(expit(logit))
    if (abs(logit - float(reasoner.logits(evidence)[0])) > 1e-12
            or abs(probability - saved_row["probability"]) > 1e-12
            or any(abs(by_group[name] - saved_row["group_contributions"][name])
                   > 1e-12 for name in by_group)):
        raise ValueError("evidence terms do not reconstruct saved decision")
    return {
        "dataset": "fox8", "seed": args.seed, "account_id": args.account_id,
        "raw_source_sha256": sha256_file(args.raw),
        "published_source_audit_sha256": sha256_file(args.published_source_audit),
        "native_source_audit_sha256": sha256_file(args.native_source_audit),
        "native_fact_names": list(FACT_NAMES),
        "published_fact_names": list(FEATURE_FIELDS),
        "published_source_values_recomputed_from_raw": published.astype(float).tolist(),
        "native_source_values_recomputed_from_raw": [
            float(v) if np.isfinite(v) else None for v in direct],
        "native_source_records": sources,
        "reference_version": reference.version,
        "intercept": reasoner.intercept,
        "fact_terms": terms,
        "group_logit_contributions": by_group,
        "reconstructed_logit": logit,
        "reconstructed_probability": probability,
        "saved_probability": saved_row["probability"],
        "threshold": manifest["threshold"],
        "predicted_label": int(probability >= manifest["threshold"]),
        "true_label": int(y[i]),
        "status": "raw_source_to_fact_to_reference_to_logit_verified",
        "scope": "source and fixed-model arithmetic; semantic validity requires independent external truth",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--native-facts", type=Path, required=True)
    parser.add_argument("--native-source-audit", type=Path, required=True)
    parser.add_argument("--published-source-audit", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = trace(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"account_id": result["account_id"],
                      "source_events": len(result["native_source_records"][
                          "raw_events_touching_account"]),
                      "fact_terms": len(result["fact_terms"]),
                      "probability": result["reconstructed_probability"],
                      "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
