"""Same-protocol control removing verified and tweet_sample_count, with source edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import ijson
import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import metric, sha256, threshold_on_dev
from verifiable_bot.twibot20_facts import account_facts
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--source-edit-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = json.loads(args.control.read_text(encoding="utf-8"))
    prior_edit = json.loads(args.source_edit_control.read_text(encoding="utf-8"))
    if prior_edit["control_certificate_sha256"] != sha256(args.control):
        raise ValueError("Prior source edit and model control differ")
    for key, path in (("source_archive_sha256", args.zip),
                      ("base_facts_sha256", args.facts),
                      ("frozen_model_artifact_sha256", args.artifact),
                      ("reference_artifact_sha256", args.reference)):
        if sha256(path) != control[key]:
            raise ValueError(f"Frozen source or artifact differs: {key}")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        split = saved["split"].astype(str)
        positions = saved["positions"].astype(int)
        labels = saved["labels"].astype(int)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        names = saved["names"].astype(str).tolist()
    train, dev, test = (np.flatnonzero(split == name) for name in ("train", "dev", "test"))
    documents = []
    edited_test_documents = []
    edited_test_facts = values[test].copy()
    edit_source_rows = []
    with zipfile.ZipFile(args.zip) as zipped:
        for part in ("train", "dev", "test"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    i = len(documents)
                    if ids[i] != str(row["ID"]) or split[i] != part or positions[i] != position:
                        raise ValueError("Official source order differs")
                    tweets = [str(tweet or "") for tweet in row.get("tweet") or []]
                    documents.append(" ".join(tweets))
                    if part == "test":
                        k = position
                        original_facts, _, context = account_facts(row)
                        if context != contexts[i] or not np.allclose(
                                original_facts, values[i, :len(original_facts)],
                                rtol=0, atol=1e-12, equal_nan=True):
                            raise ValueError(f"Official source facts differ: {ids[i]}")
                        edited_tweets = tweets[1:] if tweets else tweets
                        edited_facts, _, _ = account_facts(dict(row, tweet=edited_tweets))
                        edited_test_facts[k, :len(edited_facts)] = edited_facts
                        edited_test_documents.append(" ".join(edited_tweets))
                        edit_source_rows.append({
                            "account_id": str(ids[i]),
                            "first_tweet_locator": f"Twibot-20/test.json[{k}].tweet[0]" if tweets else None,
                            "original_tweet_count": len(tweets),
                        })
    if len(documents) != len(ids) or len(edit_source_rows) != len(test):
        raise ValueError("Official source population differs")
    if [row["account_id"] for row in edit_source_rows] != [
            row["account_id"] for row in prior_edit["rows"]]:
        raise ValueError("Source edit account order differs")
    reference = LabelBlindReference.load(args.reference)
    if reference.names != tuple(names):
        raise ValueError("Frozen reference fact names differ")
    evidence = reference.transform(values, contexts=contexts)
    all_structured = np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                                      np.log1p(evidence.lower) * evidence.reliability))
    with np.load(args.artifact, allow_pickle=False) as frozen:
        vocabulary = frozen["vocabulary"].astype(str)
        idf = frozen["idf"].astype(float)
        original_coefficient = frozen["coefficient"].astype(float)
        original_intercept = float(frozen["intercept"][0])
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                 stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
                                 ngram_range=(1, 1), sublinear_tf=True, dtype=np.float64)
    vectorizer.vocabulary_ = {term: j for j, term in enumerate(vocabulary)}
    vectorizer.fixed_vocabulary_ = True
    vectorizer.idf_ = idf
    train_text = vectorizer.transform([documents[i] for i in train])
    dev_text = vectorizer.transform([documents[i] for i in dev])
    test_text = vectorizer.transform([documents[i] for i in test])
    edited_text = vectorizer.transform(edited_test_documents)
    full_replay = expit(original_intercept + all_structured[test] @ original_coefficient[:44]
                        + np.asarray(test_text @ original_coefficient[44:]).ravel())
    original_saved = json.loads((Path(args.artifact).parent / "selected" / "test_predictions.json")
                                .read_text(encoding="utf-8"))["predictions"]
    if [row["account_id"] for row in original_saved] != ids[test].tolist():
        raise ValueError("Selected full-model test account order differs")
    replay_error = float(np.max(np.abs(full_replay - np.asarray(
        [row["probability"] for row in original_saved]))))
    if replay_error > 1e-12:
        raise ValueError("Train-only text representation does not replay selected model")
    omitted = {"verified", "tweet_sample_count"}
    keep = [j for j, name in enumerate(names) if name not in omitted]
    keep += [j + len(names) for j, name in enumerate(names) if name not in omitted]
    train_x = hstack((csr_matrix(all_structured[train][:, keep]), train_text), format="csr")
    dev_x = hstack((csr_matrix(all_structured[dev][:, keep]), dev_text), format="csr")
    test_x = hstack((csr_matrix(all_structured[test][:, keep]), test_text), format="csr")
    model = LogisticRegression(C=1.0, penalty="l1", solver="liblinear",
                               max_iter=1000, random_state=0)
    model.fit(train_x, labels[train])
    dev_p = model.predict_proba(dev_x)[:, 1]
    threshold = threshold_on_dev(dev_p, labels[dev])
    coefficient = model.coef_.ravel()
    intercept = float(model.intercept_[0])
    original_logit = np.asarray(intercept + test_x @ coefficient).ravel()
    original_p = expit(original_logit)
    if np.max(np.abs(original_p - model.predict_proba(test_x)[:, 1])) > 1e-12:
        raise ValueError("Diagnostic model direct probabilities differ")
    edited_evidence = reference.transform(edited_test_facts, contexts=contexts[test])
    edited_structured = np.column_stack((
        np.log1p(edited_evidence.upper) * edited_evidence.reliability,
        np.log1p(edited_evidence.lower) * edited_evidence.reliability))[:, keep]
    original_structured = all_structured[test][:, keep]
    non_tweet_names = {"followers_count", "friends_count", "statuses_count", "listed_count",
                       "favourites_count", "protected", "default_profile", "default_profile_image",
                       "description_length", "sampled_following_count", "sampled_follower_count",
                       "graph_out_degree", "graph_in_degree", "graph_reciprocal_degree"}
    non_tweet_cols = [k for k, original_col in enumerate(keep)
                      if names[original_col % len(names)] in non_tweet_names]
    if not np.allclose(original_structured[:, non_tweet_cols],
                       edited_structured[:, non_tweet_cols], rtol=0, atol=0):
        raise ValueError("First-tweet edit changed an unrelated encoded fact")
    original_lexical = np.asarray(test_text @ coefficient[40:]).ravel()
    edited_lexical = np.asarray(edited_text @ coefficient[40:]).ravel()
    original_structured_logit = original_structured @ coefficient[:40]
    edited_structured_logit = edited_structured @ coefficient[:40]
    edited_logit = intercept + edited_structured_logit + edited_lexical
    delta_error = float(np.max(np.abs((edited_logit - original_logit)
                                      - ((edited_structured_logit - original_structured_logit)
                                         + (edited_lexical - original_lexical)))))
    if delta_error > 1e-10:
        raise ValueError("Source edit score does not reconstruct")
    original_class = original_p >= threshold
    edited_class = expit(edited_logit) >= threshold
    prior_rows = prior_edit["rows"]
    prior_class = np.asarray([row["original_class"] for row in prior_rows], dtype=int)
    prior_edited_class = np.asarray([row["first_removed_class"] for row in prior_rows], dtype=int)
    result = {
        "scope": "post hoc same-protocol TwiBot-20 diagnostic; official test previously explored",
        "operation": "retrain without verified and tweet_sample_count, then remove first sampled tweet in memory",
        "source_archive_sha256": control["source_archive_sha256"],
        "base_facts_sha256": control["base_facts_sha256"],
        "reference_artifact_sha256": control["reference_artifact_sha256"],
        "control_certificate_sha256": sha256(args.control),
        "prior_source_edit_certificate_sha256": sha256(args.source_edit_control),
        "original_selected_model_probability_replay_max_error": replay_error,
        "source_edit_score_delta_reconstruction_max_error": delta_error,
        "retained_structured_facts": [name for name in names if name not in omitted],
        "active_structured_terms": int(np.count_nonzero(coefficient[:40])),
        "active_unigram_terms": int(np.count_nonzero(coefficient[40:])),
        "dev": metric(labels[dev], dev_p, threshold),
        "test": metric(labels[test], original_p, threshold),
        "threshold": threshold,
        "source_edit": {
            "eligible_nonempty_lists": int(np.count_nonzero(
                [row["original_tweet_count"] > 0 for row in edit_source_rows])),
            "changed_score_accounts": int(np.count_nonzero(np.abs(edited_logit - original_logit) > 1e-12)),
            "changed_class_accounts": int(np.count_nonzero(edited_class != original_class)),
            "edited_errors": int(np.count_nonzero(edited_class != labels[test])),
            "correct_to_wrong": int(np.count_nonzero((original_class == labels[test])
                                                       & (edited_class != labels[test]))),
            "wrong_to_correct": int(np.count_nonzero((original_class != labels[test])
                                                       & (edited_class == labels[test]))),
        },
        "paired_with_no_verified_count_retained": {
            "prior_test_errors": control["without_verified"]["test"]["errors"],
            "prior_first_tweet_class_changes": prior_edit["remove_first_tweet"]["changed_class_accounts"],
            "prior_first_tweet_edited_errors": prior_edit["remove_first_tweet"]["errors"],
            "only_prior_model_correct": int(np.count_nonzero(
                (prior_class == labels[test]) & (original_class != labels[test]))),
            "only_new_model_correct": int(np.count_nonzero(
                (prior_class != labels[test]) & (original_class == labels[test]))),
            "prior_only_edit_flips": int(np.count_nonzero(
                (prior_edited_class != prior_class) & (edited_class == original_class))),
            "new_only_edit_flips": int(np.count_nonzero(
                (prior_edited_class == prior_class) & (edited_class != original_class))),
        },
        "rows": [{**edit_source_rows[k], "label_for_evaluation": int(labels[i]),
                  "original_logit": float(original_logit[k]),
                  "first_tweet_removed_logit": float(edited_logit[k]),
                  "structured_delta": float(edited_structured_logit[k]
                                            - original_structured_logit[k]),
                  "unigram_delta": float(edited_lexical[k] - original_lexical[k]),
                  "original_class": int(original_class[k]),
                  "first_tweet_removed_class": int(edited_class[k])}
                 for k, i in enumerate(test)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
