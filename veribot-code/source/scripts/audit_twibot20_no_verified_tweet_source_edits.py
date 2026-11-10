"""Replay frozen no-verified decisions after controlled official tweet-list edits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import ijson
import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256
from verifiable_bot.twibot20_facts import account_facts
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


TWEET_FIELDS = ("tweet_sample_count", "retweet_fraction", "url_fraction",
                "hashtag_fraction", "mention_fraction", "mean_tweet_length",
                "unique_tweet_fraction")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = json.loads(args.control.read_text(encoding="utf-8"))
    for key, path in (("source_archive_sha256", args.zip),
                      ("base_facts_sha256", args.facts),
                      ("frozen_model_artifact_sha256", args.artifact),
                      ("reference_artifact_sha256", args.reference)):
        if sha256(path) != control[key]:
            raise ValueError(f"Official source or frozen artifact differs: {key}")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        split = saved["split"].astype(str)
        positions = saved["positions"].astype(int)
        labels = saved["labels"].astype(int)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        names = saved["names"].astype(str).tolist()
    test = np.flatnonzero(split == "test")
    rows = control["rows"]
    if [row["account_id"] for row in rows] != ids[test].tolist():
        raise ValueError("Frozen test population differs")
    original_facts = values[test].copy()
    first_facts = original_facts.copy()
    empty_facts = original_facts.copy()
    original_documents = []
    first_documents = []
    empty_documents = []
    source_rows = []
    with zipfile.ZipFile(args.zip) as zipped, zipped.open("Twibot-20/test.json") as stream:
        for position, row in enumerate(ijson.items(stream, "item")):
            if position >= len(test):
                raise ValueError("Official test member has more records than cache")
            i = int(test[position])
            if ids[i] != str(row["ID"]) or positions[i] != position:
                raise ValueError("Official test account order differs")
            extracted, _, context = account_facts(row)
            if context != contexts[i] or not np.allclose(extracted, values[i, :len(extracted)],
                                                          rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError(f"Original source facts differ: {ids[i]}")
            tweets = [str(tweet or "") for tweet in row.get("tweet") or []]
            first = tweets[1:] if tweets else tweets
            first_row = dict(row, tweet=first)
            empty_row = dict(row, tweet=[])
            first_value, _, _ = account_facts(first_row)
            empty_value, _, _ = account_facts(empty_row)
            first_facts[position, :len(first_value)] = first_value
            empty_facts[position, :len(empty_value)] = empty_value
            changed = {names[k] for k in range(len(extracted))
                       if not np.isclose(first_value[k], extracted[k], rtol=0, atol=1e-12,
                                         equal_nan=True)}
            changed_empty = {names[k] for k in range(len(extracted))
                             if not np.isclose(empty_value[k], extracted[k], rtol=0, atol=1e-12,
                                               equal_nan=True)}
            if not changed.issubset(TWEET_FIELDS) or not changed_empty.issubset(TWEET_FIELDS):
                raise ValueError("Tweet edit altered a non-tweet source fact")
            original_documents.append(" ".join(tweets))
            first_documents.append(" ".join(first))
            empty_documents.append("")
            source_rows.append({
                "account_id": str(ids[i]),
                "tweet_list_locator": f"Twibot-20/test.json[{position}].tweet",
                "first_tweet_locator": f"Twibot-20/test.json[{position}].tweet[0]" if tweets else None,
                "first_tweet_sha256": hashlib.sha256(tweets[0].encode("utf-8")).hexdigest() if tweets else None,
                "original_tweet_count": len(tweets),
                "changed_fact_names_first": sorted(changed),
                "changed_fact_names_empty": sorted(changed_empty),
            })
    if len(source_rows) != len(test):
        raise ValueError("Official test member has fewer records than cache")
    reference = LabelBlindReference.load(args.reference)
    if reference.names != tuple(names):
        raise ValueError("Reference fact names differ")
    with np.load(args.artifact, allow_pickle=False) as saved:
        vocabulary = saved["vocabulary"].astype(str)
        idf = saved["idf"].astype(float)
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                 stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
                                 ngram_range=(1, 1), sublinear_tf=True, dtype=np.float64)
    vectorizer.vocabulary_ = {term: j for j, term in enumerate(vocabulary)}
    vectorizer.fixed_vocabulary_ = True
    vectorizer.idf_ = idf
    active = control["without_verified"]["active_unigram_coefficients"]
    text_coeff = np.zeros(len(vocabulary), dtype=float)
    word_to_index = {term: j for j, term in enumerate(vocabulary)}
    for term, coefficient in active.items():
        text_coeff[word_to_index[term]] = coefficient
    structural = control["without_verified"]["structured_coefficients_by_fact"]
    structured_coeff = np.asarray(
        [structural[name]["upper"] for name in names if name != "verified"]
        + [structural[name]["lower"] for name in names if name != "verified"], dtype=float)
    keep = [j for j, name in enumerate(names) if name != "verified"]
    keep += [j + len(names) for j, name in enumerate(names) if name != "verified"]
    intercept = float(control["without_verified"]["intercept"])
    threshold = float(control["without_verified"]["threshold"])
    test_contexts = contexts[test]

    def score(facts: np.ndarray, documents: list[str]):
        evidence = reference.transform(facts, contexts=test_contexts)
        encoded = np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                                   np.log1p(evidence.lower) * evidence.reliability))[:, keep]
        structured_logit = encoded @ structured_coeff
        lexical_logit = np.asarray(vectorizer.transform(documents) @ text_coeff).ravel()
        return intercept + structured_logit + lexical_logit, structured_logit, lexical_logit, encoded

    original_logit, original_structured, original_lexical, original_encoded = score(
        original_facts, original_documents)
    first_logit, first_structured, first_lexical, first_encoded = score(first_facts, first_documents)
    empty_logit, empty_structured, empty_lexical, empty_encoded = score(empty_facts, empty_documents)
    saved_logit = np.asarray([row["without_verified_logit"] for row in rows], dtype=float)
    replay_error = float(np.max(np.abs(original_logit - saved_logit)))
    if replay_error > 1e-10:
        raise ValueError("Original model does not replay frozen no-verified score")
    tweet_keep_cols = [keep.index(names.index(name)) for name in TWEET_FIELDS]
    tweet_keep_cols += [keep.index(names.index(name) + len(names)) for name in TWEET_FIELDS]
    other_cols = [j for j in range(len(keep)) if j not in tweet_keep_cols]
    if (not np.allclose(original_encoded[:, other_cols], first_encoded[:, other_cols], rtol=0, atol=0)
            or not np.allclose(original_encoded[:, other_cols], empty_encoded[:, other_cols], rtol=0, atol=0)):
        raise ValueError("Tweet edit changed an unrelated encoded structured coordinate")
    first_delta_structured = first_structured - original_structured
    empty_delta_structured = empty_structured - original_structured
    first_delta_lexical = first_lexical - original_lexical
    empty_delta_lexical = empty_lexical - original_lexical
    first_field_deltas = {}
    for field in TWEET_FIELDS:
        upper_col = keep.index(names.index(field))
        lower_col = keep.index(names.index(field) + len(names))
        first_field_deltas[field] = (
            (first_encoded[:, upper_col] - original_encoded[:, upper_col])
            * structured_coeff[upper_col]
            + (first_encoded[:, lower_col] - original_encoded[:, lower_col])
            * structured_coeff[lower_col])
    first_field_error = float(np.max(np.abs(
        sum(first_field_deltas.values()) - first_delta_structured)))
    if first_field_error > 1e-10:
        raise ValueError("First-tweet structured score change does not decompose by fact")
    delta_error = float(max(np.max(np.abs((first_logit - original_logit)
                                        - (first_delta_structured + first_delta_lexical))),
                            np.max(np.abs((empty_logit - original_logit)
                                        - (empty_delta_structured + empty_delta_lexical)))))
    if delta_error > 1e-10:
        raise ValueError("Source edit score response does not match affected inputs")
    original_class = expit(original_logit) >= threshold
    first_class = expit(first_logit) >= threshold
    empty_class = expit(empty_logit) >= threshold
    zero_mask_logit = saved_logit - np.asarray([
        row["without_verified_group_effects"]["sampled_tweet_stats"]
        + row["without_verified_group_effects"]["tweet_unigrams"] for row in rows])
    zero_mask_class = expit(zero_mask_logit) >= threshold
    eligible = np.asarray([row["original_tweet_count"] > 0 for row in source_rows], dtype=bool)
    count200 = np.asarray([row["original_tweet_count"] == 200 for row in source_rows], dtype=bool)
    first_changes = first_class != original_class
    empty_zero_score_difference = np.abs(empty_logit - zero_mask_logit)
    first_change_index = int(np.flatnonzero(first_changes)[0])
    count_index = names.index("tweet_sample_count")
    context_case = test_contexts[[first_change_index]]
    original_tail = reference.transform(original_facts[[first_change_index]], contexts=context_case)
    edited_tail = reference.transform(first_facts[[first_change_index]], contexts=context_case)
    first_change_case = {
        "account_id": source_rows[first_change_index]["account_id"],
        "first_tweet_locator": source_rows[first_change_index]["first_tweet_locator"],
        "original_tweet_count": float(original_facts[first_change_index, count_index]),
        "edited_tweet_count": float(first_facts[first_change_index, count_index]),
        "original_count_upper_tail": float(original_tail.upper[0, count_index]),
        "original_count_lower_tail": float(original_tail.lower[0, count_index]),
        "edited_count_upper_tail": float(edited_tail.upper[0, count_index]),
        "edited_count_lower_tail": float(edited_tail.lower[0, count_index]),
        "count_reliability": float(original_tail.reliability[0, count_index]),
        "count_upper_coefficient": float(structural["tweet_sample_count"]["upper"]),
        "count_lower_coefficient": float(structural["tweet_sample_count"]["lower"]),
        "count_logit_delta": float(first_field_deltas["tweet_sample_count"][first_change_index]),
        "original_logit": float(original_logit[first_change_index]),
        "edited_logit": float(first_logit[first_change_index]),
        "label_for_evaluation": int(labels[test[first_change_index]]),
    }
    result = {
        "scope": "TwiBot-20 frozen no-verified model; controlled in-memory official test tweet-list edits",
        "interpretation": "source-record sensitivity; no evidence that source tweets were erroneous",
        "source_archive_sha256": control["source_archive_sha256"],
        "base_facts_sha256": control["base_facts_sha256"],
        "reference_artifact_sha256": control["reference_artifact_sha256"],
        "control_certificate_sha256": sha256(args.control),
        "frozen_original_logit_replay_max_error": replay_error,
        "source_edit_score_delta_reconstruction_max_error": delta_error,
        "first_tweet_field_delta_reconstruction_max_error": first_field_error,
        "test_accounts": len(test),
        "nonempty_tweet_lists": int(np.count_nonzero(eligible)),
        "full_200_tweet_lists": int(np.count_nonzero(count200)),
        "official_train_accounts": int(np.count_nonzero(split == "train")),
        "official_train_200_tweet_lists": int(np.count_nonzero(
            (split == "train") & (values[:, count_index] == 200))),
        "original_errors": int(np.count_nonzero(original_class != labels[test])),
        "first_class_change_case": first_change_case,
        "remove_first_tweet": {
            "changed_score_accounts": int(np.count_nonzero(np.abs(first_logit - original_logit) > 1e-12)),
            "changed_class_accounts": int(np.count_nonzero(first_class != original_class)),
            "class_changes_from_200_tweet_lists": int(np.count_nonzero(first_changes & count200)),
            "class_changes_from_other_nonempty_lists": int(np.count_nonzero(first_changes & eligible & ~count200)),
            "errors": int(np.count_nonzero(first_class != labels[test])),
            "largest_absolute_logit_change": float(np.max(np.abs(first_logit - original_logit))),
            "field_delta_summary": {
                name: {"nonzero_score_response_accounts": int(np.count_nonzero(np.abs(effect) > 1e-12)),
                       "mean_absolute_logit_response": float(np.mean(np.abs(effect[eligible]))),
                       "largest_absolute_logit_response": float(np.max(np.abs(effect)))}
                for name, effect in first_field_deltas.items()},
        },
        "empty_all_sampled_tweets": {
            "changed_score_accounts": int(np.count_nonzero(np.abs(empty_logit - original_logit) > 1e-12)),
            "changed_class_accounts": int(np.count_nonzero(empty_class != original_class)),
            "errors": int(np.count_nonzero(empty_class != labels[test])),
            "largest_absolute_logit_change": float(np.max(np.abs(empty_logit - original_logit))),
        },
        "zero_mask_tweet_stats_and_unigrams": {
            "changed_class_accounts": int(np.count_nonzero(zero_mask_class != original_class)),
            "errors": int(np.count_nonzero(zero_mask_class != labels[test])),
            "class_disagreement_with_empty_source": int(np.count_nonzero(zero_mask_class != empty_class)),
            "max_absolute_logit_disagreement_with_empty_source": float(np.max(empty_zero_score_difference)),
            "nonzero_logit_disagreement_with_empty_source": int(np.count_nonzero(
                empty_zero_score_difference > 1e-12)),
        },
        "rows": [{**source_rows[k],
                  "label_for_evaluation": int(labels[index]),
                  "original_logit": float(original_logit[k]),
                  "first_removed_logit": float(first_logit[k]),
                  "all_tweets_empty_logit": float(empty_logit[k]),
                  "zero_mask_tweet_groups_logit": float(zero_mask_logit[k]),
                  "first_removed_structured_delta": float(first_delta_structured[k]),
                  "first_removed_unigram_delta": float(first_delta_lexical[k]),
                  "first_removed_structured_field_deltas": {
                      name: float(effect[k]) for name, effect in first_field_deltas.items()},
                  "all_tweets_empty_structured_delta": float(empty_delta_structured[k]),
                  "all_tweets_empty_unigram_delta": float(empty_delta_lexical[k]),
                  "original_class": int(original_class[k]),
                  "first_removed_class": int(first_class[k]),
                  "all_tweets_empty_class": int(empty_class[k]),
                  "zero_mask_tweet_groups_class": int(zero_mask_class[k])}
                 for k, index in enumerate(test)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
