"""Same-protocol TwiBot-20 dev/test controls with verified omitted or isolated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import ijson
import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import metric, sha256, threshold_on_dev
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
    original = json.loads((args.selected_dir / "report.json").read_text(encoding="utf-8"))
    manifest = original["manifest"]
    for name, path in (("source_archive_sha256", args.zip),
                       ("base_facts_sha256", args.facts),
                       ("model_artifact_sha256", args.artifact),
                       ("reference_artifact_sha256", args.reference)):
        if sha256(path) != manifest[name]:
            raise ValueError(f"Frozen source or artifact mismatch: {name}")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        split = saved["split"].astype(str)
        positions = saved["positions"].astype(int)
        labels = saved["labels"].astype(int)
        facts = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        names = saved["names"].astype(str).tolist()
    train, dev, test = (np.flatnonzero(split == name) for name in ("train", "dev", "test"))
    documents = []
    with zipfile.ZipFile(args.zip) as zipped:
        for part in ("train", "dev", "test"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    i = len(documents)
                    if ids[i] != str(row["ID"]) or split[i] != part or positions[i] != position:
                        raise ValueError("Official source order differs")
                    documents.append(" ".join(str(t or "") for t in row.get("tweet") or []))
    if len(documents) != len(ids):
        raise ValueError("Official source population differs")
    reference = LabelBlindReference.load(args.reference)
    if reference.version != manifest["reference_version"] or reference.names != tuple(names):
        raise ValueError("Frozen label-blind reference differs")
    evidence = reference.transform(facts, contexts=contexts)
    structured = np.column_stack((np.log1p(evidence.upper) * evidence.reliability,
                                  np.log1p(evidence.lower) * evidence.reliability))
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                 stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
                                 ngram_range=(1, 1), min_df=10, max_features=20000,
                                 sublinear_tf=True, dtype=np.float64)
    train_text = vectorizer.fit_transform([documents[i] for i in train])
    dev_text = vectorizer.transform([documents[i] for i in dev])
    test_text = vectorizer.transform([documents[i] for i in test])
    with np.load(args.artifact, allow_pickle=False) as frozen:
        vocabulary = frozen["vocabulary"].astype(str)
        idf = frozen["idf"].astype(float)
        coefficient = frozen["coefficient"].astype(float)
        intercept = float(frozen["intercept"][0])
        original_threshold = float(frozen["threshold"][0])
    if not np.array_equal(vectorizer.get_feature_names_out(), vocabulary) or not np.array_equal(vectorizer.idf_, idf):
        raise ValueError("Frozen train-only text representation differs")
    original_test_input = hstack((csr_matrix(structured[test]), test_text), format="csr")
    frozen_p = expit(intercept + original_test_input @ coefficient)
    predictions = json.loads((args.selected_dir / "test_predictions.json").read_text(
        encoding="utf-8"))["predictions"]
    if [p["account_id"] for p in predictions] != ids[test].tolist():
        raise ValueError("Frozen test population differs")
    replay_error = float(np.max(np.abs(frozen_p - np.asarray([p["probability"] for p in predictions]))))
    if replay_error > 1e-12:
        raise ValueError("Frozen full-model decision does not replay")
    j = names.index("verified")
    verified_cols = [j, j + len(names)]
    remaining_cols = [k for k in range(structured.shape[1]) if k not in verified_cols]
    no_verified_train = hstack((csr_matrix(structured[train][:, remaining_cols]), train_text), format="csr")
    no_verified_dev = hstack((csr_matrix(structured[dev][:, remaining_cols]), dev_text), format="csr")
    no_verified_test = hstack((csr_matrix(structured[test][:, remaining_cols]), test_text), format="csr")
    single_train = structured[train][:, verified_cols]
    single_dev = structured[dev][:, verified_cols]
    single_test = structured[test][:, verified_cols]

    def fit(train_x, dev_x, test_x):
        model = LogisticRegression(C=1.0, penalty="l1", solver="liblinear",
                                   max_iter=1000, random_state=0)
        model.fit(train_x, labels[train])
        dev_p = model.predict_proba(dev_x)[:, 1]
        threshold = threshold_on_dev(dev_p, labels[dev])
        test_p = model.predict_proba(test_x)[:, 1]
        return model, threshold, dev_p, test_p

    no_verified_model, no_verified_threshold, no_verified_dev_p, no_verified_test_p = fit(
        no_verified_train, no_verified_dev, no_verified_test)
    single_model, single_threshold, single_dev_p, single_test_p = fit(
        single_train, single_dev, single_test)
    single_field_controls = []
    for name_index, name in enumerate(names):
        if name == "verified":
            continue
        cols = [name_index, name_index + len(names)]
        _, field_threshold, field_dev_p, field_test_p = fit(
            structured[train][:, cols], structured[dev][:, cols], structured[test][:, cols])
        single_field_controls.append({
            "field": name,
            "threshold": field_threshold,
            "dev": metric(labels[dev], field_dev_p, field_threshold),
            "test": metric(labels[test], field_test_p, field_threshold),
        })
    best_single = max(single_field_controls,
                      key=lambda row: (row["dev"]["macro_f1"], -names.index(row["field"])))
    frozen_class = frozen_p >= original_threshold
    no_verified_class = no_verified_test_p >= no_verified_threshold
    single_class = single_test_p >= single_threshold
    groups = {}
    for source_value in (0, 1):
        for label in (0, 1):
            indices = np.flatnonzero((facts[test, j] == source_value) & (labels[test] == label))
            groups[f"verified={source_value},label={label}"] = {
                "n": len(indices),
                "original_correct": int(np.count_nonzero(frozen_class[indices] == labels[test][indices])),
                "without_verified_correct": int(np.count_nonzero(no_verified_class[indices] == labels[test][indices])),
                "verified_only_correct": int(np.count_nonzero(single_class[indices] == labels[test][indices])),
            }
    no_verified_coefficient = no_verified_model.coef_.ravel()
    source_groups = {
        "profile_counts": {"followers_count", "friends_count", "statuses_count",
                           "listed_count", "favourites_count"},
        "profile_presentation": {"protected", "default_profile", "default_profile_image",
                                 "description_length"},
        "sampled_tweet_stats": {"tweet_sample_count", "retweet_fraction", "url_fraction",
                                "hashtag_fraction", "mention_fraction", "mean_tweet_length",
                                "unique_tweet_fraction"},
        "native_follow_graph": {"sampled_following_count", "sampled_follower_count",
                                "graph_out_degree", "graph_in_degree", "graph_reciprocal_degree"},
    }
    all_group_names = set().union(*source_groups.values())
    if all_group_names != set(names) - {"verified"} or sum(map(len, source_groups.values())) != 21:
        raise ValueError("Source groups do not partition remaining structured facts")
    group_effects = {}
    for group_name, group_names in source_groups.items():
        group_cols = [k for k, original_col in enumerate(remaining_cols)
                      if names[original_col % len(names)] in group_names]
        group_effects[group_name] = np.asarray(
            structured[test][:, np.asarray(remaining_cols)[group_cols]]
            @ no_verified_coefficient[group_cols]).ravel()
    group_effects["tweet_unigrams"] = np.asarray(test_text @ no_verified_coefficient[42:]).ravel()
    no_verified_logit = np.asarray(float(no_verified_model.intercept_[0])
                                   + no_verified_test @ no_verified_coefficient).ravel()
    field_effects = {}
    for name_index, name in enumerate(names):
        if name == "verified":
            continue
        original_cols = [name_index, name_index + len(names)]
        positions_in_remaining = [remaining_cols.index(col) for col in original_cols]
        field_effects[name] = np.asarray(
            structured[test][:, original_cols]
            @ no_verified_coefficient[positions_in_remaining]).ravel()
    field_reconstruction_error = float(np.max(np.abs(
        sum(field_effects.values()) + group_effects["tweet_unigrams"]
        + float(no_verified_model.intercept_[0]) - no_verified_logit)))
    if field_reconstruction_error > 1e-10:
        raise ValueError("Individual structured facts do not reconstruct logits")
    reconstructed_logit = (float(no_verified_model.intercept_[0])
                           + sum(group_effects.values()))
    group_reconstruction_error = float(np.max(np.abs(no_verified_logit - reconstructed_logit)))
    if group_reconstruction_error > 1e-10:
        raise ValueError("Frozen no-verified source groups do not reconstruct logits")
    group_diagnostics = {}
    for group_name, effect in group_effects.items():
        masked_class = expit(no_verified_logit - effect) >= no_verified_threshold
        group_diagnostics[group_name] = {
            "nonzero_score_effect_accounts": int(np.count_nonzero(np.abs(effect) > 1e-12)),
            "masked_class_changes": int(np.count_nonzero(no_verified_class != masked_class)),
            "masked_errors": int(np.count_nonzero(masked_class != labels[test])),
            "mean_absolute_logit_effect": float(np.mean(np.abs(effect))),
        }
    field_diagnostics = {}
    for field_name, effect in field_effects.items():
        masked_class = expit(no_verified_logit - effect) >= no_verified_threshold
        field_diagnostics[field_name] = {
            "nonzero_score_effect_accounts": int(np.count_nonzero(np.abs(effect) > 1e-12)),
            "masked_class_changes": int(np.count_nonzero(no_verified_class != masked_class)),
            "masked_errors": int(np.count_nonzero(masked_class != labels[test])),
            "mean_absolute_logit_effect": float(np.mean(np.abs(effect))),
            "min_logit_effect": float(np.min(effect)),
            "max_logit_effect": float(np.max(effect)),
        }
    graph_names = {"sampled_following_count", "sampled_follower_count", "graph_out_degree",
                   "graph_in_degree", "graph_reciprocal_degree"}
    graph_cols_in_remaining = [k for k, original_col in enumerate(remaining_cols)
                               if names[original_col % len(names)] in graph_names]
    graph_contribution = np.asarray(
        structured[test][:, np.asarray(remaining_cols)[graph_cols_in_remaining]]
        @ no_verified_coefficient[graph_cols_in_remaining]).ravel()
    graph_mask_p = expit(no_verified_logit - graph_contribution)
    graph_class = graph_mask_p >= no_verified_threshold
    if np.max(np.abs(graph_contribution - group_effects["native_follow_graph"])) > 1e-10:
        raise ValueError("Graph group does not match prior graph diagnostic")
    split_contingency = {}
    for part, indices in (("train", train), ("dev", dev), ("test", test)):
        split_contingency[part] = {
            f"verified={source_value},label={label}": int(np.count_nonzero(
                (facts[indices, j] == source_value) & (labels[indices] == label)))
            for source_value in (0, 1) for label in (0, 1)
        }
    result = {
        "scope": "TwiBot-20 official train/dev/test; post hoc same-protocol diagnostic, test population previously explored",
        "source_archive_sha256": manifest["source_archive_sha256"],
        "base_facts_sha256": manifest["base_facts_sha256"],
        "frozen_model_artifact_sha256": manifest["model_artifact_sha256"],
        "reference_artifact_sha256": manifest["reference_artifact_sha256"],
        "frozen_replay_max_probability_error": replay_error,
        "verified_label_contingency_by_official_split": split_contingency,
        "protocol": "L1 LogisticRegression C=1 liblinear random_state=0; same train-only TF-IDF and label-blind reference; dev Macro-F1 threshold",
        "original": {"threshold": original_threshold,
                     "dev": original["dev"], "test": metric(labels[test], frozen_p, original_threshold)},
        "without_verified": {
            "threshold": no_verified_threshold,
            "dev": metric(labels[dev], no_verified_dev_p, no_verified_threshold),
            "test": metric(labels[test], no_verified_test_p, no_verified_threshold),
            "nonzero_structured_terms": int(np.count_nonzero(no_verified_coefficient[:42])),
            "nonzero_lexical_terms": int(np.count_nonzero(no_verified_coefficient[42:])),
            "intercept": float(no_verified_model.intercept_[0]),
            "all_zero_input_baseline_margin": float(no_verified_model.intercept_[0] - logit(no_verified_threshold)),
            "all_zero_input_baseline_class": int(expit(float(no_verified_model.intercept_[0])) >= no_verified_threshold),
            "full_class_differs_from_zero_input_baseline": int(np.count_nonzero(
                no_verified_class != (expit(float(no_verified_model.intercept_[0])) >= no_verified_threshold))),
            "group_logit_reconstruction_max_error": group_reconstruction_error,
            "field_logit_reconstruction_max_error": field_reconstruction_error,
            "group_diagnostics": group_diagnostics,
            "field_diagnostics": field_diagnostics,
            "structured_coefficients_by_fact": {
                name: {"upper": float(no_verified_coefficient[remaining_cols.index(index)]),
                       "lower": float(no_verified_coefficient[remaining_cols.index(index + len(names))])}
                for index, name in enumerate(names) if name != "verified"},
            "active_unigram_coefficients": {
                str(vocabulary[index]): float(no_verified_coefficient[42 + index])
                for index in np.flatnonzero(no_verified_coefficient[42:] != 0)},
            "graph_encoded_group_mask_class_changes": int(np.count_nonzero(no_verified_class != graph_class)),
            "graph_encoded_group_mask_errors": int(np.count_nonzero(graph_class != labels[test])),
        },
        "remaining_single_field_controls": single_field_controls,
        "best_remaining_single_field_by_dev_macro_f1": best_single["field"],
        "verified_only": {
            "threshold": single_threshold,
            "dev": metric(labels[dev], single_dev_p, single_threshold),
            "test": metric(labels[test], single_test_p, single_threshold),
            "coefficients": single_model.coef_.ravel().tolist(),
        },
        "paired_original_only_correct": int(np.count_nonzero((frozen_class == labels[test]) &
                                                                  (no_verified_class != labels[test]))),
        "paired_without_verified_only_correct": int(np.count_nonzero((frozen_class != labels[test]) &
                                                                         (no_verified_class == labels[test]))),
        "subgroups": groups,
        "rows": [{"account_id": str(ids[i]), "source_locator": f"Twibot-20/test.json[{positions[i]}].profile.verified",
                  "verified": None if not np.isfinite(facts[i, j]) else int(facts[i, j]),
                  "label": int(labels[i]), "original_probability": float(frozen_p[k]),
                  "without_verified_probability": float(no_verified_test_p[k]),
                  "verified_only_probability": float(single_test_p[k]),
                  "without_verified_graph_mask_probability": float(graph_mask_p[k]),
                  "without_verified_logit": float(no_verified_logit[k]),
                  "without_verified_group_effects": {
                      name: float(effect[k]) for name, effect in group_effects.items()},
                  "without_verified_structured_field_effects": {
                      name: float(effect[k]) for name, effect in field_effects.items()},
                  }
                 for k, i in enumerate(test)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
