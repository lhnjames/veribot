"""Dev-only capacity control using the same source-bound TwiBot evidence inputs."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import ijson
import numpy as np
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from scipy.sparse import issparse
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import groups, metric, sha256, threshold_on_dev
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lightgbm", action="store_true")
    parser.add_argument("--evaluate-selected", action="store_true")
    args = parser.parse_args()
    if args.evaluate_selected and not args.lightgbm:
        raise ValueError("the locked candidate is a LightGBM model")
    root = Path(__file__).resolve().parents[2]
    raw_manifest = json.loads((root / "data/raw/twibot20/manifest.json")
                              .read_text(encoding="utf-8"))
    base_path = root / "data/processed/twibot20_v3/facts.npz"
    neighbor_path = root / "data/processed/twibot20_v3/neighbor_active_unigram_facts.npz"
    neighbor_manifest = json.loads(neighbor_path.with_suffix(".json")
                                   .read_text(encoding="utf-8"))
    selected_path = root / "output/v3_twibot20_sparse_text_evidence/sparse_l1_c1_model.npz"
    if (neighbor_manifest["fact_cache_sha256"] != sha256(neighbor_path)
            or neighbor_manifest["base_facts_sha256"] != sha256(base_path)
            or neighbor_manifest["source_archive_sha256"] != raw_manifest["archive_sha256"]):
        raise ValueError("neighbor and own-account sources differ")
    with np.load(base_path, allow_pickle=False) as base:
        ids = base["ids"].astype(str)
        parts = base["split"].astype(str)
        positions = base["positions"].astype(int)
        labels = base["labels"].astype(int)
        values = base["values"].astype(float)
        contexts = base["contexts"].astype(str)
        base_names = base["names"].astype(str).tolist()
    with np.load(neighbor_path, allow_pickle=False) as neighbor:
        if (not np.array_equal(neighbor["ids"].astype(str), ids)
                or not np.array_equal(neighbor["split"].astype(str), parts)
                or not np.array_equal(neighbor["positions"].astype(int), positions)):
            raise ValueError("native-neighbor and owner rows differ")
        neighbor_values = neighbor["values"].astype(float)
        neighbor_support = neighbor["supports"].astype(float)
        neighbor_names = neighbor["names"].astype(str).tolist()
    train = np.flatnonzero(parts == "train")
    dev = np.flatnonzero(parts == "dev")
    test = np.flatnonzero(parts == "test")
    documents = []
    test_rows = []
    with zipfile.ZipFile(raw_manifest["archive_path"]) as zipped:
        for part in (("train", "dev", "test") if args.evaluate_selected
                     else ("train", "dev")):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    index = len(documents)
                    if (ids[index] != str(row["ID"]) or parts[index] != part
                            or positions[index] != position):
                        raise ValueError("own-account tweet source order differs")
                    documents.append(" ".join(str(tweet or "")
                                              for tweet in row.get("tweet") or []))
                    if part == "test":
                        test_rows.append(row)
    own_reference = LabelBlindReference.load(
        root / "output/v3_twibot20_sparse_text_evidence/reference.json.gz")
    own = own_reference.transform(values[:len(documents)],
                                  contexts=contexts[:len(documents)])
    structured = np.column_stack((np.log1p(own.upper) * own.reliability,
                                  np.log1p(own.lower) * own.reliability))
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", ngram_range=(1, 1),
        min_df=10, max_features=20000, sublinear_tf=True, dtype=np.float64)
    text_train = vectorizer.fit_transform([documents[index] for index in train])
    text_dev = vectorizer.transform([documents[index] for index in dev])
    text_test = (vectorizer.transform([documents[index] for index in test])
                 if args.evaluate_selected else None)
    with np.load(selected_path, allow_pickle=False) as selected:
        if (not np.array_equal(vectorizer.get_feature_names_out(),
                               selected["vocabulary"].astype(str))
                or not np.array_equal(vectorizer.idf_,
                                      selected["idf"].astype(float))):
            raise ValueError("train-only text evidence vocabulary differs")
    own_train = hstack((csr_matrix(structured[train]), text_train), format="csr")
    own_dev = hstack((csr_matrix(structured[dev]), text_dev), format="csr")
    own_test = (hstack((csr_matrix(structured[test]), text_test), format="csr")
                if args.evaluate_selected else None)
    neighbor_reference = LabelBlindReference(min_context_n=30).fit(
        neighbor_values[train], neighbor_names)
    neighbor = neighbor_reference.transform(neighbor_values[:len(documents)])
    reliability = neighbor.reliability.copy()
    for direction in range(2):
        support = neighbor_support[:len(documents), direction]
        reliability[:, direction * 109:(direction + 1) * 109] *= (
            support[:, None] / (support[:, None] + 5.0))
    neighbor_matrix = np.column_stack((
        np.log1p(neighbor.upper) * reliability,
        np.log1p(neighbor.lower) * reliability))
    if args.evaluate_selected:
        prior_path = root / "output/v3_twibot20_neighbor_unigram/lightgbm_dev_only.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if (prior["source_archive_sha256"] != raw_manifest["archive_sha256"]
                or prior["base_facts_sha256"] != sha256(base_path)
                or prior["neighbor_fact_cache_sha256"] != sha256(neighbor_path)
                or prior["own_reference_version"] != own_reference.version
                or prior["neighbor_reference_version"] != neighbor_reference.version):
            raise ValueError("validation selection source binding differs")
        chosen = max(prior["grid"],
                     key=lambda row: (row["validation"]["macro_f1"],
                                      row["validation"]["acc"]))
        if not chosen["use_native_neighbor_text"]:
            raise ValueError("locked validation winner does not consume neighbor evidence")
        fit_x = hstack((own_train, csr_matrix(neighbor_matrix[train])), format="csr")
        eval_x = hstack((own_dev, csr_matrix(neighbor_matrix[dev])), format="csr")
        test_x = hstack((own_test, csr_matrix(neighbor_matrix[test])), format="csr")
        model = LGBMClassifier(n_estimators=300, num_leaves=chosen["capacity"],
                               learning_rate=0.05, min_child_samples=30,
                               reg_lambda=1.0, random_state=0, n_jobs=4,
                               verbosity=-1)
        model.fit(fit_x, labels[train])
        dev_probability = model.predict_proba(eval_x)[:, 1]
        threshold = threshold_on_dev(dev_probability, labels[dev])
        replayed = metric(labels[dev], dev_probability, threshold)
        if (abs(threshold - chosen["threshold"]) > 1e-12
                or any(abs(replayed[key] - chosen["validation"][key]) > 1e-12
                       for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("validation winner did not replay before fixed test")
        probability = model.predict_proba(test_x)[:, 1]
        raw_score = model.booster_.predict(test_x, raw_score=True)
        contributions = model.booster_.predict(test_x, pred_contrib=True)
        reconstructed = np.asarray(contributions.sum(axis=1)).ravel()
        maximum_logit_difference = float(np.max(np.abs(reconstructed - raw_score)))
        if maximum_logit_difference > 1e-9:
            raise ValueError("test evidence contribution sum differs from model logit")
        masked_x = hstack((own_test, csr_matrix(neighbor_matrix[test].shape)),
                          format="csr")
        masked_probability = model.predict_proba(masked_x)[:, 1]
        full_prediction = probability >= threshold
        masked_prediction = masked_probability >= threshold
        fact_group_names = groups(base_names)
        dimension_columns = {
            "user_information": [
                tail * len(base_names) + base_names.index(name)
                for name in (fact_group_names["profile_size"] +
                             fact_group_names["profile_presentation"])
                for tail in (0, 1)],
            "behavioral_pattern": [
                tail * len(base_names) + base_names.index(name)
                for name in fact_group_names["tweet_sample"]
                for tail in (0, 1)],
            "own_text_unigrams": list(range(44, 44 + len(vectorizer.get_feature_names_out()))),
            "interactive_situation": [
                tail * len(base_names) + base_names.index(name)
                for name in fact_group_names["native_follow_graph"]
                for tail in (0, 1)] +
                list(range(44 + len(vectorizer.get_feature_names_out()),
                           test_x.shape[1])),
        }
        assigned = [index for columns in dimension_columns.values()
                    for index in columns]
        if len(assigned) != test_x.shape[1] or len(set(assigned)) != len(assigned):
            raise ValueError("four evidence dimensions do not partition model input")
        dimension_mask = {}
        for name, columns in dimension_columns.items():
            keep = np.ones(test_x.shape[1], dtype=np.float64)
            keep[columns] = 0.0
            masked_dimension = model.predict_proba(test_x.multiply(keep).tocsr())[:, 1]
            masked_class = masked_dimension >= threshold
            dimension_mask[name] = {
                "input_columns": len(columns),
                "changed_class_accounts": int(np.count_nonzero(
                    full_prediction != masked_class)),
                "full_correct": int(np.count_nonzero(full_prediction == labels[test])),
                "masked_correct": int(np.count_nonzero(masked_class == labels[test])),
                "full_to_masked_mean_absolute_probability_change": float(
                    np.mean(np.abs(probability - masked_dimension))),
            }
        destination = root / "output/v3_twibot20_neighbor_unigram/selected"
        destination.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(destination / "model.txt"))
        np.savez_compressed(destination / "test_predictions.npz",
                            ids=ids[test], labels=labels[test], probability=probability,
                            raw_score=raw_score, contribution_sum=reconstructed,
                            masked_probability=masked_probability)
        incidence_path = root / "data/processed/twibot20_v3/neighbor_active_unigram_incidence.npz"
        incidence_manifest = json.loads(incidence_path.with_suffix(".json")
                                        .read_text(encoding="utf-8"))
        if (incidence_manifest["fact_cache_sha256"] != sha256(incidence_path)
                or incidence_manifest["source_archive_sha256"] !=
                raw_manifest["archive_sha256"]):
            raise ValueError("neighbor source locator cache differs")
        with np.load(incidence_path, allow_pickle=False) as source:
            source_ids = source["ids"].astype(str)
            source_parts = source["parts"].astype(int)
            source_positions = source["positions"].astype(int)
            source_counts = source["tweet_counts"].astype(int)
            source_incidence = source["incidence"].astype(np.uint8)
            active_words = source["vocabulary"].astype(str)
        source_lookup = {account_id: index for index, account_id in enumerate(source_ids)}
        if (len(test_rows) != len(test) or len(base_names) != 22
                or len(neighbor_names) != 218 or len(active_words) != 109):
            raise ValueError("test evidence locator vocabulary differs")
        feature_names = ([f"own_upper_{name}" for name in base_names] +
                         [f"own_lower_{name}" for name in base_names] +
                         [f"own_unigram_{word}" for word in vectorizer.get_feature_names_out()] +
                         [f"neighbor_upper_{name}" for name in neighbor_names] +
                         [f"neighbor_lower_{name}" for name in neighbor_names])
        if len(feature_names) + 1 != contributions.shape[1]:
            raise ValueError("evidence feature names differ from tree contributions")
        analyzer = vectorizer.build_analyzer()
        cards_path = destination / "evidence_cards.jsonl"
        with cards_path.open("w", encoding="utf-8") as cards:
            for row_index, official_index in enumerate(test):
                contribution_row = (contributions.getrow(row_index).toarray().ravel()
                                    if issparse(contributions) else
                                    np.asarray(contributions[row_index]).ravel())
                top_indices = np.argsort(np.abs(contribution_row[:-1]))[-8:][::-1]
                needed_words = {feature_names[index][len("own_unigram_"):]
                                for index in top_indices
                                if 44 <= index < 44 + len(vectorizer.get_feature_names_out())}
                own_tweet_hits = {word: [] for word in needed_words}
                if needed_words:
                    for tweet_index, tweet in enumerate(test_rows[row_index].get("tweet") or []):
                        for word in set(analyzer(str(tweet or ""))) & needed_words:
                            own_tweet_hits[word].append(tweet_index)
                top = []
                for index in top_indices:
                    item = {"feature": feature_names[index],
                            "model_input": float(test_x[row_index, index]),
                            "logit_contribution": float(contribution_row[index]),
                            "source_member": "Twibot-20/test.json",
                            "source_owner_row": int(positions[official_index]),
                            "source_owner_id": ids[official_index]}
                    if index < 44:
                        item["raw_fact_value"] = (float(values[official_index, index % 22])
                                                  if np.isfinite(values[official_index, index % 22])
                                                  else None)
                    elif index < 44 + len(vectorizer.get_feature_names_out()):
                        word = feature_names[index][len("own_unigram_"):]
                        item["raw_tweet_indices_containing_word"] = own_tweet_hits[word]
                    else:
                        relative = index - 44 - len(vectorizer.get_feature_names_out())
                        fact_index = relative % 218
                        direction_index = fact_index // 109
                        word_index = fact_index % 109
                        direction = ("following", "follower")[direction_index]
                        word = active_words[word_index]
                        neighbor_ids = set(map(str, (test_rows[row_index].get("neighbor") or {})
                                               .get(direction) or ()))
                        matching = sorted(source_lookup[neighbor_id]
                                          for neighbor_id in neighbor_ids
                                          if neighbor_id in source_lookup and
                                          source_counts[source_lookup[neighbor_id]] > 0)
                        positives = [source_index for source_index in matching
                                     if source_incidence[source_index, word_index]]
                        if len(matching) != int(neighbor_support[official_index, direction_index]):
                            raise ValueError("evidence card neighbor support differs")
                        item.update({
                            "native_relation": direction,
                            "raw_word": word,
                            "raw_neighbor_share":
                                (float(neighbor_values[official_index, fact_index])
                                 if matching else None),
                            "neighbor_support": len(matching),
                            "neighbor_word_hit_count": len(positives),
                            "sample_source_neighbors_with_word": [
                                {"account_id": source_ids[source_index],
                                 "member": "Twibot-20/" +
                                 ("train", "dev", "test", "support")[source_parts[source_index]] +
                                 ".json",
                                 "row": int(source_positions[source_index])}
                                for source_index in positives[:5]],
                        })
                    top.append(item)
                card = {
                    "account_id": ids[official_index],
                    "prediction": int(full_prediction[row_index]),
                    "probability": float(probability[row_index]),
                    "raw_logit": float(raw_score[row_index]),
                    "contribution_bias": float(contribution_row[-1]),
                    "contribution_sum": float(reconstructed[row_index]),
                    "top_absolute_contributions": top,
                    "top_contributions_are_attributions_not_individual_causal_proofs": True,
                }
                cards.write(json.dumps(card, ensure_ascii=False) + "\n")
        result = {
            "dataset": "twibot20",
            "scope": "validation_locked_native_neighbor_text_lightgbm_fixed_test",
            "selection_artifact_sha256": sha256(prior_path),
            "source_archive_sha256": raw_manifest["archive_sha256"],
            "base_facts_sha256": sha256(base_path),
            "neighbor_fact_cache_sha256": sha256(neighbor_path),
            "selected_num_leaves": chosen["capacity"],
            "threshold": threshold,
            "validation_replay": replayed,
            "test": metric(labels[test], probability, threshold),
            "test_evidence_contribution_logit_max_abs_difference":
                maximum_logit_difference,
            "test_neighbor_group_mask": {
                "changed_class_accounts": int(np.count_nonzero(
                    full_prediction != masked_prediction)),
                "full_correct": int(np.count_nonzero(full_prediction == labels[test])),
                "masked_correct": int(np.count_nonzero(masked_prediction == labels[test])),
            },
            "test_four_dimension_group_mask": dimension_mask,
            "test_population_previously_screened_by_other_candidates": True,
            "model_sha256": sha256(destination / "model.txt"),
            "test_predictions_sha256": sha256(destination / "test_predictions.npz"),
            "test_evidence_cards": len(test),
            "test_evidence_cards_sha256": sha256(cards_path),
            "neighbor_source_locator_cache_sha256": sha256(incidence_path),
        }
        (destination / "report.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return
    results = []
    for use_neighbor in (False, True):
        fit_x = (hstack((own_train, csr_matrix(neighbor_matrix[train])), format="csr")
                 if use_neighbor else own_train)
        eval_x = (hstack((own_dev, csr_matrix(neighbor_matrix[dev])), format="csr")
                  if use_neighbor else own_dev)
        capacities = ((15, 31) if args.lightgbm else (0.01, 0.1, 1.0))
        for capacity in capacities:
            model = (LGBMClassifier(n_estimators=300, num_leaves=capacity,
                                    learning_rate=0.05, min_child_samples=30,
                                    reg_lambda=1.0, random_state=0, n_jobs=4,
                                    verbosity=-1)
                     if args.lightgbm else
                     LinearSVC(C=capacity, random_state=0, max_iter=3000))
            model.fit(fit_x, labels[train])
            score = (model.predict_proba(eval_x)[:, 1] if args.lightgbm else
                     expit(model.decision_function(eval_x)))
            threshold = threshold_on_dev(score, labels[dev])
            result = {
                "use_native_neighbor_text": use_neighbor,
                "capacity": capacity,
                "validation": metric(labels[dev], score, threshold),
                "threshold": threshold,
                "nonzero_neighbor_coefficients":
                    (int(np.count_nonzero(model.feature_importances_[own_dev.shape[1]:]))
                     if args.lightgbm else
                     int(np.count_nonzero(model.coef_.ravel()[own_dev.shape[1]:])))
                    if use_neighbor else 0,
            }
            if use_neighbor:
                masked_score = (
                    model.predict_proba(hstack((own_dev,
                                                csr_matrix(neighbor_matrix[dev].shape)),
                                               format="csr"))[:, 1]
                    if args.lightgbm else
                    expit(model.intercept_[0] +
                          own_dev @ model.coef_.ravel()[:own_dev.shape[1]]))
                result["neighbor_group_mask_changed_class_accounts"] = int(
                    np.count_nonzero((score >= threshold) !=
                                     (masked_score >= threshold)))
            results.append(result)
            print(use_neighbor, capacity, result["validation"], flush=True)
    destination = (root / "output/v3_twibot20_neighbor_unigram" /
                   ("lightgbm_dev_only.json" if args.lightgbm else
                    "linear_svc_dev_only.json"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "dataset": "twibot20",
        "scope": ("official_train_dev_only_lightgbm_evidence_capacity_control"
                  if args.lightgbm else
                  "official_train_dev_only_linear_svc_evidence_capacity_control"),
        "test_labels_used_for_fit_or_selection": False,
        "source_archive_sha256": raw_manifest["archive_sha256"],
        "base_facts_sha256": sha256(base_path),
        "neighbor_fact_cache_sha256": sha256(neighbor_path),
        "own_reference_version": own_reference.version,
        "neighbor_reference_version": neighbor_reference.version,
        "grid": results,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
