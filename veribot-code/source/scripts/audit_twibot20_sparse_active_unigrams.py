"""Independently recompute every active unigram TF-IDF value from the ZIP."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import zipfile

import ijson
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


TOKEN = re.compile(r"(?u)\b[a-zA-Z]{3,}\b")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tokens(text: str) -> list[str]:
    # Match sklearn's lower-then-Unicode-normalize order. Compatibility
    # characters can expand to uppercase ASCII after the lower() step.
    normalized = unicodedata.normalize("NFKD", str(text or "").lower())
    normalized = "".join(char for char in normalized
                         if not unicodedata.combining(char))
    return [value for value in TOKEN.findall(normalized)
            if value not in ENGLISH_STOP_WORDS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.base_facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        parts = cache["split"].astype(str)
        positions = cache["positions"].astype(int)
    with np.load(args.model_artifact, allow_pickle=False) as saved:
        vocabulary = saved["vocabulary"].astype(str).tolist()
        idf = saved["idf"].astype(float)
        coefficient = saved["coefficient"].astype(float)
    column = {term: j for j, term in enumerate(vocabulary)}
    active = np.flatnonzero(coefficient[44:] != 0)
    active_position = {int(j): k for k, j in enumerate(active)}
    values = np.zeros((len(ids), len(active)), dtype=np.float64)
    document_frequency = np.zeros(len(vocabulary), dtype=np.int64)
    index = 0
    with zipfile.ZipFile(args.zip) as zipped:
        for part in ("train", "dev", "test"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for position, row in enumerate(ijson.items(stream, "item")):
                    if (index >= len(ids) or ids[index] != str(row["ID"])
                            or parts[index] != part
                            or positions[index] != position):
                        raise ValueError("official source account order differs")
                    counts: Counter[int] = Counter()
                    for tweet in row.get("tweet") or []:
                        for term in tokens(tweet):
                            j = column.get(term)
                            if j is not None:
                                counts[j] += 1
                    if part == "train":
                        document_frequency[list(counts)] += 1
                    weighted = {j: (1.0 + np.log(count)) * idf[j]
                                for j, count in counts.items()}
                    norm = float(np.sqrt(sum(value * value
                                             for value in weighted.values())))
                    if norm:
                        for j, weighted_value in weighted.items():
                            k = active_position.get(j)
                            if k is not None:
                                values[index, k] = weighted_value / norm
                    index += 1
    if index != len(ids):
        raise ValueError("official source account count differs")
    expected_idf = np.log((1.0 + int(np.sum(parts == "train"))) /
                          (1.0 + document_frequency)) + 1.0
    idf_difference = float(np.max(np.abs(idf - expected_idf)))
    if idf_difference > 1e-12:
        ranking = np.argsort(-np.abs(idf - expected_idf))[:10]
        raise ValueError("saved train IDF differs from independent source counts: " +
                         repr([(vocabulary[int(j)], float(idf[j]),
                                float(expected_idf[j]), int(document_frequency[j]))
                               for j in ranking]))
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_cache, ids=ids, active_indices=active,
                        active_terms=np.asarray([vocabulary[j] for j in active]),
                        values=values)
    report = {
        "dataset": "twibot20",
        "status": "verified_train_idf_and_recomputed_all_active_unigram_values",
        "account_count": len(ids),
        "vocabulary_size": len(vocabulary),
        "active_unigram_count": len(active),
        "active_fact_cells": int(values.size),
        "nonzero_active_fact_cells": int(np.count_nonzero(values)),
        "max_train_idf_difference": idf_difference,
        "label_fields_read": False,
        "source_archive_sha256": sha256(args.zip),
        "base_facts_sha256": sha256(args.base_facts),
        "model_artifact_sha256": sha256(args.model_artifact),
        "active_cache_sha256": sha256(args.output_cache),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n",
                                  encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
