"""Frozen, label-free TwiBot-20 inference through audited evidence only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer

from .v3_evidence_reasoner import LabelBlindReference


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TwiBot20SparseRuntime:
    """Predict from 22 named source facts and official tweet text, without labels."""

    def __init__(self, *, archive: Path, base_facts: Path,
                 base_source_audit: Path, active_source_audit: Path,
                 active_cache: Path, screen: Path, selection_lock: Path,
                 artifact: Path, reference: Path, selected_report: Path):
        audit = json.loads(base_source_audit.read_text(encoding="utf-8"))
        active = json.loads(active_source_audit.read_text(encoding="utf-8"))
        dev = json.loads(screen.read_text(encoding="utf-8"))
        lock = json.loads(selection_lock.read_text(encoding="utf-8"))
        report = json.loads(selected_report.read_text(encoding="utf-8"))
        manifest = report["manifest"]
        archive_hash = sha256(archive)
        fact_hash = sha256(base_facts)
        artifact_hash = sha256(artifact)
        reference_hash = sha256(reference)
        if (audit["status"] !=
                "verified_all_labeled_account_fact_cells_against_official_zip"
                or audit["source_archive_sha256"] != archive_hash
                or audit["fact_cache_sha256"] != fact_hash
                or active["status"] !=
                "verified_train_idf_and_recomputed_all_active_unigram_values"
                or active["source_archive_sha256"] != archive_hash
                or active["base_facts_sha256"] != fact_hash
                or active["model_artifact_sha256"] != artifact_hash
                or active["active_cache_sha256"] != sha256(active_cache)
                or dev["source_archive_sha256"] != archive_hash
                or dev["base_facts_sha256"] != fact_hash
                or lock["selected_model"] != "source_backed_unigram_l1_c1"
                or lock["screen_sha256"] != sha256(screen)
                or lock["model_artifact_sha256"] != artifact_hash
                or lock["reference_artifact_sha256"] != reference_hash
                or manifest["source_archive_sha256"] != archive_hash
                or manifest["base_facts_sha256"] != fact_hash
                or manifest["base_source_audit_sha256"] != sha256(base_source_audit)
                or manifest["active_source_audit_sha256"] != sha256(active_source_audit)
                or manifest["active_source_cache_sha256"] != sha256(active_cache)
                or manifest["selection_lock_sha256"] != sha256(selection_lock)
                or manifest["model_artifact_sha256"] != artifact_hash
                or manifest["reference_artifact_sha256"] != reference_hash):
            raise ValueError("frozen TwiBot source, audit or model binding differs")
        self.reference = LabelBlindReference.load(reference)
        if self.reference.version != manifest["reference_version"]:
            raise ValueError("frozen label-blind reference differs")
        with np.load(artifact, allow_pickle=False) as saved:
            vocabulary = saved["vocabulary"].astype(str)
            idf = saved["idf"].astype(float)
            coefficient = saved["coefficient"].astype(float)
            intercept = float(saved["intercept"][0])
            threshold = float(saved["threshold"][0])
            structured_names = saved["structured_names"].astype(str).tolist()
        expected_names = ([f"{name}:upper" for name in self.reference.names] +
                          [f"{name}:lower" for name in self.reference.names])
        if (len(self.reference.names) != 22 or structured_names != expected_names
                or len(vocabulary) != 20000 or len(idf) != len(vocabulary)
                or len(coefficient) != 44 + len(vocabulary)
                or not np.all(np.isfinite(idf)) or not np.all(np.isfinite(coefficient))
                or not np.isfinite(intercept) or not 0 < threshold < 1
                or abs(threshold - manifest["threshold"]) > 1e-12):
            raise ValueError("frozen TwiBot evidence vocabulary or coefficients differ")
        with np.load(active_cache, allow_pickle=False) as checked:
            active_indices = checked["active_indices"].astype(int)
            if (not np.array_equal(active_indices, np.flatnonzero(coefficient[44:] != 0))
                    or not np.array_equal(checked["active_terms"].astype(str),
                                          vocabulary[active_indices])):
                raise ValueError("active source-audited unigram terms differ")
        self.vectorizer = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", ngram_range=(1, 1),
            sublinear_tf=True, dtype=np.float64)
        self.vectorizer.vocabulary_ = {term: i for i, term in enumerate(vocabulary)}
        self.vectorizer.fixed_vocabulary_ = True
        self.vectorizer.idf_ = idf
        self.vocabulary = vocabulary
        self.coefficient = coefficient
        self.intercept = intercept
        self.threshold = threshold
        self.active_indices = active_indices
        self.source_archive_sha256 = archive_hash
        self.artifact_sha256 = artifact_hash

    def predict(self, facts: np.ndarray, contexts: Sequence[str],
                tweets: Sequence[Sequence[str]], account_ids: Sequence[str],
                source_rows: Sequence[tuple[str, int]] | None = None
                ) -> list[dict]:
        """Expose every active logit term; no class label enters this method."""
        values = np.asarray(facts, dtype=float)
        n = len(values)
        if (values.shape != (n, 22) or len(contexts) != n
                or len(tweets) != n or len(account_ids) != n
                or (source_rows is not None and len(source_rows) != n)
                or len(set(account_ids)) != n):
            raise ValueError("source fact, context, tweet or account population differs")
        evidence = self.reference.transform(values, contexts=contexts)
        structured = np.column_stack((
            np.log1p(evidence.upper) * evidence.reliability,
            np.log1p(evidence.lower) * evidence.reliability))
        documents = [" ".join(str(tweet or "") for tweet in row) for row in tweets]
        lexical = self.vectorizer.transform(documents)
        structured_contribution = structured * self.coefficient[:44]
        active = self.active_indices
        lexical_value = lexical[:, active].toarray()
        lexical_contribution = lexical_value * self.coefficient[44 + active]
        logit = (self.intercept + structured_contribution.sum(axis=1)
                 + lexical_contribution.sum(axis=1))
        direct = self.intercept + structured @ self.coefficient[:44] + \
            np.asarray(lexical @ self.coefficient[44:]).ravel()
        if np.max(np.abs(logit - direct)) > 1e-10:
            raise ValueError("evidence terms do not reconstruct the decision")
        results = []
        analyzer = self.vectorizer.build_analyzer() if source_rows is not None else None
        active_terms = set(self.vocabulary[active]) if source_rows is not None else set()
        for row, account_id in enumerate(account_ids):
            source_locator = None
            lexical_locations = None
            if source_rows is not None:
                member, ordinal = source_rows[row]
                if not member.startswith("Twibot-20/") or ordinal < 0:
                    raise ValueError("invalid official source locator")
                source_locator = f"{member}[{ordinal}]"
                lexical_locations = {}
                for tweet_position, tweet in enumerate(tweets[row]):
                    for term in set(analyzer(str(tweet or ""))) & active_terms:
                        lexical_locations.setdefault(term, []).append(
                            f"{source_locator}.tweet[{tweet_position}]")
                if any((lexical_value[row, j] > 0) !=
                       (str(self.vocabulary[index]) in lexical_locations)
                       for j, index in enumerate(active)):
                    raise ValueError("active unigram value differs from tweet locator")
            results.append({
                "dataset": "twibot20", "account_id": str(account_id),
                "reference_version": evidence.reference_version,
                "model_sha256": self.artifact_sha256,
                "source_archive_sha256": self.source_archive_sha256,
                "source_account_locator": source_locator,
                "intercept": self.intercept,
                "structured_names": list(expected for expected in
                                         ([f"{name}:upper" for name in evidence.names] +
                                          [f"{name}:lower" for name in evidence.names])),
                "structured_values": structured[row].tolist(),
                "structured_contributions": structured_contribution[row].tolist(),
                "active_unigram_terms": self.vocabulary[active].tolist(),
                "active_unigram_values": lexical_value[row].tolist(),
                "active_unigram_contributions": lexical_contribution[row].tolist(),
                "active_unigram_source_locations": lexical_locations,
                "logit": float(logit[row]),
                "probability": float(expit(logit[row])),
                "threshold": self.threshold,
                "prediction": int(expit(logit[row]) >= self.threshold),
            })
        return results
