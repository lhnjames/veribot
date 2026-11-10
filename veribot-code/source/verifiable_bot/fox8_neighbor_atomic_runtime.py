"""Label-free, source-bound Fox8 inference through 74 verifiable atomic facts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from .compact_source_population import load_source_population
from .fox8_adapter import FEATURE_FIELDS
from .fox8_source_path import resolve_fox8_source_dir
from .v3_evidence_reasoner import LabelBlindReference


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Fox8NeighborAtomicRuntime:
    """Load one frozen evidence-only model; never opens an account class file."""

    def __init__(self, root: Path, seed: int):
        if seed not in (0, 1, 2):
            raise ValueError("unsupported frozen seed")
        self.root = Path(root).resolve()
        self.seed = seed
        folder = self.root / "output/v3_fox8_neighbor_atomic_evidence_capacity"
        self.lock = json.loads((folder / "selection_lock.json").read_text(encoding="utf-8"))
        deployment = json.loads((folder / "inference_manifest.json")
                                .read_text(encoding="utf-8"))
        manifest_audit = json.loads((self.root /
            "research/certificates/high_performance_inference_manifests_audit.json")
            .read_text(encoding="utf-8"))
        source_audit_path = (self.root /
            "research/certificates/fox8_neighbor_volume_full_raw_source.json")
        if (manifest_audit["status"] !=
                "inference_only_manifests_match_frozen_model_artifacts"
                or manifest_audit["models"]["fox8"]["manifest_sha256"] !=
                _sha256(folder / "inference_manifest.json")
                or deployment["status"] != "inference_only_metadata_no_test_results"
                or deployment["model"] != "raw_neighbor_74"
                or deployment["selection_lock_sha256"] !=
                _sha256(folder / "selection_lock.json")
                or deployment["source_audit_sha256"] != _sha256(source_audit_path)
                or self.lock["selected"]["variant"] != "raw_neighbor_74"
                or self.lock["config"] != [15, 200, 30, 10.0]
                or self.lock["test_metric_or_prediction_used_to_create_lock"]
                is not False):
            raise ValueError("model selection record differs")
        self.run_dir = folder / f"seed{seed}"
        inference = deployment["runs"][seed]
        model_path = self.run_dir / "model.joblib"
        ref_path = self.run_dir / "label_blind_reference.json.gz"
        source_metadata_path = self.root / inference["source_metadata_path"]
        if (inference["seed"] != seed
                or model_path != self.root / inference["model_path"]
                or ref_path != self.root / inference["reference_path"]
                or _sha256(model_path) != inference["model_sha256"]
                or _sha256(ref_path) != inference["reference_sha256"]
                or _sha256(source_metadata_path) !=
                inference["source_metadata_sha256"]
                or inference["split_sha256"] !=
                self.lock["split_sha256_by_seed"][str(seed)]):
            raise ValueError("frozen model or source reference differs")
        self.model = joblib.load(model_path)
        self.model.set_params(n_jobs=1)
        self.reference = LabelBlindReference.load(ref_path)
        if (self.reference.version != inference["reference_version"] or
                self.reference.version !=
                self.lock["reference_version_by_seed"][str(seed)]):
            raise ValueError("label-blind reference differs")
        self.threshold = float(inference["threshold"])
        if self.threshold != float(self.lock["threshold"]):
            raise ValueError("Fox8 dev-selected threshold differs")
        self.data_dir = resolve_fox8_source_dir(self.root)
        self.features_path = self.data_dir / "features.csv"
        self.event_path = self.root / "data/processed/fox8_v3/interaction_event_index.json"
        self.raw_path = self.root / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
        if (_sha256(self.features_path) != deployment["published_feature_sha256"]
                or _sha256(self.event_path) != deployment["native_event_index_sha256"]):
            raise ValueError("current source differs from frozen source")
        old, old_names, ids = load_source_population(
            "fox8", self.data_dir,
            self.root / f"output/v3_fox8_native_spline_l2/selected/seed{seed}",
            native_facts=self.root / "data/processed/fox8_v3/native_interaction_facts.npz",
            raw_source=self.raw_path,
            inference_metadata=source_metadata_path)
        if old.shape[1] != 43 or len(ids) != 2279:
            raise ValueError("old source fact vocabulary differs")
        features = pd.read_csv(self.features_path)
        if features["user_id"].astype(str).tolist() != ids:
            raise ValueError("current source account order differs")
        index = {uid: j for j, uid in enumerate(ids)}
        neighbors = [set() for _ in ids]
        events = json.loads(self.event_path.read_text(encoding="utf-8"))["events"]
        for event in events:
            u = index.get(str(event["source_id"]))
            v = index.get(str(event["target_id"]))
            if u is not None and v is not None and u != v:
                neighbors[u].add(v)
                neighbors[v].add(u)
        raw = features[list(FEATURE_FIELDS)].to_numpy(dtype=float)
        neighbor_mean = np.full((len(ids), len(FEATURE_FIELDS)), np.nan,
                                dtype=np.float32)
        neighbor_support = np.zeros_like(neighbor_mean, dtype=np.int32)
        for i, adjacent in enumerate(neighbors):
            if not adjacent:
                continue
            selected = raw[sorted(adjacent)]
            valid = np.isfinite(selected)
            counts = valid.sum(axis=0)
            values = np.nansum(selected, axis=0) / np.maximum(counts, 1)
            values[counts == 0] = np.nan
            neighbor_mean[i] = values.astype(np.float32)
            neighbor_support[i] = counts
        fact_names = old_names + [
            f"native_interaction_neighbor_mean_{field}" for field in FEATURE_FIELDS]
        if tuple(fact_names) != self.reference.names:
            raise ValueError("model and source fact names differ")
        self.facts = np.column_stack((old, neighbor_mean))
        self.supports = neighbor_support
        self.ids = ids
        self.index = index
        self.neighbors = [[ids[j] for j in sorted(group)] for group in neighbors]

    def predict(self, account_id: str) -> dict:
        if str(account_id) not in self.index:
            raise KeyError("account is absent from frozen source population")
        i = self.index[str(account_id)]
        batch = self.reference.transform(self.facts[i:i + 1])
        reliability = batch.reliability.copy()
        reliability[:, 43:] *= self.supports[i:i + 1] / (
            self.supports[i:i + 1] + 5.0)
        matrix = np.column_stack((np.log1p(batch.upper) * reliability,
                                  np.log1p(batch.lower) * reliability))
        probability = float(self.model.predict_proba(matrix)[0, 1])
        logit = float(self.model.booster_.predict(matrix, raw_score=True)[0])
        shap = np.asarray(self.model.booster_.predict(matrix, pred_contrib=True))[0]
        if (abs(float(shap.sum()) - logit) > 1e-9
                or abs(float(expit(logit)) - probability) > 1e-12):
            raise ValueError("frozen prediction lacks exact evidence reconstruction")
        empty_probability = float(self.model.predict_proba(
            np.zeros_like(matrix))[0, 1])
        without_neighbors = matrix.copy()
        without_neighbors[:, 43:74] = 0.0
        without_neighbors[:, 117:148] = 0.0
        without_neighbor_probability = float(self.model.predict_proba(
            without_neighbors)[0, 1])
        full_class = probability >= self.threshold
        contribution = shap[:74] + shap[74:148]
        ordered = np.argsort(np.abs(contribution))[::-1]
        terms = []
        for rank in ordered:
            j = int(rank)
            value = self.facts[i, j]
            terms.append({
                "fact": self.reference.names[j],
                "raw_value": float(value) if np.isfinite(value) else None,
                "upper_tail": float(batch.upper[0, j]),
                "lower_tail": float(batch.lower[0, j]),
                "reliability": float(reliability[0, j]),
                "logit_contribution": float(contribution[j]),
                "source_locator": (
                    {"feature_table": str(self.features_path),
                     "field": FEATURE_FIELDS[j - 43],
                     "neighbor_account_ids": self.neighbors[i],
                     "relation_events": str(self.event_path)}
                    if j >= 43 else
                    {"feature_table": str(self.features_path),
                     "native_relation_fact_cache": str(self.root /
                        "data/processed/fox8_v3/native_interaction_facts.npz"),
                     "account_id": str(account_id)}),
            })
        return {
            "account_id": str(account_id), "seed": self.seed,
            "prediction": int(full_class),
            "probability": probability, "logit": logit,
            "intercept_logit": float(shap[-1]),
            "evidence_dependency": {
                "all_evidence_zero_probability": empty_probability,
                "all_evidence_zero_class": int(empty_probability >= self.threshold),
                "all_evidence_changes_class": bool(
                    full_class != (empty_probability >= self.threshold)),
                "native_neighbor_evidence_zero_probability":
                    without_neighbor_probability,
                "native_neighbor_evidence_changes_class": bool(
                    full_class != (without_neighbor_probability >= self.threshold)),
                "intervention": "zero encoded evidence columns in the frozen model; source records remain unchanged",
            },
            "reference_version": self.reference.version,
            "source_feature_sha256": _sha256(self.features_path),
            "native_event_index_sha256": _sha256(self.event_path),
            "evidence": terms,
        }
