"""Source-bound inference for the validation-locked 20-concept Fox8 model."""

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
from .v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from .v3_group_tail_bottleneck import collapse_group_tails


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _AuditedFox8Source:
    """Load 74 source facts without opening either classifier's weights."""

    def __init__(self, root: Path, seed: int,
                 reference_names: tuple[str, ...],
                 expected_source_metadata_sha256: str):
        self.root = root
        self.data_dir = resolve_fox8_source_dir(root)
        self.features_path = self.data_dir / "features.csv"
        self.event_path = root / "data/processed/fox8_v3/interaction_event_index.json"
        self.raw_path = root / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
        audit = json.loads((root /
            "research/certificates/fox8_neighbor_volume_full_raw_source.json"
                            ).read_text(encoding="utf-8"))
        if (audit["status"] != "all_target_accounts_and_neighbor_raw_profiles_verified"
                or _sha256(self.features_path) != audit["published_feature_sha256"]
                or _sha256(self.event_path) != audit["event_index_sha256"]
                or _sha256(self.raw_path) != audit["raw_source_sha256"]):
            raise ValueError("Fox8 raw source differs from audited population")
        source_metadata = root / (
            f"output/v3_fox8_targeted_twenty_ab_group/seed{seed}/source_metadata.json")
        if _sha256(source_metadata) != expected_source_metadata_sha256:
            raise ValueError("Fox8 inference source metadata differs")
        self.source_metadata_sha256 = _sha256(source_metadata)
        old, old_names, ids = load_source_population(
            "fox8", self.data_dir,
            root / f"output/v3_fox8_targeted_twenty_ab_group/seed{seed}",
            native_facts=root / "data/processed/fox8_v3/native_interaction_facts.npz",
            raw_source=self.raw_path,
            inference_metadata=source_metadata)
        if old.shape != (2279, 43):
            raise ValueError("Fox8 source fact population differs")
        features = pd.read_csv(self.features_path)
        if features["user_id"].astype(str).tolist() != ids:
            raise ValueError("Fox8 source account order differs")
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
        support = np.zeros_like(neighbor_mean, dtype=np.int32)
        for i, adjacent in enumerate(neighbors):
            if not adjacent:
                continue
            selected = raw[sorted(adjacent)]
            valid = np.isfinite(selected)
            counts = valid.sum(axis=0)
            values = np.nansum(selected, axis=0) / np.maximum(counts, 1)
            values[counts == 0] = np.nan
            neighbor_mean[i] = values.astype(np.float32)
            support[i] = counts
        names = tuple(old_names + [
            f"native_interaction_neighbor_mean_{field}"
            for field in FEATURE_FIELDS])
        if names != reference_names:
            raise ValueError("20-concept source vocabulary differs")
        self.facts = np.column_stack((old, neighbor_mean))
        self.supports = support
        self.ids = ids
        self.index = index
        self.neighbors = [[ids[j] for j in sorted(group)] for group in neighbors]
        self.names = names


class Fox8TargetedTwentyRuntime:
    """Read audited account facts and pass only 20 fixed concepts to the tree."""

    def __init__(self, root: Path, seed: int):
        if seed not in (0, 1, 2):
            raise ValueError("unsupported frozen Fox8 seed")
        self.root = Path(root).resolve()
        self.seed = seed
        self.folder = self.root / "output/v3_fox8_targeted_twenty_ab_group"
        lock_path = self.folder / "selection_lock.json"
        self.lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if self.lock["status"] != "selected_on_official_dev_before_targeted_twenty_test":
            raise ValueError("20-concept validation lock differs")
        self.record = self.lock["runs"][seed]
        self.run_dir = self.folder / f"seed{seed}"
        for filename, key in (("model.joblib", "model_sha256"),
                              ("reference.json.gz", "reference_sha256"),
                              ("groups.json", "groups_sha256"),
                              ("source_metadata.json", "source_metadata_sha256")):
            if _sha256(self.run_dir / filename) != self.record[key]:
                raise ValueError("20-concept frozen artifact differs")
        self.model = joblib.load(self.run_dir / "model.joblib")
        self.model.set_params(n_jobs=1)
        self.reference = LabelBlindReference.load(
            self.run_dir / "reference.json.gz")
        self.groups = json.loads((self.run_dir / "groups.json").read_text(
            encoding="utf-8"))
        self.threshold = float(self.record["threshold"])
        if (self.reference.version != self.record["reference_version"] or
                len(self.groups) != 10 or
                self.model.get_params()["monotone_constraints"] !=
                self.record["monotone_constraints"]):
            raise ValueError("20-concept model contract differs")
        self.source = _AuditedFox8Source(
            self.root, seed, self.reference.names,
            self.record["source_metadata_sha256"])
        if (_sha256(self.root /
                    "data/processed/fox8_v3/native_neighbor_profile_facts.npz")
                != self.record["source_cache_sha256"]):
            raise ValueError("source facts differ from 20-concept training")
        self.ids = self.source.ids
        self.index = self.source.index
        self.kept = [self.reference.names.index(name)
                     for name in self.record["retained_fact_names"]]
        if (len(self.kept) != 33 or
                [self.reference.names[j] for j in self.kept] !=
                self.record["retained_fact_names"]):
            raise ValueError("source-to-concept selection differs")
        self.group_columns = {
            group: [2*k, 2*k+1, 20+2*k, 21+2*k]
            for k, group in enumerate(self.groups)
        }

    def _source_locator(self, account_id: str, fact: str | None) -> dict | None:
        if fact is None:
            return None
        if fact == "native_interaction_neighbor_mean_statuses_count":
            return {
                "source_level": "raw_account_profiles_and_native_events",
                "field": "statuses_count",
                "feature_table": str(self.source.features_path),
                "neighbor_account_ids": self.source.neighbors[
                    self.index[account_id]],
                "relation_events": str(self.source.event_path),
            }
        if fact.startswith("native_reply_") or fact.startswith(
                "native_retweet_"):
            return {
                "source_level": "raw_native_interaction_events",
                "account_id": account_id,
                "relation_events": str(self.source.event_path),
                "fact_cache": str(self.root /
                    "data/processed/fox8_v3/native_interaction_facts.npz"),
            }
        return {
            "source_level": "raw_profile_and_sampled_tweets",
            "account_id": account_id,
            "fact": fact,
            "raw_records": str(self.source.raw_path),
            "feature_table": str(self.source.features_path),
        }

    def predict_many(self, account_ids: list[str]) -> list[dict]:
        if (not account_ids or len(set(map(str, account_ids))) != len(account_ids)
                or not set(map(str, account_ids)) <= set(self.index)):
            raise ValueError("account IDs must be unique and in the source population")
        account_ids = list(map(str, account_ids))
        rows = np.asarray([self.index[account] for account in account_ids],
                          dtype=int)
        atomic = self.reference.transform(self.source.facts[rows])
        reliability = atomic.reliability.copy()
        reliability[:, 43:74] *= self.source.supports[rows] / (
            self.source.supports[rows] + 5.0)
        selected = EvidenceBatch(
            atomic.upper[:, self.kept], atomic.lower[:, self.kept],
            reliability[:, self.kept], atomic.available[:, self.kept],
            atomic.reference_version,
            tuple(atomic.names[j] for j in self.kept))
        concepts, winners = collapse_group_tails(selected, self.groups)
        if list(concepts.names) != self.record["concept_names"]:
            raise ValueError("20-concept vocabulary differs")
        matrix = np.column_stack((np.log1p(concepts.upper),
                                  np.log1p(concepts.lower)))
        logit = np.asarray(self.model.booster_.predict(
            matrix, raw_score=True), dtype=float)
        probability = expit(logit)
        baseline = np.asarray(self.model.booster_.predict(
            np.zeros_like(matrix), raw_score=True), dtype=float)
        without = {}
        for group, columns in self.group_columns.items():
            altered = matrix.copy()
            altered[:, columns] = 0.0
            without[group] = np.asarray(self.model.booster_.predict(
                altered, raw_score=True), dtype=float)
        group_effect = np.column_stack([
            logit - without[group] for group in self.groups])
        max_additivity = float(np.max(np.abs(
            baseline + group_effect.sum(axis=1) - logit)))
        if max_additivity > 1e-9:
            raise ValueError("source-group logit contributions are not additive")
        results = []
        for position, account_id in enumerate(account_ids):
            terms = []
            for j, name in enumerate(concepts.names):
                winner = int(winners[name][position])
                fact = selected.names[winner] if winner >= 0 else None
                value = (float(self.source.facts[rows[position], self.kept[winner]])
                         if winner >= 0 and np.isfinite(
                             self.source.facts[rows[position], self.kept[winner]])
                         else None)
                terms.append({
                    "concept": name,
                    "value": float(concepts.upper[position, j]),
                    "available": bool(concepts.available[position, j]),
                    "winning_source_fact": fact,
                    "winning_source_value": value,
                    "source_locator": self._source_locator(account_id, fact),
                })
            groups = []
            for k, group in enumerate(self.groups):
                without_probability = float(expit(without[group][position]))
                groups.append({
                    "group": group,
                    "logit_effect": float(group_effect[position, k]),
                    "probability_without_group": without_probability,
                    "class_changes_without_group": bool(
                        (without_probability >= self.threshold) !=
                        (probability[position] >= self.threshold)),
                })
            results.append({
                "account_id": account_id, "seed": self.seed,
                "prediction": int(probability[position] >= self.threshold),
                "probability": float(probability[position]),
                "logit": float(logit[position]),
                "threshold": self.threshold,
                "zero_evidence_logit": float(baseline[position]),
                "zero_evidence_probability": float(expit(baseline[position])),
                "group_additivity_error": float(abs(
                    baseline[position] + group_effect[position].sum() -
                    logit[position])),
                "concepts": terms,
                "source_group_effects": groups,
                "model_binding": {
                    "model_sha256": self.record["model_sha256"],
                    "reference_sha256": self.record["reference_sha256"],
                    "source_metadata_sha256": self.source.source_metadata_sha256,
                    "validation_lock_sha256": _sha256(
                        self.folder / "selection_lock.json"),
                },
            })
        return results

    def predict(self, account_id: str) -> dict:
        return self.predict_many([account_id])[0]
