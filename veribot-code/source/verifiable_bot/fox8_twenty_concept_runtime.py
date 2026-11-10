"""Label-free source-bound inference for the Fox8 twenty-concept classifier."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from .compact_source_population import load_source_population
from .fox8_adapter import FEATURE_FIELDS
from .v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from .v3_group_tail_bottleneck import collapse_group_tails

warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Fox8TwentyConceptRuntime:
    """Rebuild twenty evidence concepts from the frozen raw-source population."""

    def __init__(self, root: Path, seed: int):
        if seed not in (0, 1, 2):
            raise ValueError("unsupported Fox8 seed")
        self.root, self.seed = Path(root).resolve(), seed
        folder = self.root / "output/v3_fox8_targeted_neighbor_20_concepts"
        lock_path = folder / "selection_lock_ab_group_tree.json"
        deployment_path = self.root / "output/v3_current_compact_deployment/fox8_manifest.json"
        deployment_audit_path = self.root / "research/certificates/current_compact_deployment_manifest_audit.json"
        self.lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
        deployment_audit = json.loads(deployment_audit_path.read_text(encoding="utf-8"))
        audited_row = next(row for row in deployment_audit["runs"]
                           if row["dataset"] == "fox8")
        if (self.deployment["selection_lock_sha256"] != _sha256(lock_path)
                or self.deployment["status"] !=
                "inference_only_metadata_from_audited_frozen_model"
                or audited_row["manifest_sha256"] != _sha256(deployment_path)
                or self.lock["selected"]["removed_group"] != "presentation_flags"):
            raise ValueError("Fox8 compact model selection differs")
        run_dir = folder / f"seed{seed}"
        model_path = run_dir / "ab_group_tree.joblib"
        ref_path = run_dir / "label_blind_reference.json.gz"
        locked_run = self.deployment["runs"][seed]
        if (locked_run["seed"] != seed
                or _sha256(model_path) != locked_run["model_sha256"]
                or _sha256(ref_path) != locked_run["reference_sha256"]):
            raise ValueError("Frozen Fox8 compact model or reference differs")
        self.model = joblib.load(model_path)
        self.model.set_params(n_jobs=1)
        self.reference = LabelBlindReference.load(ref_path)
        if self.reference.version != self.lock["reference_version_by_seed"][str(seed)]:
            raise ValueError("Frozen label-blind reference differs")
        self.threshold = float(locked_run["threshold"])
        if self.threshold != float(self.lock["threshold_by_seed"][str(seed)]):
            raise ValueError("Fox8 sealed decision threshold differs")
        self.data_dir = Path("D:/Code/SAHG/SAHG/code/data/fox8_23")
        self.source_metadata_path = (self.root /
            f"output/v3_current_compact_deployment/fox8_source_seed{seed}.json")
        if _sha256(self.source_metadata_path) != locked_run["source_metadata_sha256"]:
            raise ValueError("Fox8 inference source metadata differs")
        self.raw_path = self.root / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
        self.event_path = self.root / "data/processed/fox8_v3/interaction_event_index.json"
        self.feature_path = self.data_dir / "features.csv"
        raw_audit_path = self.root / "research/certificates/fox8_neighbor_volume_full_raw_source.json"
        raw_audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
        if (raw_audit["raw_source_sha256"] != _sha256(self.raw_path)
                or raw_audit["published_feature_sha256"] != _sha256(self.feature_path)
                or raw_audit["event_index_sha256"] != _sha256(self.event_path)
                or self.deployment["source_audit_sha256"] != _sha256(raw_audit_path)):
            raise ValueError("Fox8 raw source or native graph differs")
        trace_path = self.root / raw_audit["trace_path"]
        if _sha256(trace_path) != raw_audit["trace_sha256"]:
            raise ValueError("Fox8 raw neighbor source trace differs")
        with gzip.open(trace_path, "rt", encoding="utf-8") as source:
            raw_neighbor_trace = {row["account_id"]: row for row in
                                  (json.loads(line) for line in source)}
        self.raw_neighbor_trace = raw_neighbor_trace
        self.raw_audit = raw_audit
        old, old_names, ids = load_source_population(
            "fox8", self.data_dir,
            self.root / "output/v3_fox8_native_spline_l2/selected" / f"seed{seed}",
            native_facts=self.root / "data/processed/fox8_v3/native_interaction_facts.npz",
            raw_source=self.raw_path,
            inference_metadata=self.source_metadata_path)
        if old.shape != (2279, 43):
            raise ValueError("Fox8 source fact population differs")
        features = pd.read_csv(self.feature_path)
        if features["user_id"].astype(str).tolist() != ids:
            raise ValueError("Fox8 published feature account order differs")
        index = {uid: k for k, uid in enumerate(ids)}
        self.raw_lines = {}
        with gzip.open(self.raw_path, "rt", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                self.raw_lines.setdefault(str(json.loads(line)["user_id"]), number)
        if len(self.raw_lines) != len(ids):
            raise ValueError("Fox8 raw record locator population differs")
        neighbors = [set() for _ in ids]
        self.event_positions = {uid: [] for uid in ids}
        for position, event in enumerate(json.loads(
                self.event_path.read_text(encoding="utf-8"))["events"]):
            for uid in (str(event["source_id"]), str(event["target_id"])):
                if uid in self.event_positions:
                    self.event_positions[uid].append(position)
            a, b = index.get(str(event["source_id"])), index.get(str(event["target_id"]))
            if a is not None and b is not None and a != b:
                neighbors[a].add(b)
                neighbors[b].add(a)
        source_profile = features[list(FEATURE_FIELDS)].to_numpy(dtype=float)
        means = np.full((len(ids), len(FEATURE_FIELDS)), np.nan, dtype=np.float32)
        supports = np.zeros_like(means, dtype=np.int32)
        for i, adjacent in enumerate(neighbors):
            if adjacent:
                selected = source_profile[sorted(adjacent)]
                counts = np.isfinite(selected).sum(axis=0)
                average = np.nansum(selected, axis=0) / np.maximum(counts, 1)
                average[counts == 0] = np.nan
                means[i] = average.astype(np.float32)
                supports[i] = counts
        cache_path = self.root / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
        if _sha256(cache_path) != raw_audit["fact_cache_sha256"]:
            raise ValueError("Neighbor fact source cache differs")
        with np.load(cache_path, allow_pickle=False) as cache:
            if (cache["ids"].astype(str).tolist() != ids
                    or not np.array_equal(cache["values"], means, equal_nan=True)
                    or not np.array_equal(cache["supports"], supports)):
                raise ValueError("Neighbor facts do not rebuild from published graph")
        new_names = [f"native_interaction_neighbor_mean_{field}"
                     for field in FEATURE_FIELDS]
        if tuple(old_names + new_names) != self.reference.names:
            raise ValueError("Fox8 fact vocabulary differs from reference")
        self.facts = np.column_stack((old, means))
        self.supports = supports
        self.ids = ids
        self.index = index
        self.neighbors = [[ids[j] for j in sorted(group)] for group in neighbors]
        old_groups = json.loads((self.root / "research/fox8_20_concept_groups.json")
                                .read_text(encoding="utf-8"))
        removed = set(old_groups["presentation_flags"])
        neighbor_fact = "native_interaction_neighbor_mean_statuses_count"
        self.groups = {name: members for name, members in old_groups.items()
                       if name != "presentation_flags"}
        self.groups["native_neighbor_volume"] = [neighbor_fact]
        names = old_names + new_names
        self.kept = [j for j, name in enumerate(names)
                     if (j < 43 and name not in removed) or name == neighbor_fact]
        self.group_cols = [[2*k, 2*k+1, 20+2*k, 21+2*k] for k in range(10)]
        if len(self.kept) != 33 or len(self.groups) != 10:
            raise ValueError("Fox8 twenty-concept source partition differs")

    def _source_locator(self, uid: str, fact: str) -> dict:
        if fact == "native_interaction_neighbor_mean_statuses_count":
            return {"raw_profile_gzip_sha256": self.raw_audit["raw_source_sha256"],
                    "native_event_index_sha256": self.raw_audit["event_index_sha256"],
                    "neighbor_profiles": self.raw_neighbor_trace[uid]["raw_neighbor_values"]}
        if fact.startswith("native_"):
            return {"native_event_index_sha256": self.raw_audit["event_index_sha256"],
                    "event_positions_zero_based": self.event_positions[uid],
                    "account_id": uid}
        if fact in {"followers_count", "friends_count", "statuses_count",
                    "favourites_count", "listed_count", "description_len", "name_len",
                    "screen_name_len", "has_location", "has_url", "has_description",
                    "ff_ratio"}:
            return {"feature_table_sha256": self.raw_audit["published_feature_sha256"],
                    "raw_gzip_line": self.raw_lines[uid], "account_id": uid}
        return {"raw_tweet_archive_sha256": self.raw_audit["raw_source_sha256"],
                "raw_gzip_line": self.raw_lines[uid], "account_id": uid}

    def predict(self, account_id: str) -> dict:
        uid = str(account_id)
        if uid not in self.index:
            raise KeyError("account is absent from Fox8 release")
        i = self.index[uid]
        atomic = self.reference.transform(self.facts[i:i+1])
        reliability = atomic.reliability.copy()
        reliability[:, 43:] *= self.supports[i:i+1] / (self.supports[i:i+1] + 5.0)
        atomic = EvidenceBatch(atomic.upper[:, self.kept],
                               atomic.lower[:, self.kept],
                               reliability[:, self.kept],
                               atomic.available[:, self.kept],
                               atomic.reference_version,
                               tuple(atomic.names[j] for j in self.kept))
        concept, winners = collapse_group_tails(atomic, self.groups)
        x = np.column_stack((np.log1p(concept.upper) * concept.reliability,
                             np.log1p(concept.lower) * concept.reliability))
        probability = float(self.model.predict_proba(x)[0, 1])
        logit = float(self.model.booster_.predict(x, raw_score=True)[0])
        shap = np.asarray(self.model.booster_.predict(x, pred_contrib=True))[0]
        if (shap.shape != (41,) or abs(float(shap.sum()) - logit) > 1e-9
                or abs(float(expit(logit)) - probability) > 1e-12):
            raise ValueError("Fox8 compact evidence does not reconstruct score")
        effects = {}
        for name, cols in zip(self.groups, self.group_cols):
            masked = x.copy()
            masked[:, cols] = 0
            p = float(self.model.predict_proba(masked)[0, 1])
            effects[name] = {
                "model_logit_attribution": float(shap[cols].sum()),
                "probability_without_group": p,
                "class_changes_without_group": bool(
                    (p >= self.threshold) != (probability >= self.threshold))}
        evidence = []
        for k, name in enumerate(concept.names):
            winner = int(winners[name][0])
            fact_name = atomic.names[winner] if winner >= 0 else None
            raw = (float(self.facts[i, self.kept[winner]])
                   if winner >= 0 else None)
            if raw is not None and not np.isfinite(raw):
                raw = None
            evidence.append({
                "concept": name, "value": float(concept.upper[0, k]),
                "winning_fact": fact_name, "winning_fact_value": raw,
                "source_locator": self._source_locator(uid, fact_name) if fact_name else None,
                "logit_attribution_from_model": float(shap[k] + shap[k+20]),
                "source": ({"account_id": uid, "feature_table": str(self.feature_path),
                            "native_relation_events": str(self.event_path),
                            "neighbor_account_ids": self.neighbors[i]}
                           if fact_name == "native_interaction_neighbor_mean_statuses_count"
                           else {"account_id": uid, "raw_archive": str(self.raw_path),
                                 "native_relation_events": str(self.event_path)}
                           if fact_name else None)})
        return {"account_id": uid, "prediction": int(probability >= self.threshold),
                "probability": probability, "threshold": self.threshold,
                "logit": logit, "intercept_logit": float(shap[-1]),
                "source_group_effects": effects, "evidence": evidence,
                "all_evidence_zero_probability": float(self.model.predict_proba(
                    np.zeros_like(x))[0, 1])}

    def predict_with_atomic_fact_effects(self, account_id: str) -> dict:
        """Report exact effects of withholding each cited source fact.

        A fact is withheld from the evidence builder. This does not edit the
        released raw record or native interaction graph.
        """
        base = self.predict(account_id)
        uid = base["account_id"]
        i = self.index[uid]
        full = self.reference.transform(self.facts[i:i+1])
        reliability = full.reliability.copy()
        reliability[:, 43:] *= self.supports[i:i+1] / (
            self.supports[i:i+1] + 5.0)
        atomic = EvidenceBatch(full.upper[:, self.kept],
                               full.lower[:, self.kept],
                               reliability[:, self.kept],
                               full.available[:, self.kept],
                               full.reference_version,
                               tuple(full.names[j] for j in self.kept))
        concept, winners = collapse_group_tails(atomic, self.groups)
        original = [int(winners[name][0]) for name in concept.names]
        source_locators = {}
        for item in base["evidence"]:
            fact = item["winning_fact"]
            if fact is not None:
                if fact in source_locators and source_locators[fact] != item["source_locator"]:
                    raise ValueError("cited fact has inconsistent source locators")
                source_locators[fact] = item["source_locator"]
        effects = []
        for fact_index in sorted(set(k for k in original if k >= 0)):
            high = atomic.upper.copy()
            low = atomic.lower.copy()
            weight = atomic.reliability.copy()
            available = atomic.available.copy()
            high[:, fact_index] = low[:, fact_index] = 0.0
            weight[:, fact_index] = 0.0
            available[:, fact_index] = False
            altered = EvidenceBatch(high, low, weight, available,
                                    atomic.reference_version, atomic.names)
            revised, replacement = collapse_group_tails(altered, self.groups)
            x = np.column_stack((np.log1p(revised.upper) * revised.reliability,
                                 np.log1p(revised.lower) * revised.reliability))
            p = float(self.model.predict_proba(x)[0, 1])
            fact = atomic.names[fact_index]
            changed_concepts = [
                name for j, name in enumerate(concept.names)
                if int(replacement[name][0]) != original[j]]
            if not changed_concepts or fact not in source_locators:
                raise ValueError("cited winner withholding failed source linkage")
            changes_class = (p >= self.threshold) != bool(base["prediction"])
            changes_score = abs(p - base["probability"]) > 1e-10
            effects.append({
                "fact": fact,
                "source_locator": source_locators[fact],
                "replacement_concepts": changed_concepts,
                "probability_without_fact": p,
                "probability_changes_gt_1e_10": changes_score,
                "class_changes": changes_class,
                "effect_level": ("class_critical" if changes_class else
                                 "score_relevant" if changes_score else
                                 "context_only"),
            })
        base["atomic_fact_effects"] = effects
        base["class_critical_source_facts"] = [
            row["fact"] for row in effects if row["class_changes"]]
        base["score_relevant_source_facts"] = [
            row["fact"] for row in effects
            if row["probability_changes_gt_1e_10"] and not row["class_changes"]]
        return base
