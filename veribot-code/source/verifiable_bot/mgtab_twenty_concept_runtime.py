"""Label-free, source-bound inference for the compact MGTAB evidence model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

from .compact_source_population import load_source_population
from .mgtab_profile_schema import PROFILE_SCHEMA
from .mgtab_source_path import resolve_mgtab_source_dir
from .v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from .v3_group_tail_bottleneck import collapse_group_tails


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected(batch: EvidenceBatch, indices: list[int]) -> EvidenceBatch:
    return EvidenceBatch(batch.upper[:, indices], batch.lower[:, indices],
                         batch.reliability[:, indices], batch.available[:, indices],
                         batch.reference_version,
                         tuple(batch.names[j] for j in indices))


class MGTABTwentyConceptRuntime:
    """Recompute each account decision from published tensors without labels."""

    def __init__(self, root: Path, seed: int):
        if seed not in (0, 1, 2):
            raise ValueError("unsupported MGTAB seed")
        self.root = Path(root).resolve()
        self.seed = seed
        self.folder = self.root / "output/v3_mgtab_targeted_20_concept_ab_group_tree"
        lock_path = self.folder / "selection_lock.json"
        deployment_path = self.root / "output/v3_current_compact_deployment/mgtab_manifest.json"
        deployment_audit_path = self.root / "research/certificates/current_compact_deployment_manifest_audit.json"
        self.lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
        deployment_audit = json.loads(deployment_audit_path.read_text(encoding="utf-8"))
        audited_row = next(row for row in deployment_audit["runs"]
                           if row["dataset"] == "mgtab")
        if (self.deployment["selection_lock_sha256"] != _sha256(lock_path)
                or self.deployment["status"] !=
                "inference_only_metadata_from_audited_frozen_model"
                or audited_row["manifest_sha256"] != _sha256(deployment_path)
                or self.lock["selected"]["scheme"] !=
                "four_profile_two_relation_four_selected_content"):
            raise ValueError("Compact MGTAB selection differs")
        self.run_dir = self.folder / f"seed{seed}"
        model_path = self.run_dir / "model.joblib"
        reference_path = self.run_dir / "reference.json.gz"
        locked_run = self.deployment["runs"][seed]
        if (locked_run["seed"] != seed
                or _sha256(model_path) != locked_run["model_sha256"]
                or _sha256(reference_path) != locked_run["reference_sha256"]):
            raise ValueError("Frozen compact model or reference differs")
        self.model = joblib.load(model_path)
        self.reference = LabelBlindReference.load(reference_path)
        if self.reference.version != self.lock["reference_version_by_seed"][str(seed)]:
            raise ValueError("Label-blind reference version differs")
        self.threshold = float(locked_run["threshold"])
        if self.threshold != float(self.lock["threshold_by_seed"][str(seed)]):
            raise ValueError("MGTAB sealed decision threshold differs")
        self.data_dir = resolve_mgtab_source_dir(self.root)
        self.source_metadata_path = (self.root /
            f"output/v3_current_compact_deployment/mgtab_source_seed{seed}.json")
        if _sha256(self.source_metadata_path) != locked_run["source_metadata_sha256"]:
            raise ValueError("MGTAB inference source metadata differs")
        source_audit_path = self.root / "research/certificates/mgtab_full_source_audit.json"
        if _sha256(source_audit_path) != self.deployment["source_audit_sha256"]:
            raise ValueError("MGTAB published source audit differs from sealed metadata")
        self.source_hashes = {
            "features": _sha256(self.data_dir / "features.pt"),
            "edges": _sha256(self.data_dir / "edge_index.pt"),
            "edge_types": _sha256(self.data_dir / "edge_type.pt"),
        }
        self.structural_path = self.root / "data/processed/mgtab_v3/structural_facts.npz"
        self.neighbor_path = self.root / "data/processed/mgtab_v3/neighbor_facts.npz"
        self.projection_path = (self.root / "output/v3_mgtab_ab_spline_content/selected" /
                                f"seed{seed}/projection.npz")
        facts, names, ids = load_source_population(
            "mgtab", self.data_dir,
            self.root / "output/v3_mgtab_ab_spline_content/selected" / f"seed{seed}",
            structural_facts=self.structural_path,
            neighbor_facts=self.neighbor_path,
            inference_metadata=self.source_metadata_path)
        if (facts.shape != (10199, 150) or names != list(self.reference.names)
                or _sha256(self.data_dir / "features.pt") !=
                self.lock["source_features_sha256"]):
            raise ValueError("Published source facts or account population differ")
        with np.load(self.neighbor_path, allow_pickle=False) as cache:
            if cache["ids"].astype(str).tolist() != ids:
                raise ValueError("Neighbor source population differs")
            supports = cache["supports"].astype(float)
        self.facts, self.names, self.ids, self.supports = facts, names, ids, supports
        self.index = {uid: i for i, uid in enumerate(ids)}
        selected_content = self.lock["selected_content_by_seed"][str(seed)]
        profile_indices = [3, 6, 13, 14]
        direct = [j for j in range(20, 86) if "_neighbor_" not in names[j]]
        neighbor = [j for j in range(20, 86) if "_neighbor_" in names[j]]
        self.groups = {f"profile_{j:02d}": [names[j]] for j in profile_indices}
        self.groups["native_relation_event_structure"] = [names[j] for j in direct]
        self.groups["native_relation_neighbor_profile"] = [names[j] for j in neighbor]
        for name in selected_content:
            self.groups[f"published_content_coordinate_{int(name[-3:]):03d}"] = [name]
        self.kept = [j for j, name in enumerate(names)
                     if name in {fact for members in self.groups.values()
                                 for fact in members}]
        if len(self.groups) != 10 or len(self.kept) != 74:
            raise ValueError("Frozen twenty-concept group vocabulary differs")
        self.group_columns = [[2*k, 2*k+1, 20+2*k, 21+2*k] for k in range(10)]
        if len(self.model.get_params()["monotone_constraints"]) != 40:
            raise ValueError("Frozen monotone constraint length differs")

    def _source_locator(self, original_index: int, fact_name: str) -> dict:
        if fact_name.startswith("published_profile_coordinate_"):
            coordinate = int(fact_name[-2:])
            return {"level": "published_normalized_profile_tensor",
                    "features_sha256": self.source_hashes["features"],
                    "row": original_index, "column": coordinate,
                    "paper_table_a5_alias": PROFILE_SCHEMA[coordinate][0]}
        if fact_name.startswith("published_tweet_projection_"):
            return {"level": "published_average_tweet_tensor_projection",
                    "features_sha256": self.source_hashes["features"],
                    "row": original_index, "input_columns": "20:788",
                    "projection_sha256": _sha256(self.projection_path),
                    "output_coordinate": int(fact_name[-3:])}
        return {"level": "published_typed_edge_and_profile_tensors",
                "edge_index_sha256": self.source_hashes["edges"],
                "edge_type_sha256": self.source_hashes["edge_types"],
                "features_sha256": self.source_hashes["features"],
                "account_row": original_index, "derived_fact": fact_name}

    def predict(self, account_id: str) -> dict:
        uid = str(account_id)
        if uid not in self.index:
            raise KeyError("account row is absent from MGTAB release")
        i = self.index[uid]
        atomic = self.reference.transform(self.facts[i:i+1])
        reliability = atomic.reliability.copy()
        reliability[:, 62:86] *= self.supports[i:i+1] / (self.supports[i:i+1] + 5.0)
        atomic = EvidenceBatch(atomic.upper, atomic.lower, reliability,
                               atomic.available, atomic.reference_version, atomic.names)
        retained = _selected(atomic, self.kept)
        concepts, winners = collapse_group_tails(retained, self.groups)
        x = np.column_stack((np.log1p(concepts.upper) * concepts.reliability,
                             np.log1p(concepts.lower) * concepts.reliability))
        probability = float(self.model.predict_proba(x)[0, 1])
        logit = float(self.model.booster_.predict(x, raw_score=True)[0])
        shap = np.asarray(self.model.booster_.predict(x, pred_contrib=True))[0]
        if (shap.shape != (41,) or abs(float(shap.sum()) - logit) > 1e-9
                or abs(float(expit(logit)) - probability) > 1e-12):
            raise ValueError("Compact evidence does not reconstruct prediction")
        base_logit = float(self.model.booster_.predict(
            np.zeros_like(x), raw_score=True)[0])
        groups = {}
        exact_deltas = []
        for k, (name, cols) in enumerate(zip(self.groups, self.group_columns)):
            masked = x.copy()
            masked[:, cols] = 0
            masked_p = float(self.model.predict_proba(masked)[0, 1])
            masked_logit = float(self.model.booster_.predict(
                masked, raw_score=True)[0])
            delta = logit - masked_logit
            exact_deltas.append(delta)
            groups[name] = {
                "model_logit_attribution": float(shap[cols].sum()),
                "exact_zero_mask_logit_difference": delta,
                "probability_without_group": masked_p,
                "class_changes_without_group": bool(
                    (masked_p >= self.threshold) != (probability >= self.threshold))}
        if abs(logit - base_logit - sum(exact_deltas)) > 1e-9:
            raise ValueError("Source-group effects do not sum to score")
        evidence = []
        for k, name in enumerate(concepts.names):
            winner = int(winners[name][0])
            fact_name = retained.names[winner] if winner >= 0 else None
            raw_value = (float(self.facts[i, self.kept[winner]])
                         if winner >= 0 else None)
            if raw_value is not None and not np.isfinite(raw_value):
                raw_value = None
            evidence.append({
                "concept": name,
                "value": float(concepts.upper[0, k]),
                "winning_source_fact": fact_name,
                "winning_source_value": raw_value,
                "model_logit_attribution": float(shap[k] + shap[k+20]),
                "source": self._source_locator(i, fact_name) if fact_name else None})
        return {
            "account_id": uid,
            "prediction": int(probability >= self.threshold),
            "probability": probability, "threshold": self.threshold,
            "logit": logit, "intercept_logit": float(shap[-1]),
            "zero_evidence_logit": base_logit,
            "source_group_effects": groups, "evidence": evidence,
            "content_provenance_level": "published_average_tweet_tensor",
        }
