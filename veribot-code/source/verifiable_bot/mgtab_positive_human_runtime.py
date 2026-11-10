"""Label-free source-to-positive-Human-evidence MGTAB inference."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import joblib
import numpy as np
from scipy.special import expit

from .compact_source_population import load_source_population
from .mgtab_twenty_concept_runtime import (
    MGTABTwentyConceptRuntime, _selected, _sha256)
from .mgtab_source_path import resolve_mgtab_source_dir
from .v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from .v3_group_tail_bottleneck import collapse_group_tails

warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")


class MGTABPositiveHumanRuntime(MGTABTwentyConceptRuntime):
    """Classify a published account row through source-backed fifty-channel evidence."""

    def __init__(self, root: Path, seed: int):
        if seed not in (0, 1, 2):
            raise ValueError("unsupported MGTAB seed")
        self.root = Path(root).resolve()
        self.seed = seed
        self.folder = self.root / "output/v3_mgtab_positive_human_evidence"
        manifest_path = self.root / "output/v3_mgtab_positive_human_evidence/inference_manifest.json"
        self.positive_deployment = json.loads(
            manifest_path.read_text(encoding="utf-8"))
        lock_path = self.root / "output/v3_mgtab_positive_human_evidence/selection_lock.json"
        self.positive_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        deployment = self.positive_deployment
        if (deployment["status"] !=
                "inference_only_positive_human_evidence_metadata" or
                deployment["selection_lock_sha256"] != _sha256(lock_path) or
                self.positive_lock["test_metric_or_prediction_used_to_create_lock"] is not False):
            raise ValueError("positive Human deployment metadata differs")
        run = deployment["runs"][seed]
        model_path = self.root / run["model_path"]
        reference_path = self.root / run["reference_path"]
        source_meta_path = self.root / run["source_metadata_path"]
        if (run["seed"] != seed or
                _sha256(model_path) != run["model_sha256"] or
                _sha256(reference_path) != run["reference_sha256"] or
                _sha256(source_meta_path) != run["source_metadata_sha256"] or
                run["selected_content"] !=
                self.positive_lock["selected_content_by_seed"][str(seed)]):
            raise ValueError("frozen source, model, or group vocabulary differs")
        self.model = joblib.load(model_path)
        self.model.set_params(n_jobs=1)
        self.reference = LabelBlindReference.load(reference_path)
        if self.reference.version != run["reference_version"]:
            raise ValueError("label-blind reference version differs")
        self.threshold = float(run["threshold"])
        self.data_dir = resolve_mgtab_source_dir(self.root)
        self.source_metadata_path = source_meta_path
        source_audit_path = self.root / "research/certificates/mgtab_full_source_audit.json"
        if _sha256(source_audit_path) != deployment["source_audit_sha256"]:
            raise ValueError("published source audit differs")
        self.source_hashes = {
            "features": _sha256(self.data_dir / "features.pt"),
            "edges": _sha256(self.data_dir / "edge_index.pt"),
            "edge_types": _sha256(self.data_dir / "edge_type.pt"),
        }
        self.structural_path = self.root / "data/processed/mgtab_v3/structural_facts.npz"
        self.neighbor_path = self.root / "data/processed/mgtab_v3/neighbor_facts.npz"
        self.projection_path = (self.root /
            f"output/v3_mgtab_ab_spline_content/selected/seed{seed}/projection.npz")
        facts, names, ids = load_source_population(
            "mgtab", self.data_dir,
            self.root / f"output/v3_mgtab_ab_spline_content/selected/seed{seed}",
            structural_facts=self.structural_path,
            neighbor_facts=self.neighbor_path,
            inference_metadata=self.source_metadata_path)
        if facts.shape != (10199, 150) or tuple(names) != self.reference.names:
            raise ValueError("published fact population differs")
        with np.load(self.neighbor_path, allow_pickle=False) as cache:
            if cache["ids"].astype(str).tolist() != ids:
                raise ValueError("neighbor support population differs")
            supports = cache["supports"].astype(float)
        self.facts, self.names, self.ids, self.supports = facts, names, ids, supports
        self.index = {uid: i for i, uid in enumerate(ids)}
        profile_indices = [3, 6, 13, 14]
        direct = [j for j in range(20, 86) if "_neighbor_" not in names[j]]
        neighbor = [j for j in range(20, 86) if "_neighbor_" in names[j]]
        self.groups = {f"profile_{j:02d}": [names[j]] for j in profile_indices}
        self.groups["native_relation_event_structure"] = [names[j] for j in direct]
        self.groups["native_relation_neighbor_profile"] = [names[j] for j in neighbor]
        for name in run["selected_content"]:
            self.groups[f"published_content_coordinate_{int(name[-3:]):03d}"] = [name]
        self.kept = [j for j, name in enumerate(names)
                     if name in {fact for members in self.groups.values()
                                 for fact in members}]
        if len(self.groups) != 10 or len(self.kept) != 74:
            raise ValueError("frozen twenty-concept group vocabulary differs")
        self.group_columns = [[2*k, 2*k+1, 20+2*k, 21+2*k, 40+k]
                              for k in range(10)]
        signs = self.model.get_params()["monotone_constraints"]
        if len(signs) != 50 or signs[40:] != [-1]*10:
            raise ValueError("positive Human monotone constraints differ")

    def predict(self, account_id: str) -> dict:
        uid = str(account_id)
        if uid not in self.index:
            raise KeyError("account row is absent from MGTAB release")
        i = self.index[uid]
        atomic = self.reference.transform(self.facts[i:i+1])
        reliability = atomic.reliability.copy()
        reliability[:, 62:86] *= self.supports[i:i+1] / (
            self.supports[i:i+1] + 5.0)
        atomic = EvidenceBatch(atomic.upper, atomic.lower, reliability,
                               atomic.available, atomic.reference_version,
                               atomic.names)
        retained = _selected(atomic, self.kept)
        concepts, winners = collapse_group_tails(retained, self.groups)
        base = np.column_stack((np.log1p(concepts.upper)*concepts.reliability,
                                np.log1p(concepts.lower)*concepts.reliability))
        present = np.array([
            np.any(concepts.available[:, 2*k:2*k+2], axis=1)
            for k in range(10)], dtype=float).T
        deviation = np.column_stack([
            np.maximum(base[:, 2*k], base[:, 2*k+1])
            for k in range(10)])
        normality = present*np.exp(-deviation)
        x = np.column_stack((base, normality))
        probability = float(self.model.predict_proba(x)[0, 1])
        logit = float(self.model.booster_.predict(x, raw_score=True,
                                                  num_threads=1)[0])
        shap = np.asarray(self.model.booster_.predict(
            x, pred_contrib=True, num_threads=1))[0]
        if (shap.shape != (51,) or
                abs(float(shap.sum())-logit) > 1e-9 or
                abs(float(expit(logit))-probability) > 1e-12):
            raise ValueError("source evidence does not reconstruct prediction")
        base_logit = float(self.model.booster_.predict(
            np.zeros_like(x), raw_score=True, num_threads=1)[0])
        group_effects = {}
        deltas = []
        for name, cols in zip(self.groups, self.group_columns):
            masked = x.copy()
            masked[:, cols] = 0
            masked_probability = float(self.model.predict_proba(masked)[0, 1])
            masked_logit = float(self.model.booster_.predict(
                masked, raw_score=True, num_threads=1)[0])
            delta = logit-masked_logit
            deltas.append(delta)
            group_effects[name] = {
                "model_logit_attribution": float(shap[cols].sum()),
                "exact_zero_mask_logit_difference": delta,
                "probability_without_group": masked_probability,
                "class_changes_without_group": bool(
                    (masked_probability >= self.threshold) !=
                    (probability >= self.threshold)),
            }
        if abs(logit-base_logit-sum(deltas)) > 1e-9:
            raise ValueError("source group effects do not reconstruct score")
        evidence = []
        for k, concept_name in enumerate(concepts.names):
            winner = int(winners[concept_name][0])
            fact = retained.names[winner] if winner >= 0 else None
            raw = float(self.facts[i, self.kept[winner]]) if winner >= 0 else None
            if raw is not None and not np.isfinite(raw):
                raw = None
            evidence.append({
                "concept": concept_name,
                "value": float(concepts.upper[0, k]),
                "winning_source_fact": fact,
                "winning_source_value": raw,
                "source": self._source_locator(i, fact) if fact else None,
                "model_logit_attribution": float(shap[k]+shap[k+20]),
            })
        human_evidence = []
        for k, group in enumerate(self.groups):
            source_rows = evidence[2*k:2*k+2]
            cited = [row for row in source_rows if row["source"] is not None]
            if bool(cited) != bool(present[0, k]):
                raise ValueError("Human evidence source availability differs")
            human_evidence.append({
                "source_group": group,
                "source_present": bool(present[0, k]),
                "deviation_score": float(deviation[0, k]),
                "human_normality_value": float(normality[0, k]),
                "source_facts": cited,
                "model_logit_attribution": float(shap[40+k]),
            })
        without_human = x.copy()
        without_human[:, 40:] = 0
        without_human_probability = float(
            self.model.predict_proba(without_human)[0, 1])
        return {
            "account_id": uid,
            "prediction": int(probability >= self.threshold),
            "probability": probability,
            "threshold": self.threshold,
            "logit": logit,
            "intercept_logit": float(shap[-1]),
            "zero_evidence_logit": base_logit,
            "zero_evidence_probability": float(expit(base_logit)),
            "without_positive_human_evidence_probability": without_human_probability,
            "without_positive_human_evidence_changes_class": bool(
                (without_human_probability >= self.threshold) !=
                (probability >= self.threshold)),
            "source_group_effects": group_effects,
            "evidence": evidence,
            "positive_human_evidence": human_evidence,
            "content_provenance_level": "published_average_tweet_tensor",
        }
