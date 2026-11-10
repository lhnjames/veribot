"""Source-bound 309-channel MGTAB classifier with affirmative Human evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

from .compact_source_population import load_source_population
from .mgtab_full_atomic_runtime import AB_RELATIONS, _sha256
from .mgtab_profile_schema import PROFILE_SCHEMA
from .mgtab_source_path import resolve_mgtab_source_dir
from .v3_evidence_reasoner import EvidenceBatch, LabelBlindReference


class MGTABABPositiveHumanRuntime:
    """Classify a published account row using source facts and Human scores."""

    def __init__(self, root: Path, seed: int):
        if seed not in (0, 1, 2):
            raise ValueError("unsupported MGTAB seed")
        self.root = Path(root).resolve()
        self.seed = seed
        self.variant = "ab_group_positive_human"
        folder = self.root / "output/v3_mgtab_ab_group_positive_human"
        manifest_path = folder / "inference_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit = json.loads((self.root /
            "research/certificates/mgtab_ab_positive_human_inference_manifest_audit.json")
            .read_text(encoding="utf-8"))
        lock_path = folder / "selection_lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if (manifest["status"] != "inference_only_metadata_no_test_results" or
                manifest["model"] != "ab_group_additive_positive_human_309" or
                manifest["selection_lock_sha256"] != _sha256(lock_path) or
                manifest["source_audit_sha256"] != _sha256(self.root /
                    "research/certificates/mgtab_full_source_audit.json") or
                manifest["source_features_sha256"] !=
                lock["source_features_sha256"] or
                audit["status"] !=
                "inference_only_manifest_matches_dev_locked_frozen_model" or
                audit["manifest_sha256"] != _sha256(manifest_path) or
                lock["test_metric_or_prediction_used_to_create_lock"] is not False):
            raise ValueError("positive Human inference manifest differs")
        entry = manifest["runs"][seed]
        model_path = self.root / entry["model_path"]
        reference_path = self.root / entry["reference_path"]
        contract_path = self.root / entry["ab_contract_path"]
        source_metadata_path = self.root / entry["source_metadata_path"]
        if (entry["seed"] != seed or
                _sha256(model_path) != entry["model_sha256"] or
                _sha256(reference_path) != entry["reference_sha256"] or
                _sha256(contract_path) != entry["ab_contract_sha256"] or
                _sha256(source_metadata_path) !=
                entry["source_metadata_sha256"] or
                entry["reference_version"] !=
                lock["reference_version_by_seed"][str(seed)] or
                entry["split_sha256"] != lock["split_sha256_by_seed"][str(seed)]):
            raise ValueError("positive Human frozen artifacts differ")
        self.model = joblib.load(model_path)
        self.model.set_params(n_jobs=1)
        self.reference = LabelBlindReference.load(reference_path)
        if self.reference.version != entry["reference_version"]:
            raise ValueError("positive Human label-blind reference differs")
        self.threshold = float(entry["threshold"])
        if self.threshold != float(lock["threshold_by_seed"][str(seed)]):
            raise ValueError("positive Human dev threshold differs")
        self.data_dir = resolve_mgtab_source_dir(self.root)
        self.structural_path = self.root / "data/processed/mgtab_v3/structural_facts.npz"
        self.neighbor_path = self.root / "data/processed/mgtab_v3/neighbor_facts.npz"
        self.projection_path = self.root / (
            "output/v3_mgtab_ab_spline_content/selected/"
            f"seed{seed}/projection.npz")
        if _sha256(self.data_dir / "features.pt") != lock[
                "source_features_sha256"]:
            raise ValueError("published MGTAB source differs")
        facts, names, ids = load_source_population(
            "mgtab", self.data_dir,
            self.root / f"output/v3_mgtab_ab_spline_content/selected/seed{seed}",
            structural_facts=self.structural_path,
            neighbor_facts=self.neighbor_path,
            inference_metadata=source_metadata_path)
        if facts.shape != (10199, 150) or tuple(names) != self.reference.names:
            raise ValueError("published source fact population differs")
        with np.load(self.neighbor_path, allow_pickle=False) as cache:
            if cache["ids"].astype(str).tolist() != ids:
                raise ValueError("neighbor support source rows differ")
            supports = cache["supports"].astype(float)
        self.facts, self.names, self.ids, self.supports = (
            facts, names, ids, supports)
        self.index = {uid: i for i, uid in enumerate(ids)}
        with np.load(contract_path, allow_pickle=False) as contract:
            signs = contract["monotone_constraints"].astype(int).tolist()
            contract_group_names = contract["source_group_names"].astype(str).tolist()
        self.group_fact_indices = {
            "published_profile_coordinates": list(range(20)),
            **{f"published_relation_{k}": [
                j for j in range(20, 86)
                if names[j].startswith(f"published_{relation}_")]
                for k, relation in enumerate(AB_RELATIONS)},
            "published_tweet_projections": list(range(86, 150)),
        }
        groups = self.group_fact_indices
        if (sorted(j for members in groups.values() for j in members) !=
                list(range(150)) or
                [len(members) for members in groups.values()] !=
                [20, 14, 14, 6, 14, 6, 6, 6, 64]):
            raise ValueError("MGTAB Human source groups do not partition facts")
        source_columns = [members + [j+150 for j in members] + [300+k]
                          for k, members in enumerate(groups.values())]
        if (len(signs) != 309 or signs[300:] != [-1]*9 or
                contract_group_names != list(groups) or
                self.model.get_params()["monotone_constraints"] != signs):
            raise ValueError("positive Human A/B or source groups differ")
        by_feature = {j: k for k, cols in enumerate(source_columns)
                      for j in cols}
        if sorted(by_feature) != list(range(309)):
            raise ValueError("source groups do not partition model inputs")
        for tree in self.model.booster_.dump_model()["tree_info"]:
            used = set()

            def walk(node: dict) -> None:
                if "split_feature" not in node:
                    return
                used.add(by_feature[int(node["split_feature"])])
                walk(node["left_child"])
                walk(node["right_child"])

            walk(tree["tree_structure"])
            if len(used) > 1:
                raise ValueError("positive Human tree crossed source groups")
        self.source_columns = source_columns

    def predict(self, account_id: str) -> dict:
        uid = str(account_id)
        if uid not in self.index:
            raise KeyError("account row is absent from MGTAB release")
        i = self.index[uid]
        batch = self.reference.transform(self.facts[i:i+1])
        reliability = batch.reliability.copy()
        reliability[:, 62:86] *= self.supports[i:i+1] / (
            self.supports[i:i+1] + 5.0)
        atomic = EvidenceBatch(batch.upper, batch.lower, reliability,
                               batch.available, batch.reference_version,
                               batch.names)
        base = np.column_stack((np.log1p(atomic.upper)*atomic.reliability,
                                np.log1p(atomic.lower)*atomic.reliability))
        normality = []
        human_rows = []
        for k, (group, fact_indices) in enumerate(self.group_fact_indices.items()):
            present = bool(np.any(atomic.available[:, fact_indices]))
            deviation = float(np.max(base[:, self.source_columns[k][:-1]]))
            value = math.exp(-deviation) if present else 0.0
            normality.append(value)
            human_rows.append({
                "source_group": group,
                "source_present": present,
                "maximum_tail_deviation": deviation,
                "small_deviation_score": value,
                "source_fact_names": [self.names[j] for j in fact_indices
                                      if bool(atomic.available[0, j])],
                "source_scope": "published tensor and typed relation facts",
            })
        x = np.column_stack((base, np.array(normality, dtype=float)[None, :]))
        probability = float(self.model.predict_proba(x)[0, 1])
        logit = float(self.model.booster_.predict(x, raw_score=True)[0])
        shap = np.asarray(self.model.booster_.predict(x, pred_contrib=True))[0]
        if (shap.shape != (310,) or
                abs(float(shap.sum())-logit) > 1e-9 or
                abs(float(expit(logit))-probability) > 1e-12):
            raise ValueError("positive Human source evidence does not reconstruct score")
        zero = np.zeros_like(x)
        zero_logit = float(self.model.booster_.predict(zero, raw_score=True)[0])
        zero_probability = float(expit(zero_logit))
        without_human = x.copy()
        without_human[:, 300:] = 0
        without_human_p = float(self.model.predict_proba(without_human)[0, 1])
        effects = {}
        group_logit_differences = []
        for group, columns in zip(self.group_fact_indices, self.source_columns):
            masked = x.copy()
            masked[:, columns] = 0
            masked_p = float(self.model.predict_proba(masked)[0, 1])
            masked_logit = float(self.model.booster_.predict(
                masked, raw_score=True)[0])
            difference = logit-masked_logit
            group_logit_differences.append(difference)
            effects[group] = {
                "probability_without_group": masked_p,
                "group_logit_difference": difference,
                "class_changes_without_group": bool(
                    (masked_p >= self.threshold) !=
                    (probability >= self.threshold)),
            }
        additivity_error = abs(logit-zero_logit-sum(group_logit_differences))
        if additivity_error > 1e-9:
            raise ValueError("positive Human source group effects do not add up")
        contribution = shap[:150] + shap[150:300]
        evidence = []
        for j in np.argsort(np.abs(contribution))[::-1]:
            j = int(j)
            raw = float(self.facts[i, j])
            if j < 20:
                source = {"level": "published_normalized_profile_tensor",
                          "file": str(self.data_dir / "features.pt"),
                          "row": i, "column": j,
                          "paper_field": PROFILE_SCHEMA[j][0]}
            elif j < 86:
                source = {"level": "published_typed_relation_tensors_and_derived_fact",
                          "edge_index": str(self.data_dir / "edge_index.pt"),
                          "edge_type": str(self.data_dir / "edge_type.pt"),
                          "account_row": i,
                          "fact_cache": (None if j < 34 else
                                         str(self.structural_path) if j < 62
                                         else str(self.neighbor_path))}
            else:
                source = {"level": "published_averaged_tweet_tensor_train_only_projection",
                          "file": str(self.data_dir / "features.pt"),
                          "row": i, "input_columns": "20:788",
                          "projection_artifact": str(self.projection_path),
                          "semantic_status": "numeric projection; no individual tweet text"}
            evidence.append({
                "fact": self.names[j],
                "raw_value": raw if np.isfinite(raw) else None,
                "upper_tail": float(atomic.upper[0, j]),
                "lower_tail": float(atomic.lower[0, j]),
                "reliability": float(atomic.reliability[0, j]),
                "logit_contribution": float(contribution[j]),
                "source": source,
            })
        return {
            "account_id": uid, "seed": self.seed,
            "variant": self.variant,
            "prediction": int(probability >= self.threshold),
            "probability": probability, "threshold": self.threshold,
            "logit": logit, "intercept_logit": float(shap[-1]),
            "evidence": evidence,
            "positive_human_evidence": human_rows,
            "positive_human_logit_attributions": [
                float(value) for value in shap[300:309]],
            "without_positive_human_evidence_probability": without_human_p,
            "without_positive_human_evidence_changes_class": bool(
                (without_human_p >= self.threshold) !=
                (probability >= self.threshold)),
            "zero_evidence_probability": zero_probability,
            "zero_evidence_prediction": int(zero_probability >= self.threshold),
            "all_evidence_changes_class": bool(
                (zero_probability >= self.threshold) !=
                (probability >= self.threshold)),
            "group_dependencies": effects,
            "group_logit_additivity_error": additivity_error,
            "content_provenance_level": "published_average_tweet_tensor",
        }
