"""Label-free source-bound MGTAB inference through 150 numeric atomic facts."""

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
from .v3_evidence_reasoner import LabelBlindReference


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


GROUPS = {
    "published_profile": slice(0, 20),
    "published_native_relations": slice(20, 86),
    "published_tweet_projection": slice(86, 150),
}

AB_RELATIONS = ("followers", "friends", "mention", "reply",
                "quoted", "url", "hashtag")


class MGTABFullAtomicRuntime:
    """Accept a published tensor row ID and return its frozen evidence ledger."""

    def __init__(self, root: Path, seed: int, *, variant: str = "full_atomic"):
        if seed not in (0, 1, 2):
            raise ValueError("unsupported frozen seed")
        folders = {
            "full_atomic": "output/v3_mgtab_full_atomic_evidence_lightgbm",
            "ab_group_additive": "output/v3_mgtab_ab_group_additive_tree",
        }
        if variant not in folders:
            raise ValueError("unsupported MGTAB evidence model variant")
        self.root = Path(root).resolve()
        self.seed = seed
        self.variant = variant
        folder = self.root / folders[variant]
        self.lock = json.loads((folder / "selection_lock.json")
                               .read_text(encoding="utf-8"))
        deployment = json.loads((folder / "inference_manifest.json")
                                .read_text(encoding="utf-8"))
        if variant == "ab_group_additive":
            manifest_audit = json.loads((self.root /
                "research/certificates/high_performance_inference_manifests_audit.json")
                .read_text(encoding="utf-8"))
            if (manifest_audit["status"] !=
                    "inference_only_manifests_match_frozen_model_artifacts" or
                    manifest_audit["models"]["mgtab_ab_group_additive"][
                        "manifest_sha256"] !=
                    _sha256(folder / "inference_manifest.json")):
                raise ValueError("MGTAB inference manifest lacks frozen audit")
        if (deployment["status"] != "inference_only_metadata_no_test_results"
                or deployment["variant"] != variant
                or deployment["selection_lock_sha256"] !=
                _sha256(folder / "selection_lock.json")
                or deployment["source_audit_sha256"] != _sha256(
                    self.root / "research/certificates/mgtab_full_source_audit.json")
                or deployment["source_features_sha256"] !=
                self.lock["source_features_sha256"]
                or self.lock["selected_config"] != [31, 400, 50, 10.0]
                or self.lock["test_metric_or_prediction_used_to_create_lock"]
                is not False):
            raise ValueError("MGTAB frozen selection differs")
        self.run_dir = folder / f"seed{seed}"
        inference = deployment["runs"][seed]
        model_path = self.run_dir / "model.joblib"
        reference_path = self.run_dir / "label_blind_reference.json.gz"
        source_metadata_path = (self.root / inference["source_metadata_path"])
        if (inference["seed"] != seed
                or model_path != self.root / inference["model_path"]
                or reference_path != self.root / inference["reference_path"]
                or _sha256(model_path) != inference["model_sha256"]
                or _sha256(reference_path) != inference["reference_sha256"]
                or _sha256(source_metadata_path) !=
                inference["source_metadata_sha256"]
                or inference["split_sha256"] !=
                self.lock["split_sha256_by_seed"][str(seed)]):
            raise ValueError("MGTAB model or reference differs")
        self.model = joblib.load(model_path)
        self.model.set_params(n_jobs=1)
        self.reference = LabelBlindReference.load(reference_path)
        if (self.reference.version != inference["reference_version"] or
                self.reference.version !=
                self.lock["reference_version_by_seed"][str(seed)]):
            raise ValueError("MGTAB evidence reference version differs")
        self.threshold = float(inference["threshold"])
        if self.threshold != float(
                self.lock["threshold_by_seed"][str(seed)]
                if variant == "ab_group_additive" else self.lock["threshold"]):
            raise ValueError("MGTAB dev-selected threshold differs")
        self.data_dir = resolve_mgtab_source_dir(self.root)
        self.structural_path = self.root / "data/processed/mgtab_v3/structural_facts.npz"
        self.neighbor_path = self.root / "data/processed/mgtab_v3/neighbor_facts.npz"
        self.projection_path = self.root / ("output/v3_mgtab_ab_spline_content/"
                                            f"selected/seed{seed}/projection.npz")
        if (_sha256(self.data_dir / "features.pt") !=
                self.lock["source_features_sha256"]
                or (variant == "full_atomic" and
                    (_sha256(self.structural_path) !=
                     self.lock["structural_facts_sha256"] or
                     _sha256(self.neighbor_path) !=
                     self.lock["neighbor_facts_sha256"]))):
            raise ValueError("Current published MGTAB source differs")
        facts, names, ids = load_source_population(
            "mgtab", self.data_dir,
            self.root / f"output/v3_mgtab_ab_spline_content/selected/seed{seed}",
            structural_facts=self.structural_path,
            neighbor_facts=self.neighbor_path,
            inference_metadata=source_metadata_path)
        with np.load(self.neighbor_path, allow_pickle=False) as cache:
            if cache["ids"].astype(str).tolist() != ids:
                raise ValueError("MGTAB neighbor support rows differ")
            supports = cache["supports"].astype(float)
        if facts.shape != (10199, 150) or tuple(names) != self.reference.names:
            raise ValueError("Published fact population differs")
        self.facts, self.names, self.ids = facts, names, ids
        self.supports = supports
        self.index = {uid: i for i, uid in enumerate(ids)}
        if variant == "ab_group_additive":
            contract_path = self.run_dir / "ab_direction_contract.npz"
            if (contract_path != self.root / inference["ab_contract_path"] or
                    _sha256(contract_path) != inference["ab_contract_sha256"]):
                raise ValueError("A/B direction contract differs")
            with np.load(contract_path, allow_pickle=False) as contract:
                signs = contract["monotone_constraints"].astype(int).tolist()
                group_names = contract["source_group_names"].astype(str).tolist()
            if self.model.get_params()["monotone_constraints"] != signs:
                raise ValueError("model monotone directions differ")
            self.group_fact_indices = {
                "published_profile_coordinates": list(range(20)),
                **{f"published_relation_{k}": [
                    j for j in range(20, 86)
                    if names[j].startswith(f"published_{relation}_")]
                    for k, relation in enumerate(AB_RELATIONS)},
                "published_tweet_projections": list(range(86, 150)),
            }
            indices = [j for members in self.group_fact_indices.values()
                       for j in members]
            if (group_names != list(self.group_fact_indices)
                    or sorted(indices) != list(range(150))
                    or [len(members) for members in self.group_fact_indices.values()]
                    != [20, 14, 14, 6, 14, 6, 6, 6, 64]):
                raise ValueError("A/B source group names or partition differ")
            by_feature = {
                feature: group for group, indices in enumerate(
                    self.group_fact_indices.values())
                for j in indices for feature in (j, j + 150)}
            for tree in self.model.booster_.dump_model()["tree_info"]:
                used: set[int] = set()

                def walk(node: dict) -> None:
                    if "split_feature" not in node:
                        return
                    used.add(by_feature[int(node["split_feature"])])
                    walk(node["left_child"])
                    walk(node["right_child"])

                walk(tree["tree_structure"])
                if len(used) > 1:
                    raise ValueError("frozen tree crossed source groups")
        else:
            self.group_fact_indices = {
                group: list(range(group_slice.start, group_slice.stop))
                for group, group_slice in GROUPS.items()}

    def predict(self, account_id: str) -> dict:
        if str(account_id) not in self.index:
            raise KeyError("account row is absent from MGTAB release")
        i = self.index[str(account_id)]
        batch = self.reference.transform(self.facts[i:i + 1])
        reliability = batch.reliability.copy()
        reliability[:, 62:86] *= self.supports[i:i + 1] / (
            self.supports[i:i + 1] + 5.0)
        x = np.column_stack((np.log1p(batch.upper) * reliability,
                             np.log1p(batch.lower) * reliability))
        probability = float(self.model.predict_proba(x)[0, 1])
        logit = float(self.model.booster_.predict(x, raw_score=True)[0])
        shap = np.asarray(self.model.booster_.predict(x, pred_contrib=True))[0]
        if (shap.shape != (301,) or abs(float(shap.sum()) - logit) > 1e-9
                or abs(float(expit(logit)) - probability) > 1e-12):
            raise ValueError("MGTAB evidence contributions do not reconstruct score")
        empty_probability = float(self.model.predict_proba(np.zeros_like(x))[0, 1])
        masks = {}
        group_logit_deltas = []
        for group, indices in self.group_fact_indices.items():
            masked = x.copy()
            masked[:, indices] = 0.0
            masked[:, np.asarray(indices) + 150] = 0.0
            value = float(self.model.predict_proba(masked)[0, 1])
            masked_logit = float(self.model.booster_.predict(
                masked, raw_score=True)[0])
            group_logit_deltas.append(logit - masked_logit)
            masks[group] = {
                "probability_without_group": value,
                "group_logit_difference": logit - masked_logit,
                "class_changes_without_group": bool(
                    (value >= self.threshold) != (probability >= self.threshold)),
            }
        empty_logit = float(self.model.booster_.predict(
            np.zeros_like(x), raw_score=True)[0])
        group_additivity_error = abs(
            logit - empty_logit - sum(group_logit_deltas))
        if self.variant == "ab_group_additive" and group_additivity_error > 1e-9:
            raise ValueError("source-group logit terms do not add up")
        decision_certificate = None
        if self.variant == "ab_group_additive":
            group_names = list(GROUPS)
            subset_logits = []
            for subset in range(8):
                retained = np.zeros_like(x)
                for k, group in enumerate(group_names):
                    if subset & (1 << k):
                        group_slice = GROUPS[group]
                        retained[:, group_slice] = x[:, group_slice]
                        start = group_slice.start + 150
                        stop = group_slice.stop + 150
                        retained[:, start:stop] = x[:, start:stop]
                subset_logits.append(float(self.model.booster_.predict(
                    retained, raw_score=True)[0]))
            if abs(subset_logits[7] - logit) > 1e-9:
                raise ValueError("full source subset differs from prediction")
            subset_probabilities = [float(expit(value))
                                    for value in subset_logits]
            subset_classes = [value >= self.threshold
                              for value in subset_probabilities]
            full_class = subset_classes[7]
            viable = [subset for subset in range(8)
                      if subset_classes[subset] == full_class]
            minimum_size = min(subset.bit_count() for subset in viable)
            minimal = [subset for subset in viable
                       if subset.bit_count() == minimum_size]
            shift = logit - subset_logits[0]
            decision_certificate = {
                "zero_evidence_prediction": int(subset_classes[0]),
                "zero_evidence_probability": subset_probabilities[0],
                "zero_evidence_changes_prediction": bool(
                    subset_classes[0] != full_class),
                "evidence_logit_shift_from_zero": shift,
                "evidence_strengthens_predicted_class": bool(
                    shift > 1e-12 if full_class else shift < -1e-12),
                "pivotal_source_groups": [
                    name for k, name in enumerate(group_names)
                    if subset_classes[7 ^ (1 << k)] != full_class],
                "smallest_sufficient_source_group_sets": [
                    [name for k, name in enumerate(group_names)
                     if subset & (1 << k)] for subset in minimal],
                "source_group_logit_effect_when_removed": {
                    name: logit - subset_logits[7 ^ (1 << k)]
                    for k, name in enumerate(group_names)},
                "without_tweet_projection_probability": subset_probabilities[3],
                "without_tweet_projection_prediction": int(subset_classes[3]),
                "tweet_projection_changes_prediction": bool(
                    subset_classes[3] != full_class),
                "decision_source_scope": (
                    "published_averaged_tweet_tensor_required_for_class"
                    if subset_classes[3] != full_class else
                    "class_reproduced_with_published_profile_and_typed_relations"),
            }
        contributions = shap[:150] + shap[150:300]
        evidence = []
        for position in np.argsort(np.abs(contributions))[::-1]:
            j = int(position)
            raw = self.facts[i, j]
            if j < 20:
                source = {
                    "level": "published_normalized_profile_tensor",
                    "file": str(self.data_dir / "features.pt"),
                    "row": i, "column": j,
                    "paper_field": PROFILE_SCHEMA[j][0],
                }
            elif j < 86:
                source = {
                    "level": "published_typed_relation_tensors_and_derived_fact",
                    "edge_index": str(self.data_dir / "edge_index.pt"),
                    "edge_type": str(self.data_dir / "edge_type.pt"),
                    "fact_cache": (None if j < 34 else
                                   str(self.structural_path) if j < 62 else
                                   str(self.neighbor_path)),
                    "account_row": i,
                }
            else:
                source = {
                    "level": "published_averaged_tweet_tensor_train_only_projection",
                    "file": str(self.data_dir / "features.pt"),
                    "row": i, "input_columns": "20:788",
                    "projection_artifact": str(self.projection_path),
                    "semantic_status": "numeric projection; no individual tweet text",
                }
            evidence.append({
                "fact": self.names[j],
                "raw_value": float(raw) if np.isfinite(raw) else None,
                "upper_tail": float(batch.upper[0, j]),
                "lower_tail": float(batch.lower[0, j]),
                "reliability": float(reliability[0, j]),
                "logit_contribution": float(contributions[j]),
                "source": source,
            })
        return {
            "account_id": str(account_id), "seed": self.seed,
            "variant": self.variant,
            "prediction": int(probability >= self.threshold),
            "probability": probability, "logit": logit,
            "intercept_logit": float(shap[-1]),
            "reference_version": self.reference.version,
            "all_evidence_zero_probability": empty_probability,
            "all_evidence_changes_class": bool(
                (empty_probability >= self.threshold) !=
                (probability >= self.threshold)),
            "group_dependencies": masks,
            "group_logit_additivity_error": group_additivity_error,
            "source_group_decision_certificate": decision_certificate,
            "evidence": evidence,
        }
