"""Source-bound inference for the dev-selected eight-coordinate MGTAB model."""

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MGTABReducedProjectionRuntime:
    """Predict from published tensors, reference, and this model's own weight."""

    def __init__(self, root: Path, seed: int):
        if seed not in (0, 1, 2):
            raise ValueError("unsupported seed")
        self.root = Path(root).resolve()
        self.seed = seed
        folder = self.root / "output/v3_mgtab_reduced_projection_ab_group"
        lock_path = folder / "selection_lock.json"
        manifest_path = folder / "inference_manifest.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parent_path = (self.root /
                       "output/v3_mgtab_ab_group_additive_tree/inference_manifest.json")
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        source_audit_path = self.root / "research/certificates/mgtab_full_source_audit.json"
        source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
        if (manifest["status"] != "inference_only_metadata_no_test_results"
                or manifest["variant"] != "reduced_projection_ab_group_additive"
                or manifest["selection_lock_sha256"] != sha256_file(lock_path)
                or manifest["parent_source_inference_manifest_sha256"] !=
                sha256_file(parent_path)
                or lock["parent_selection_lock_sha256"] !=
                parent["selection_lock_sha256"]
                or manifest["source_audit_sha256"] !=
                sha256_file(source_audit_path)
                or manifest["source_audit_sha256"] !=
                parent["source_audit_sha256"]
                or source_audit["status"] !=
                "verified_all_published_tensor_derived_fact_cells"
                or manifest["selected_projection_dimensions"] != 8
                or lock["selected_projection_dimensions"] != 8
                or manifest["projection_output_absolute_logit_cap"] != 3.0):
            raise ValueError("MGTAB inference selection differs")
        self.dimension = 8
        self.cap = 3.0
        self.threshold = float(lock["threshold_by_seed"][str(seed)])
        run = manifest["runs"][seed]
        parent_run = parent["runs"][seed]
        if (run["seed"] != seed or run["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]
                or run["threshold"] != self.threshold):
            raise ValueError("MGTAB seed manifest differs")
        model_path = self.root / run["model_path"]
        reference_path = self.root / run["reference_path"]
        metadata_path = self.root / run["source_metadata_path"]
        contract_path = self.root / run["ab_contract_path"]
        for path, field in ((model_path, "model_sha256"),
                            (reference_path, "reference_sha256"),
                            (metadata_path, "source_metadata_sha256"),
                            (contract_path, "ab_contract_sha256")):
            if sha256_file(path) != run[field]:
                raise ValueError(f"MGTAB inference artifact differs: {field}")
        if (run["reference_sha256"] != parent_run["reference_sha256"]
                or run["source_metadata_sha256"] !=
                parent_run["source_metadata_sha256"]
                or run["ab_contract_sha256"] !=
                lock["a_b_contract_sha256_by_seed"][str(seed)]):
            raise ValueError("MGTAB source reference or A/B contract differs")
        self.model = joblib.load(model_path)
        self.model.set_params(n_jobs=1)
        self.reference = LabelBlindReference.load(reference_path)
        if self.reference.version != run["reference_version"]:
            raise ValueError("MGTAB label-blind reference differs")
        self.data_dir = resolve_mgtab_source_dir(self.root)
        if sha256_file(self.data_dir / "features.pt") != manifest["source_features_sha256"]:
            raise ValueError("MGTAB released feature tensor differs")
        self.structural_path = self.root / "data/processed/mgtab_v3/structural_facts.npz"
        self.neighbor_path = self.root / "data/processed/mgtab_v3/neighbor_facts.npz"
        self.projection_path = (self.root /
            f"output/v3_mgtab_ab_spline_content/selected/seed{seed}/projection.npz")
        facts, names, ids = load_source_population(
            "mgtab", self.data_dir, self.projection_path.parent,
            structural_facts=self.structural_path,
            neighbor_facts=self.neighbor_path,
            inference_metadata=metadata_path)
        if (facts.shape != (10199, 150)
                or tuple(names) != self.reference.names):
            raise ValueError("MGTAB released fact population differs")
        with np.load(self.neighbor_path, allow_pickle=False) as cache:
            if cache["ids"].astype(str).tolist() != ids:
                raise ValueError("MGTAB neighbor support IDs differ")
            supports = cache["supports"].astype(float)
        self.facts, self.names, self.ids = facts, names, ids
        self.index = {uid: i for i, uid in enumerate(ids)}
        self.supports = supports
        self.group_indices = {
            "published_profile_coordinates": list(range(20)),
            **{f"published_relation_{j}": [
                i for i in range(20, 86)
                if names[i].startswith(f"published_{relation}_")]
               for j, relation in enumerate(("followers", "friends",
                    "mention", "reply", "quoted", "url", "hashtag"))},
            "published_tweet_projections": list(range(86, 94)),
        }
        if sorted(i for group in self.group_indices.values() for i in group) != list(range(94)):
            raise ValueError("Reduced source groups do not partition facts")
        with np.load(contract_path, allow_pickle=False) as contract:
            signs = contract["monotone_constraints"].astype(int)
        reduced_signs = signs[np.r_[np.arange(94), np.arange(150, 244)]].tolist()
        if self.model.get_params()["monotone_constraints"] != reduced_signs:
            raise ValueError("MGTAB A/B monotone directions differ")
        by_feature = {j: name for name, members in self.group_indices.items()
                      for i in members for j in (i, i + 94)}
        for tree in self.model.booster_.dump_model()["tree_info"]:
            used: set[str] = set()
            def walk(node: dict) -> None:
                if "split_feature" not in node:
                    return
                used.add(by_feature[int(node["split_feature"])])
                walk(node["left_child"])
                walk(node["right_child"])
            walk(tree["tree_structure"])
            if len(used) > 1:
                raise ValueError("MGTAB fitted tree crosses source groups")

    def evidence_matrix(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=int)
        batch = self.reference.transform(self.facts[rows])
        reliability = batch.reliability.copy()
        reliability[:, 62:86] *= self.supports[rows] / (self.supports[rows] + 5.0)
        encoded = np.column_stack((np.log1p(batch.upper) * reliability,
                                   np.log1p(batch.lower) * reliability))
        reduced = encoded[:, np.r_[np.arange(94), np.arange(150, 244)]]
        if reduced.shape != (len(rows), 188) or not np.isfinite(reduced).all():
            raise ValueError("MGTAB encoded evidence differs")
        return reduced

    def predict_rows(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        x = self.evidence_matrix(rows)
        full = np.asarray(self.model.booster_.predict(x, raw_score=True), dtype=float)
        named_x = x.copy()
        named_x[:, 86:94] = 0.0
        named_x[:, 180:188] = 0.0
        named = np.asarray(self.model.booster_.predict(
            named_x, raw_score=True), dtype=float)
        raw_residual = full - named
        bounded_residual = np.clip(raw_residual, -self.cap, self.cap)
        logit = named + bounded_residual
        return {"logit": logit, "probability": expit(logit),
                "named_logit": named, "projection_raw_logit": raw_residual,
                "projection_bounded_logit": bounded_residual}

    def predict(self, account_id: str) -> dict:
        if str(account_id) not in self.index:
            raise KeyError("Account row is absent from MGTAB release")
        i = self.index[str(account_id)]
        x = self.evidence_matrix(np.asarray([i]))
        scores = self.predict_rows(np.asarray([i]))
        named_x = x.copy()
        named_x[:, 86:94] = 0.0
        named_x[:, 180:188] = 0.0
        booster = self.model.booster_
        zero = float(booster.predict(np.zeros_like(x), raw_score=True)[0])
        named_logit = float(scores["named_logit"][0])
        groups = []
        named_effect_sum = 0.0
        for name, members in self.group_indices.items():
            if name == "published_tweet_projections":
                effect = float(scores["projection_bounded_logit"][0])
            else:
                masked = named_x.copy()
                masked[:, members] = 0.0
                masked[:, np.asarray(members) + 94] = 0.0
                effect = named_logit - float(booster.predict(
                    masked, raw_score=True)[0])
                named_effect_sum += effect
            facts = []
            for j in members:
                if j < 20:
                    source = {"level": "published_normalized_profile_tensor",
                              "file": str(self.data_dir / "features.pt"),
                              "row": i, "column": j,
                              "paper_field": PROFILE_SCHEMA[j][0]}
                elif j < 86:
                    source = {"level": "published_typed_relation_tensors_and_derived_fact",
                              "edge_index": str(self.data_dir / "edge_index.pt"),
                              "edge_type": str(self.data_dir / "edge_type.pt"),
                              "fact_cache": (None if j < 34 else
                                  str(self.structural_path) if j < 62 else
                                  str(self.neighbor_path)), "account_row": i}
                else:
                    source = {"level": "published_averaged_tweet_tensor_train_only_projection",
                              "file": str(self.data_dir / "features.pt"),
                              "row": i, "input_columns": "20:788",
                              "projection_artifact": str(self.projection_path),
                              "semantic_status": "numeric projection; no individual tweet text"}
                value = self.facts[i, j]
                facts.append({"name": self.names[j],
                              "raw_value": float(value) if np.isfinite(value) else None,
                              "source": source})
            groups.append({"name": name, "logit_effect": effect,
                           "source_facts": facts})
        if (abs(zero + named_effect_sum - named_logit) > 1e-9
                or abs(zero + sum(group["logit_effect"] for group in groups)
                       - float(scores["logit"][0])) > 1e-9):
            raise ValueError("Source-group effects do not reconstruct decision")
        projection_effect = float(scores["projection_bounded_logit"][0])
        total_absolute_effect = sum(abs(group["logit_effect"])
                                    for group in groups)
        projection_absolute_share = (
            abs(projection_effect) / total_absolute_effect
            if total_absolute_effect > 0 else 0.0)
        named_only_probability = float(expit(named_logit))
        return {"account_id": str(account_id), "seed": self.seed,
                "prediction": int(scores["probability"][0] >= self.threshold),
                "probability": float(scores["probability"][0]),
                "logit": float(scores["logit"][0]),
                "threshold": self.threshold,
                "reference_version": self.reference.version,
                "zero_evidence_logit": zero,
                "group_logit_additivity_error": abs(zero + sum(
                    group["logit_effect"] for group in groups) -
                    float(scores["logit"][0])),
                "projection_raw_logit": float(scores["projection_raw_logit"][0]),
                "projection_bounded_logit": projection_effect,
                "projection_absolute_effect_share": projection_absolute_share,
                "named_only_probability": named_only_probability,
                "named_only_prediction": int(named_only_probability >= self.threshold),
                "projection_changes_prediction": bool(
                    (named_only_probability >= self.threshold) !=
                    (scores["probability"][0] >= self.threshold)),
                "groups": groups}
