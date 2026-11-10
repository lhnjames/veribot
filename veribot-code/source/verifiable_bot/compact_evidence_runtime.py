"""Inference-only runtime for the selected source-to-evidence classifiers.

The runtime accepts an ordered matrix of source facts. It has no label input,
does not fit parameters, and exposes the exact additive terms used to predict.
Raw-record adapters and source-location audits remain separate components.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from .v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from .v3_group_tail_bottleneck import collapse_group_tails
from .v3_monotone_spline_reasoner import MonotoneSplineReasoner


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CompactEvidenceRuntime:
    """Load one frozen seed and predict entirely through its evidence bottleneck."""

    def __init__(self, dataset: str, selected_dir: Path, full_dir: Path, seed: int,
                 atomic_groups_path: Path | None = None,
                 deployment_metadata: Path | None = None):
        if dataset not in {"botsim", "fox8", "mgtab"} or seed not in (0, 1, 2):
            raise ValueError("unsupported selected model or seed")
        self.dataset = dataset
        self.seed = seed
        self.selected_dir = selected_dir / f"seed{seed}"
        self.full_dir = full_dir / f"seed{seed}"
        if deployment_metadata is None:
            self.report = json.loads((self.selected_dir / "report.json").read_text(encoding="utf-8"))
            full_report = json.loads((self.full_dir / "report.json").read_text(encoding="utf-8"))
        else:
            metadata = json.loads(deployment_metadata.read_text(encoding="utf-8"))
            self.report = metadata["selected"]
            full_report = metadata["full"]
        manifest = self.report["manifest"]
        full_manifest = full_report["manifest"]
        reference_path = self.full_dir / "reference.json.gz"
        if (manifest["dataset"] != dataset or manifest["seed"] != seed
                or manifest["split_sha256"] != full_manifest["split_sha256"]
                or manifest["reference_version"] != full_manifest["reference_version"]
                or _sha256(reference_path) != full_manifest["reference_artifact_sha256"]):
            raise ValueError("selected model differs from its fixed source/reference")
        self.reference = LabelBlindReference.load(reference_path)
        if self.reference.version != manifest["reference_version"]:
            raise ValueError("reference version differs")
        self.reasoner = MonotoneSplineReasoner.from_manifest(self.report["reasoner"])
        self.threshold = float(manifest["threshold"])
        if not 0 < self.threshold < 1 or manifest["concept_count"] != len(self.reasoner.names):
            raise ValueError("invalid frozen decision threshold or concept count")
        if dataset == "mgtab":
            self.selected_indices = [int(i) for i in manifest["selected_fact_indices"]]
            self.atomic_groups = None
            if ([self.reference.names[i] for i in self.selected_indices]
                    != manifest["selected_fact_names"]
                    or tuple(manifest["selected_fact_names"]) != self.reasoner.names
                    or self.reasoner.reference_version != self.reference.version):
                raise ValueError("MGTAB selected vocabulary differs")
            # Selected MGTAB facts are profile/relation facts or content PCA
            # coordinates. None uses the neighbor support correction at 62:86.
            if any(62 <= i < 86 for i in self.selected_indices):
                raise ValueError("runtime needs neighbor support for selected facts")
        else:
            self.selected_indices = None
            if dataset == "fox8":
                if atomic_groups_path is None:
                    raise ValueError("Fox8 requires its fixed 20-concept atomic groups")
                lock = json.loads((selected_dir / "selection_lock.json").read_text(
                    encoding="utf-8"))
                if _sha256(atomic_groups_path) != lock["prespecified_groups_sha256"]:
                    raise ValueError("Fox8 atomic groups differ from validation lock")
                self.atomic_groups = json.loads(atomic_groups_path.read_text(encoding="utf-8"))
            else:
                self.atomic_groups = full_report["reasoner"]["groups"]
            if (tuple(manifest["concept_names"]) != self.reasoner.names
                    or manifest["concept_version"] != self.reasoner.reference_version):
                raise ValueError("group-tail concept vocabulary differs")

    def _evidence(self, facts: np.ndarray, fact_names: list[str] | tuple[str, ...]
                  ) -> tuple[EvidenceBatch, dict[str, np.ndarray]]:
        if tuple(fact_names) != self.reference.names:
            raise ValueError("source fact vocabulary or order differs from frozen reference")
        values = np.asarray(facts, dtype=float)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(self.reference.names):
            raise ValueError("source fact matrix differs from frozen reference")
        atomic = self.reference.transform(values)
        if self.dataset == "mgtab":
            indices = self.selected_indices
            evidence = EvidenceBatch(
                atomic.upper[:, indices], atomic.lower[:, indices],
                atomic.reliability[:, indices], atomic.available[:, indices],
                atomic.reference_version,
                tuple(atomic.names[i] for i in indices),
            )
            winners = {name: np.full(len(values), indices[j], dtype=int)
                       for j, name in enumerate(evidence.names)}
        else:
            evidence, winners = collapse_group_tails(atomic, self.atomic_groups)
        if (evidence.names != self.reasoner.names
                or evidence.reference_version != self.reasoner.reference_version):
            raise ValueError("evidence batch differs from fixed reasoner")
        return evidence, winners

    def predict(self, facts: np.ndarray, fact_names: list[str] | tuple[str, ...],
                account_ids: list[str] | None = None) -> list[dict]:
        facts = np.asarray(facts, dtype=float)
        evidence, winners = self._evidence(facts, fact_names)
        if account_ids is None:
            account_ids = [str(i) for i in range(len(evidence.upper))]
        if len(account_ids) != len(evidence.upper) or len(set(account_ids)) != len(account_ids):
            raise ValueError("account IDs must be unique and align with fact rows")
        probability = self.reasoner.predict_proba(evidence)
        basis = self.reasoner._design(evidence)
        contribution = np.column_stack([
            basis[:, self.reasoner._slices[name]] @
            self.reasoner.coef[self.reasoner._slices[name]]
            for name in evidence.names
        ])
        logit = self.reasoner.intercept + contribution.sum(axis=1)
        if np.max(np.abs(expit(logit) - probability)) > 1e-12:
            raise ValueError("evidence terms do not reconstruct model output")
        empty = EvidenceBatch(np.zeros_like(evidence.upper), np.zeros_like(evidence.lower),
                              evidence.reliability, evidence.available,
                              evidence.reference_version, evidence.names)
        if np.max(np.abs(self.reasoner.predict_proba(empty) -
                         expit(self.reasoner.intercept))) > 1e-12:
            raise ValueError("non-evidence path affects the frozen prediction")
        results = []
        for row, account_id in enumerate(account_ids):
            terms = []
            for j, name in enumerate(evidence.names):
                source_index = int(winners[name][row])
                selected_tail = self.reasoner.tail[j]
                selected_value = (evidence.upper[row, j] if selected_tail == "upper"
                                  else evidence.lower[row, j])
                terms.append({
                    "name": name,
                    "source_fact_name": (self.reference.names[source_index]
                                         if source_index >= 0 else None),
                    "source_fact_value": (float(facts[row, source_index])
                                          if source_index >= 0 and
                                          np.isfinite(facts[row, source_index]) else None),
                    "available": bool(evidence.available[row, j]),
                    "evidence_upper": float(evidence.upper[row, j]),
                    "evidence_lower": float(evidence.lower[row, j]),
                    "evidence_reliability": float(evidence.reliability[row, j]),
                    "selected_tail": selected_tail,
                    "model_input_magnitude": float(
                        np.log1p(selected_value) * evidence.reliability[row, j]),
                    "logit_contribution": float(contribution[row, j]),
                })
            results.append({
                "dataset": self.dataset, "seed": self.seed,
                "account_id": str(account_id),
                "reference_version": evidence.reference_version,
                "atomic_reference_version": self.reference.version,
                "intercept": float(self.reasoner.intercept),
                "terms": terms,
                "logit": float(logit[row]),
                "probability": float(probability[row]),
                "threshold": self.threshold,
                "prediction": int(probability[row] >= self.threshold),
            })
        return results
