"""Additive classifier with separately signed high- and low-tail evidence."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .v3_evidence_reasoner import EvidenceBatch, SignConstrainedReasoner


class DualTailEvidenceReasoner(SignConstrainedReasoner):
    """A fixes each tail's class direction; B fits nonnegative weights."""

    def __init__(self, groups: Mapping[str, Sequence[str]], *, l2: float = 0.01):
        super().__init__(groups, l2=l2)
        self.tail_sign: np.ndarray | None = None

    @staticmethod
    def _select_tail_signs(evidence: EvidenceBatch, labels: np.ndarray) -> np.ndarray:
        if set(labels) != {0, 1}:
            raise ValueError("direction split must contain both classes")
        sign = np.zeros((len(evidence.names), 2), dtype=float)
        for tail_index, raw in enumerate((evidence.upper, evidence.lower)):
            magnitude = np.log1p(raw) * evidence.reliability
            difference = np.mean(magnitude[labels == 1], axis=0) - np.mean(
                magnitude[labels == 0], axis=0)
            sign[:, tail_index] = np.sign(difference)
        return sign

    def _design(self, evidence: EvidenceBatch) -> np.ndarray:
        if self.names != evidence.names or self.reference_version != evidence.reference_version:
            raise ValueError("evidence names or reference version differ from reasoner")
        if self.tail_sign is None:
            raise RuntimeError("tail directions are not fixed")
        upper = np.log1p(evidence.upper) * evidence.reliability * self.tail_sign[:, 0]
        lower = np.log1p(evidence.lower) * evidence.reliability * self.tail_sign[:, 1]
        return np.column_stack((upper, lower))

    def fit(self, direction_evidence: EvidenceBatch, direction_labels: np.ndarray,
            fit_evidence: EvidenceBatch, fit_labels: np.ndarray) -> "DualTailEvidenceReasoner":
        if direction_evidence.names != fit_evidence.names or \
                direction_evidence.reference_version != fit_evidence.reference_version:
            raise ValueError("A and B evidence must share one reference and vocabulary")
        self.names = direction_evidence.names
        self.reference_version = direction_evidence.reference_version
        self._validate_groups(self.names)
        ya = np.asarray(direction_labels, dtype=int)
        yb = np.asarray(fit_labels, dtype=int)
        if len(ya) != len(direction_evidence.upper) or len(yb) != len(fit_evidence.upper):
            raise ValueError("labels and evidence rows differ")
        if set(ya) != {0, 1} or set(yb) != {0, 1}:
            raise ValueError("A and B must each contain both classes")
        self.tail_sign = self._select_tail_signs(direction_evidence, ya)
        z = self._design(fit_evidence)

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            intercept, coef = params[0], params[1:]
            logit = intercept + z @ coef
            loss = float(np.mean(np.logaddexp(0.0, logit) - yb * logit) +
                         0.5 * self.l2 * np.sum(coef ** 2))
            residual = expit(logit) - yb
            gradient = np.r_[np.mean(residual), z.T @ residual / len(yb) + self.l2 * coef]
            return loss, gradient

        prior = float(np.clip(np.mean(yb), 1e-5, 1 - 1e-5))
        initial = np.r_[np.log(prior / (1 - prior)), np.zeros(z.shape[1])]
        result = minimize(objective, initial, method="L-BFGS-B", jac=True,
                          bounds=[(None, None), *[(0.0, None)] * z.shape[1]],
                          options={"maxiter": 1000, "ftol": 1e-11})
        if not result.success:
            raise RuntimeError(f"dual-tail optimization failed: {result.message}")
        self.intercept = float(result.x[0])
        self.coef = np.asarray(result.x[1:], dtype=float)
        return self

    def contributions(self, evidence: EvidenceBatch) -> dict[str, np.ndarray]:
        if self.coef is None or self.names is None:
            raise RuntimeError("reasoner must be fitted first")
        z = self._design(evidence)
        width = len(self.names)
        by_name = {name: z[:, j] * self.coef[j] +
                   z[:, width + j] * self.coef[width + j]
                   for j, name in enumerate(self.names)}
        return {group: sum(by_name[name] for name in members)
                for group, members in self.groups.items()}

    def manifest(self) -> dict:
        if self.coef is None or self.tail_sign is None or self.names is None:
            raise RuntimeError("reasoner must be fitted first")
        width = len(self.names)
        return {
            "type": "dual_tail_sign_constrained_additive",
            "reference_version": self.reference_version,
            "groups": self.groups,
            "l2": self.l2,
            "directions": {name: {"upper": int(self.tail_sign[j, 0]),
                                  "lower": int(self.tail_sign[j, 1])}
                           for j, name in enumerate(self.names)},
            "coefficients": {name: {"upper": float(self.coef[j]),
                                    "lower": float(self.coef[width + j])}
                             for j, name in enumerate(self.names)},
            "intercept": self.intercept,
        }

    @classmethod
    def from_manifest(cls, payload: dict) -> "DualTailEvidenceReasoner":
        if payload.get("type") != "dual_tail_sign_constrained_additive":
            raise ValueError("unexpected reasoner type")
        result = cls(payload["groups"], l2=float(payload["l2"]))
        result.names = tuple(payload["directions"])
        result._validate_groups(result.names)
        if set(payload["coefficients"]) != set(result.names):
            raise ValueError("coefficient vocabulary differs")
        result.reference_version = str(payload["reference_version"])
        result.tail_sign = np.asarray([
            [payload["directions"][name][tail] for tail in ("upper", "lower")]
            for name in result.names], dtype=float)
        result.coef = np.asarray([
            payload["coefficients"][name][tail]
            for tail in ("upper", "lower") for name in result.names], dtype=float)
        result.intercept = float(payload["intercept"])
        if result.tail_sign.shape != (len(result.names), 2) or \
                not np.isfinite(result.tail_sign).all() or \
                not set(result.tail_sign.flat) <= {-1.0, 0.0, 1.0} or \
                not np.isfinite(result.coef).all() or np.any(result.coef < 0) or \
                not np.isfinite(result.intercept):
            raise ValueError("invalid dual-tail reasoner manifest")
        return result
