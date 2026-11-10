"""Monotone interval-spline evidence reasoner with A/B direction isolation."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .v3_evidence_reasoner import EvidenceBatch, SignConstrainedReasoner


class MonotoneSplineReasoner(SignConstrainedReasoner):
    """Each signed concept has nonnegative slopes on label-blind B quantile intervals."""

    def __init__(self, groups: Mapping[str, Sequence[str]], *, l2: float = 0.01,
                 quantiles: Sequence[float] = (0.5, 0.8)):
        super().__init__(groups, l2=l2)
        self.quantiles = tuple(float(q) for q in quantiles)
        if not self.quantiles or any(not 0 < q < 1 for q in self.quantiles) or \
                list(self.quantiles) != sorted(set(self.quantiles)):
            raise ValueError("quantiles must be distinct, sorted, and strictly between 0 and 1")
        self.knots: dict[str, tuple[float, ...]] | None = None
        self._slices: dict[str, slice] | None = None

    def _magnitude(self, evidence: EvidenceBatch) -> np.ndarray:
        if self.names != evidence.names or self.reference_version != evidence.reference_version:
            raise ValueError("evidence names or reference version differ from reasoner")
        if self.tail is None:
            raise RuntimeError("directions are not fixed")
        selected = np.stack([
            evidence.upper[:, j] if tail == "upper" else evidence.lower[:, j]
            for j, tail in enumerate(self.tail)
        ], axis=1)
        return np.log1p(selected) * evidence.reliability

    def _design(self, evidence: EvidenceBatch) -> np.ndarray:
        if self.knots is None or self.sign is None or self.names is None:
            raise RuntimeError("spline basis is not fitted")
        magnitude = self._magnitude(evidence)
        columns = []
        slices = {}
        for j, name in enumerate(self.names):
            start = len(columns)
            previous = 0.0
            for knot in self.knots[name]:
                columns.append(np.clip(magnitude[:, j] - previous, 0.0,
                                       knot - previous) * self.sign[j])
                previous = knot
            columns.append(np.maximum(magnitude[:, j] - previous, 0.0) * self.sign[j])
            slices[name] = slice(start, len(columns))
        self._slices = slices
        return np.stack(columns, axis=1)

    def fit(self, direction_evidence: EvidenceBatch, direction_labels: np.ndarray,
            fit_evidence: EvidenceBatch, fit_labels: np.ndarray) -> "MonotoneSplineReasoner":
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
        self.tail, self.sign = self._select_direction(direction_evidence, ya)
        magnitudes = self._magnitude(fit_evidence)
        self.knots = {}
        for j, name in enumerate(self.names):
            positive = magnitudes[magnitudes[:, j] > 1e-8, j]
            candidates = np.quantile(positive, self.quantiles) if len(positive) else []
            self.knots[name] = tuple(sorted({float(k) for k in candidates if k > 1e-8}))
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
                          options={"maxiter": 2000, "ftol": 1e-11})
        if not result.success:
            raise RuntimeError(f"spline reasoner optimization failed: {result.message}")
        self.intercept = float(result.x[0])
        self.coef = np.asarray(result.x[1:], dtype=float)
        return self

    def contributions(self, evidence: EvidenceBatch) -> dict[str, np.ndarray]:
        if self.coef is None or self.names is None:
            raise RuntimeError("reasoner must be fitted first")
        z = self._design(evidence)
        by_name = {name: z[:, self._slices[name]] @ self.coef[self._slices[name]]
                   for name in self.names}
        return {group: sum(by_name[name] for name in members)
                for group, members in self.groups.items()}

    def manifest(self) -> dict:
        if self.coef is None or self.sign is None or self.tail is None or \
                self.knots is None or self.names is None or self._slices is None:
            raise RuntimeError("reasoner must be fitted first")
        return {"type": "monotone_interval_spline", "reference_version": self.reference_version,
                "groups": self.groups, "l2": self.l2, "quantiles": self.quantiles,
                "directions": {name: {"tail": tail, "sign": int(sign)}
                               for name, tail, sign in zip(self.names, self.tail, self.sign)},
                "knots": {name: self.knots[name] for name in self.names},
                "coefficients": {name: self.coef[self._slices[name]].tolist()
                                 for name in self.names},
                "intercept": self.intercept}

    @classmethod
    def from_manifest(cls, payload: dict) -> "MonotoneSplineReasoner":
        if payload.get("type") != "monotone_interval_spline":
            raise ValueError("reasoner manifest has an unexpected type")
        result = cls(payload["groups"], l2=float(payload["l2"]),
                     quantiles=payload["quantiles"])
        result.names = tuple(payload["directions"])
        result._validate_groups(result.names)
        result.reference_version = str(payload["reference_version"])
        result.tail = tuple(payload["directions"][name]["tail"] for name in result.names)
        result.sign = np.asarray([payload["directions"][name]["sign"]
                                  for name in result.names], dtype=float)
        result.knots = {name: tuple(map(float, payload["knots"][name]))
                        for name in result.names}
        widths = [len(result.knots[name]) + 1 for name in result.names]
        result.coef = np.asarray([value for name in result.names
                                  for value in payload["coefficients"][name]], dtype=float)
        result.intercept = float(payload["intercept"])
        if len(result.coef) != sum(widths) or np.any(result.coef < 0) or \
                not np.isfinite(result.coef).all() or not np.isfinite(result.intercept) or \
                any(tail not in ("upper", "lower") for tail in result.tail) or \
                not set(result.sign) <= {-1.0, 0.0, 1.0} or \
                any(not all(0 < knot for knot in result.knots[name]) or
                    tuple(sorted(set(result.knots[name]))) != result.knots[name]
                    for name in result.names):
            raise ValueError("invalid spline reasoner manifest")
        return result
