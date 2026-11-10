"""Sign-constrained additive evidence with within-group same-direction pairs."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .v3_evidence_reasoner import EvidenceBatch, SignConstrainedReasoner


class PairwiseEvidenceReasoner(SignConstrainedReasoner):
    """Add nonnegative conjunction weights without breaking fixed directions.

    For a fixed pair whose A-split signs agree, the extra signed basis is
    sign_j * min(|z_j|, |z_k|), where z denotes the singleton design. Pairs
    whose signs disagree or are zero contribute exactly zero. Nonnegative
    coefficients preserve each concept's selected directional monotonicity.
    """

    def __init__(self, groups: Mapping[str, Sequence[str]],
                 pairs: Sequence[tuple[str, str]], *, l2: float = 0.01):
        super().__init__(groups, l2=l2)
        normalized = [tuple(map(str, pair)) for pair in pairs]
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in normalized) or \
                len({frozenset(pair) for pair in normalized}) != len(normalized):
            raise ValueError("evidence pairs must be distinct unordered feature pairs")
        self.pairs = tuple(normalized)
        self.pair_indices: tuple[tuple[int, int, str], ...] = ()

    def _bind_pairs(self, names: tuple[str, ...]) -> None:
        owner = {name: group for group, members in self.groups.items()
                 for name in members}
        if set(owner) != set(names):
            raise ValueError("group vocabulary differs from evidence names")
        index = {name: i for i, name in enumerate(names)}
        bound = []
        for left, right in self.pairs:
            if left not in index or right not in index or owner[left] != owner[right]:
                raise ValueError("pair features must be in the same known evidence group")
            bound.append((index[left], index[right], owner[left]))
        self.pair_indices = tuple(bound)

    def fit(self, direction_evidence: EvidenceBatch, direction_labels: np.ndarray,
            fit_evidence: EvidenceBatch, fit_labels: np.ndarray
            ) -> "PairwiseEvidenceReasoner":
        self._bind_pairs(direction_evidence.names)
        super().fit(direction_evidence, direction_labels, fit_evidence, fit_labels)
        return self

    def _design(self, evidence: EvidenceBatch) -> np.ndarray:
        singles = super()._design(evidence)
        if self.sign is None:
            raise RuntimeError("evidence directions are unavailable")
        extra = []
        for left, right, _ in self.pair_indices:
            common = self.sign[left] if self.sign[left] == self.sign[right] else 0.0
            extra.append(common * np.minimum(np.abs(singles[:, left]),
                                              np.abs(singles[:, right])))
        return np.column_stack((singles, *extra)) if extra else singles

    def contributions(self, evidence: EvidenceBatch) -> dict[str, np.ndarray]:
        if self.coef is None or self.names is None:
            raise RuntimeError("reasoner must be fitted first")
        design = self._design(evidence)
        if design.shape[1] != len(self.coef):
            raise ValueError("pairwise design and coefficients differ")
        result = {group: np.zeros(len(design), dtype=float) for group in self.groups}
        owner = {name: group for group, members in self.groups.items()
                 for name in members}
        for index, name in enumerate(self.names):
            result[owner[name]] += design[:, index] * self.coef[index]
        for pair_index, (_, _, group) in enumerate(self.pair_indices):
            column = len(self.names) + pair_index
            result[group] += design[:, column] * self.coef[column]
        return result

    def pair_contributions(self, evidence: EvidenceBatch) -> dict[str, np.ndarray]:
        """Return every fixed pair's signed logit contribution by stable ID."""
        if self.coef is None or self.names is None:
            raise RuntimeError("reasoner must be fitted first")
        design = self._design(evidence)
        start = len(self.names)
        if design.shape[1] != start + len(self.pairs) or \
                len(self.coef) != design.shape[1]:
            raise ValueError("pairwise design and coefficients differ")
        return {f"pair_{index:02d}": design[:, start + index] *
                self.coef[start + index]
                for index in range(len(self.pairs))}

    def manifest(self) -> dict:
        if self.coef is None or self.names is None or self.sign is None or \
                self.tail is None:
            raise RuntimeError("reasoner must be fitted first")
        single_count = len(self.names)
        return {
            "type": "pairwise_same_direction_minimum",
            "reference_version": self.reference_version,
            "groups": self.groups, "pairs": [list(pair) for pair in self.pairs],
            "directions": {name: {"tail": tail, "sign": int(sign)}
                           for name, tail, sign in zip(self.names, self.tail, self.sign)},
            "single_coefficients": {name: float(value)
                                    for name, value in zip(self.names,
                                                           self.coef[:single_count])},
            "pair_coefficients": [float(x) for x in self.coef[single_count:]],
            "intercept": self.intercept, "l2": self.l2,
        }

    @classmethod
    def from_manifest(cls, payload: dict) -> "PairwiseEvidenceReasoner":
        if payload.get("type") != "pairwise_same_direction_minimum":
            raise ValueError("unknown pairwise reasoner type")
        result = cls(payload["groups"], [tuple(pair) for pair in payload["pairs"]],
                     l2=float(payload["l2"]))
        result.names = tuple(payload["directions"])
        result._validate_groups(result.names)
        result._bind_pairs(result.names)
        result.reference_version = str(payload["reference_version"])
        result.tail = tuple(payload["directions"][name]["tail"]
                            for name in result.names)
        result.sign = np.asarray([payload["directions"][name]["sign"]
                                  for name in result.names], dtype=float)
        result.coef = np.asarray(
            [payload["single_coefficients"][name] for name in result.names] +
            payload["pair_coefficients"], dtype=float)
        result.intercept = float(payload["intercept"])
        if len(result.coef) != len(result.names) + len(result.pairs) or \
                not np.isfinite(result.coef).all() or np.any(result.coef < 0) or \
                not np.isfinite(result.intercept) or \
                any(tail not in ("upper", "lower") for tail in result.tail) or \
                not set(result.sign) <= {-1.0, 0.0, 1.0}:
            raise ValueError("invalid pairwise reasoner manifest")
        return result
