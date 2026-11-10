"""Label-blind empirical evidence and a sign-constrained additive classifier.

The reference population sees values, availability, and prespecified context
only. Labels enter later: split A fixes each concept's direction and class
support, while disjoint split B fits nonnegative additive coefficients. There
is no raw-feature or hidden-vector bypass in the classifier interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class EvidenceBatch:
    upper: np.ndarray
    lower: np.ndarray
    reliability: np.ndarray
    available: np.ndarray
    reference_version: str
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        shape = self.upper.shape
        if any(value.shape != shape for value in (self.lower, self.reliability,
                                                   self.available)):
            raise ValueError("evidence arrays have inconsistent shapes")
        if len(shape) != 2 or shape[1] != len(self.names):
            raise ValueError("evidence names do not match the feature dimension")
        if not np.isfinite(self.upper).all() or not np.isfinite(self.lower).all():
            raise ValueError("evidence scores must be finite")
        if not np.isfinite(self.reliability).all() or \
                np.any((self.reliability < 0) | (self.reliability > 1)):
            raise ValueError("reliability must lie in [0,1]")


class LabelBlindReference:
    """Empirical tail scores relative to the training population mixture."""

    def __init__(self, *, min_context_n: int = 30, support_prior: float = 0.0):
        if min_context_n < 1 or support_prior < 0:
            raise ValueError("invalid reference parameters")
        self.min_context_n = min_context_n
        self.support_prior = support_prior
        self.names: tuple[str, ...] | None = None
        self.reference: dict[tuple[str, int], np.ndarray] = {}
        self.missing_rate: np.ndarray | None = None
        self.version: str | None = None

    def fit(self, values: np.ndarray, names: Sequence[str],
            contexts: Sequence[str] | None = None) -> "LabelBlindReference":
        raw = np.asarray(values, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != len(names) or raw.shape[0] == 0:
            raise ValueError("invalid reference values or names")
        if len(set(names)) != len(names):
            raise ValueError("concept names must be unique")
        group = np.asarray(["all"] * len(raw) if contexts is None else contexts, dtype=str)
        if len(group) != len(raw):
            raise ValueError("context count differs from account count")
        self.names = tuple(names)
        self.reference = {}
        self.missing_rate = np.mean(~np.isfinite(raw), axis=0)
        for j in range(raw.shape[1]):
            pooled = np.sort(raw[np.isfinite(raw[:, j]), j])
            if not len(pooled):
                raise ValueError(f"concept {names[j]} has no observed training values")
            self.reference[("all", j)] = pooled
            for context in np.unique(group):
                selected = raw[(group == context) & np.isfinite(raw[:, j]), j]
                if len(selected) >= self.min_context_n:
                    self.reference[(str(context), j)] = np.sort(selected)
        self.version = self._digest(self._payload())
        return self

    @staticmethod
    def _digest(payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()

    def _payload(self) -> dict:
        if self.names is None or self.missing_rate is None:
            raise RuntimeError("reference must be fitted first")
        return {"names": self.names, "min_context_n": self.min_context_n,
                "support_prior": self.support_prior,
                "missing_rate": self.missing_rate.tolist(),
                "reference": {f"{context}:{j}": array.tolist()
                              for (context, j), array in sorted(self.reference.items())}}

    def save(self, path: Path) -> None:
        if self.version is None:
            raise RuntimeError("reference must be fitted first")
        payload = self._payload()
        if self._digest(payload) != self.version:
            raise RuntimeError("reference changed after fitting")
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps({"version": self.version, "payload": payload},
                             separators=(",", ":")).encode("utf-8")
        with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as stream:
            stream.write(content)

    @classmethod
    def load(cls, path: Path) -> "LabelBlindReference":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            saved = json.load(stream)
        payload = saved["payload"]
        if cls._digest(payload) != saved["version"]:
            raise ValueError("reference artifact checksum mismatch")
        result = cls(min_context_n=int(payload["min_context_n"]),
                     support_prior=float(payload["support_prior"]))
        result.names = tuple(payload["names"])
        result.missing_rate = np.asarray(payload["missing_rate"], dtype=float)
        result.reference = {(key.rsplit(":", 1)[0], int(key.rsplit(":", 1)[1])):
                            np.asarray(value, dtype=float)
                            for key, value in payload["reference"].items()}
        result.version = saved["version"]
        if len(result.missing_rate) != len(result.names) or \
                any(("all", j) not in result.reference for j in range(len(result.names))):
            raise ValueError("reference artifact lacks required fields")
        return result

    def transform(self, values: np.ndarray, contexts: Sequence[str] | None = None,
                  support_counts: np.ndarray | None = None) -> EvidenceBatch:
        if self.names is None or self.version is None or self.missing_rate is None:
            raise RuntimeError("reference must be fitted first")
        raw = np.asarray(values, dtype=float)
        if raw.ndim != 2 or raw.shape[1] != len(self.names):
            raise ValueError("concept dimension differs from reference")
        group = np.asarray(["all"] * len(raw) if contexts is None else contexts, dtype=str)
        if len(group) != len(raw):
            raise ValueError("context count differs from account count")
        support = None if support_counts is None else np.asarray(support_counts, dtype=float)
        if support is not None and (support.shape != raw.shape or np.any(support < 0)):
            raise ValueError("invalid support counts")
        available = np.isfinite(raw)
        upper = np.zeros(raw.shape, dtype=float)
        lower = np.zeros(raw.shape, dtype=float)
        reliability = np.zeros(raw.shape, dtype=float)
        for j in range(raw.shape[1]):
            for context in np.unique(group):
                mask = (group == context) & available[:, j]
                if not np.any(mask):
                    continue
                ref = self.reference.get((str(context), j), self.reference[("all", j)])
                observed = raw[mask, j]
                low_count = np.searchsorted(ref, observed, side="right")
                high_count = len(ref) - np.searchsorted(ref, observed, side="left")
                lower[mask, j] = -np.log((low_count + 1) / (len(ref) + 1))
                upper[mask, j] = -np.log((high_count + 1) / (len(ref) + 1))
                base = np.sqrt(len(ref) / (len(ref) + self.min_context_n))
                base *= 1.0 - float(self.missing_rate[j])
                if support is not None and self.support_prior:
                    base = base * support[mask, j] / (support[mask, j] + self.support_prior)
                reliability[mask, j] = base
        return EvidenceBatch(upper, lower, reliability, available,
                             self.version, self.names)


class SignConstrainedReasoner:
    """Fit nonnegative concept weights after fixing directions on split A."""

    def __init__(self, groups: Mapping[str, Sequence[str]], *, l2: float = 0.01):
        if l2 < 0 or not groups:
            raise ValueError("groups and regularization must be valid")
        self.groups = {str(key): tuple(value) for key, value in groups.items()}
        self.l2 = float(l2)
        self.names: tuple[str, ...] | None = None
        self.reference_version: str | None = None
        self.tail: tuple[str, ...] | None = None
        self.sign: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.intercept: float | None = None

    def _validate_groups(self, names: tuple[str, ...]) -> None:
        listed = [name for group in self.groups.values() for name in group]
        if len(listed) != len(names) or len(set(listed)) != len(names) or set(listed) != set(names):
            raise ValueError("each evidence concept must appear in exactly one group")

    @staticmethod
    def _select_direction(evidence: EvidenceBatch, y: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
        if len(set(y)) != 2 or set(y) != {0, 1}:
            raise ValueError("direction split must contain both Bot and Human labels")
        tails = []
        signs = []
        for j in range(len(evidence.names)):
            options = []
            for tail, data in (("upper", evidence.upper), ("lower", evidence.lower)):
                observed = data[:, j] * evidence.reliability[:, j]
                bot = observed[y == 1]
                human = observed[y == 0]
                delta = float(np.mean(bot) - np.mean(human))
                scale = float(np.sqrt((np.var(bot) + np.var(human)) / 2))
                options.append((abs(delta) / max(scale, 1e-8), tail, delta))
            _, tail, delta = max(options, key=lambda item: item[0])
            tails.append(tail)
            signs.append(int(np.sign(delta)))
        return tuple(tails), np.asarray(signs, dtype=float)

    def _design(self, evidence: EvidenceBatch) -> np.ndarray:
        if self.names != evidence.names or self.reference_version != evidence.reference_version:
            raise ValueError("evidence names or reference version differ from fitted reasoner")
        if self.tail is None or self.sign is None:
            raise RuntimeError("directions not fixed")
        columns = [evidence.upper[:, j] if choice == "upper" else evidence.lower[:, j]
                   for j, choice in enumerate(self.tail)]
        value = np.stack(columns, axis=1)
        return np.log1p(value) * evidence.reliability * self.sign

    def fit(self, direction_evidence: EvidenceBatch, direction_labels: np.ndarray,
            fit_evidence: EvidenceBatch, fit_labels: np.ndarray) -> "SignConstrainedReasoner":
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
        self.tail, self.sign = self._select_direction(direction_evidence, ya)
        z = self._design(fit_evidence)
        if len(set(yb)) != 2:
            raise ValueError("coefficient split must contain both labels")

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            intercept, coef = params[0], params[1:]
            logit = intercept + z @ coef
            loss = float(np.mean(np.logaddexp(0.0, logit) - yb * logit) +
                         0.5 * self.l2 * np.sum(coef ** 2))
            residual = expit(logit) - yb
            grad = np.r_[np.mean(residual), z.T @ residual / len(yb) + self.l2 * coef]
            return loss, grad

        prior = float(np.clip(np.mean(yb), 1e-5, 1 - 1e-5))
        initial = np.r_[np.log(prior / (1 - prior)), np.zeros(z.shape[1])]
        result = minimize(objective, initial, method="L-BFGS-B", jac=True,
                          bounds=[(None, None), *[(0.0, None)] * z.shape[1]],
                          options={"maxiter": 1000, "ftol": 1e-11})
        if not result.success:
            raise RuntimeError(f"reasoner optimization failed: {result.message}")
        self.intercept = float(result.x[0])
        self.coef = np.asarray(result.x[1:], dtype=float)
        return self

    def contributions(self, evidence: EvidenceBatch) -> dict[str, np.ndarray]:
        if self.coef is None:
            raise RuntimeError("reasoner must be fitted first")
        z = self._design(evidence)
        by_name = {name: z[:, j] * self.coef[j] for j, name in enumerate(self.names or ())}
        return {group: np.sum([by_name[name] for name in members], axis=0)
                for group, members in self.groups.items()}

    def logits(self, evidence: EvidenceBatch) -> np.ndarray:
        if self.intercept is None:
            raise RuntimeError("reasoner must be fitted first")
        return self.intercept + sum(self.contributions(evidence).values())

    def predict_proba(self, evidence: EvidenceBatch) -> np.ndarray:
        return expit(self.logits(evidence))

    def manifest(self) -> dict:
        if self.coef is None or self.sign is None or self.tail is None:
            raise RuntimeError("reasoner must be fitted first")
        return {"reference_version": self.reference_version, "groups": self.groups,
                "directions": {name: {"tail": tail, "sign": int(sign)}
                               for name, tail, sign in zip(self.names or (), self.tail, self.sign)},
                "coefficients": {name: float(value)
                                 for name, value in zip(self.names or (), self.coef)},
                "intercept": self.intercept, "l2": self.l2}

    @classmethod
    def from_manifest(cls, payload: dict) -> "SignConstrainedReasoner":
        result = cls(payload["groups"], l2=float(payload["l2"]))
        result.names = tuple(payload["directions"])
        result._validate_groups(result.names)
        result.reference_version = str(payload["reference_version"])
        result.tail = tuple(payload["directions"][name]["tail"] for name in result.names)
        result.sign = np.asarray([payload["directions"][name]["sign"]
                                  for name in result.names], dtype=float)
        result.coef = np.asarray([payload["coefficients"][name]
                                  for name in result.names], dtype=float)
        result.intercept = float(payload["intercept"])
        if any(tail not in ("upper", "lower") for tail in result.tail) or \
                not set(result.sign) <= {-1.0, 0.0, 1.0} or \
                not np.isfinite(result.coef).all() or np.any(result.coef < 0) or \
                not np.isfinite(result.intercept):
            raise ValueError("invalid reasoner manifest")
        return result
