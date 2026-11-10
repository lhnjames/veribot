"""Exact encoded-fact withholding for the frozen Fox8 and MGTAB models.

This measures a fixed model's response to zeroing one two-tail evidence input.
It is a model-input intervention, not a source edit or a causal claim about a
social account's behavior.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from .fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime
from .mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


Runtime = Fox8NeighborAtomicRuntime | MGTABFullAtomicRuntime


def evidence_design(runtime: Runtime, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=int)
    batch = runtime.reference.transform(runtime.facts[rows])
    reliability = batch.reliability.copy()
    if isinstance(runtime, Fox8NeighborAtomicRuntime):
        reliability[:, 43:74] *= runtime.supports[rows] / (
            runtime.supports[rows] + 5.0)
    else:
        reliability[:, 62:86] *= runtime.supports[rows] / (
            runtime.supports[rows] + 5.0)
    matrix = np.column_stack((np.log1p(batch.upper) * reliability,
                              np.log1p(batch.lower) * reliability))
    if (matrix.shape != (len(rows), 2 * len(runtime.reference.names)) or
            not np.isfinite(matrix).all()):
        raise ValueError("frozen evidence matrix differs")
    return matrix


def withhold_all_facts(runtime: Runtime, rows: np.ndarray,
                       *, fact_batch_size: int = 10) -> dict[str, np.ndarray]:
    """Return exact full and one-fact-withheld scores for the requested rows."""
    if fact_batch_size < 1:
        raise ValueError("fact_batch_size must be positive")
    matrix = evidence_design(runtime, rows)
    n, width = matrix.shape
    d = width // 2
    full_logit = np.asarray(runtime.model.booster_.predict(
        matrix, raw_score=True), dtype=float)
    full_probability = expit(full_logit)
    direct_probability = runtime.model.predict_proba(matrix)[:, 1]
    if np.max(np.abs(full_probability - direct_probability)) > 1e-12:
        raise ValueError("frozen probability does not match its raw score")
    without_logit = np.empty((n, d), dtype=float)
    for start in range(0, d, fact_batch_size):
        selected = list(range(start, min(start + fact_batch_size, d)))
        altered = np.tile(matrix, (len(selected), 1))
        for position, j in enumerate(selected):
            block = slice(position * n, (position + 1) * n)
            altered[block, j] = 0.0
            altered[block, j + d] = 0.0
        raw = np.asarray(runtime.model.booster_.predict(
            altered, raw_score=True), dtype=float)
        without_logit[:, selected] = raw.reshape(len(selected), n).T
    without_probability = expit(without_logit)
    encoded_nonzero = (matrix[:, :d] != 0) | (matrix[:, d:] != 0)
    if np.any(np.abs(without_logit[~encoded_nonzero] -
                     np.broadcast_to(full_logit[:, None],
                                     without_logit.shape)[~encoded_nonzero])
              > 1e-9):
        raise ValueError("zero encoded fact changed frozen logit")
    return {
        "full_logit": full_logit,
        "full_probability": full_probability,
        "without_logit": without_logit,
        "without_probability": without_probability,
        "encoded_nonzero": encoded_nonzero,
    }


def account_fact_effects(runtime: Runtime, account_id: str) -> dict:
    """Attach source-linked exact effects to one live account decision."""
    i = runtime.index[str(account_id)]
    effect = withhold_all_facts(runtime, np.asarray([i]))
    decision = runtime.predict(str(account_id))
    names = list(runtime.reference.names)
    terms = decision["evidence"]
    by_name = {term["fact"]: term for term in terms}
    if (len(by_name) != len(names) or
            abs(float(effect["full_probability"][0]) -
                float(decision["probability"])) > 1e-12):
        raise ValueError("live report and fact effects differ")
    result = []
    for j, name in enumerate(names):
        without_p = float(effect["without_probability"][0, j])
        without_logit = float(effect["without_logit"][0, j])
        delta = float(effect["full_logit"][0] - without_logit)
        class_change = (without_p >= runtime.threshold) != bool(
            decision["prediction"])
        result.append({
            "fact": name,
            "source": (by_name[name].get("source") or
                       by_name[name].get("source_locator")),
            "encoded_evidence_nonzero": bool(effect["encoded_nonzero"][0, j]),
            "probability_without_fact": without_p,
            "logit_difference_when_withheld": delta,
            "class_changes_when_withheld": bool(class_change),
            "role": ("class_critical" if class_change else
                     "score_relevant" if abs(delta) > 1e-10 else
                     "context_only"),
        })
    return {
        "account_id": str(account_id), "seed": runtime.seed,
        "probability": float(decision["probability"]),
        "prediction": int(decision["prediction"]),
        "threshold": runtime.threshold,
        "intervention": "zero both encoded tails of one fact in the frozen classifier; source records and graph stay fixed",
        "fact_effects": result,
    }
