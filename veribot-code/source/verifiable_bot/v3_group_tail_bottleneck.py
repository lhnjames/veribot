"""Label-blind, source-addressable high/low tail concepts per fact group."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from .v3_evidence_reasoner import EvidenceBatch


def collapse_group_tails(
    atomic: EvidenceBatch,
    groups: Mapping[str, Sequence[str]],
) -> tuple[EvidenceBatch, dict[str, np.ndarray]]:
    """Return two scalar concepts per group and the selected atomic source index.

    Each concept is the largest reliability-weighted tail score among a
    prespecified group's available facts. The source index identifies the
    winning atomic fact for every account; ties use vocabulary order. No labels
    or class predictions enter this operation.
    """
    vocabulary = tuple(atomic.names)
    listed = [name for members in groups.values() for name in members]
    if (not groups or any(not members for members in groups.values())
            or len(listed) != len(vocabulary)
            or len(set(listed)) != len(vocabulary)
            or set(listed) != set(vocabulary)):
        raise ValueError("groups must partition every atomic fact exactly once")
    if len(groups) < 1 or len(groups) > 10:
        raise ValueError("group-tail bottleneck requires 2-20 scalar concepts")
    name_to_index = {name: j for j, name in enumerate(vocabulary)}
    columns = []
    available = []
    concept_names = []
    winners = {}
    for group, members in groups.items():
        indices = np.asarray([name_to_index[name] for name in members], dtype=int)
        present = np.any(atomic.available[:, indices], axis=1)
        for tail, values in (("high", atomic.upper), ("low", atomic.lower)):
            signal = np.log1p(values[:, indices]) * atomic.reliability[:, indices]
            signal = np.where(atomic.available[:, indices], signal, -np.inf)
            local = np.argmax(signal, axis=1)
            score = np.where(present, signal[np.arange(len(local)), local], 0.0)
            name = f"{group}:{tail}_tail_max"
            concept_names.append(name)
            columns.append(score)
            available.append(present)
            winners[name] = np.where(present, indices[local], -1)
    matrix = np.column_stack(columns)
    mask = np.column_stack(available)
    payload = {"atomic_reference_version": atomic.reference_version,
               "groups": {str(group): list(members) for group, members in groups.items()},
               "formula": "max_available(log1p(atomic_tail_score)*atomic_reliability)",
               "concept_names": concept_names}
    version = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
    # Both slots carry the same already-directional concept value. The
    # downstream A/B reasoner learns only its class-support sign and weight.
    result = EvidenceBatch(matrix, matrix.copy(), np.ones_like(matrix), mask,
                           version, tuple(concept_names))
    return result, winners
