"""Label-free MGTAB neighbor profile means from published typed edges."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .mgtab_profile_schema import PROFILE_SCHEMA, PAPER_TABLE, PAPER_URL
from .mgtab_source_certificate import RELATIONS, sha256
from .v3_evidence_reasoner import EvidenceBatch, LabelBlindReference


# Fixed before reading Bot/Human labels: two social relations and the reply
# relation, with four paper-named profile dimensions for each edge direction.
RELATION_INDICES = (0, 1, 3)
PROFILE_INDICES = (1, 2, 13, 14)
SUPPORT_PRIOR = 5.0


def evidence_with_neighbor_support(reference: LabelBlindReference, x: np.ndarray,
                                   supports: np.ndarray | None) -> EvidenceBatch:
    """Transform facts and shrink neighbor concepts at columns 62:86 by support."""
    evidence = reference.transform(x)
    if supports is None:
        return evidence
    if supports.shape != (len(x), 24) or len(evidence.names) < 86 or \
            np.any(supports < 0):
        raise ValueError("MGTAB neighbor supports differ from evidence population")
    reliability = evidence.reliability.copy()
    reliability[:, 62:86] *= supports / (supports + SUPPORT_PRIOR)
    return EvidenceBatch(evidence.upper, evidence.lower, reliability,
                         evidence.available, evidence.reference_version,
                         evidence.names)


def extract(features: np.ndarray, edge_index: np.ndarray, edge_type: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    profile = np.asarray(features, dtype=np.float32)
    edge = np.asarray(edge_index, dtype=np.int64)
    kind = np.asarray(edge_type, dtype=np.int64)
    n = len(profile)
    if profile.ndim != 2 or profile.shape[1] < len(PROFILE_SCHEMA) or \
            not np.isfinite(profile[:, PROFILE_INDICES]).all():
        raise ValueError("published profile coordinates are unavailable")
    if edge.ndim != 2 or edge.shape[0] != 2 or edge.shape[1] != len(kind) or \
            np.any(edge < 0) or np.any(edge >= n):
        raise ValueError("published edge dimensions or endpoints differ")
    if not np.array_equal(np.unique(kind), np.arange(len(RELATIONS))):
        raise ValueError("published relation vocabulary differs")

    values, supports, names = [], [], []
    summary = {}
    for relation_index in RELATION_INDICES:
        relation = RELATIONS[relation_index]
        src, dst = edge[:, kind == relation_index]
        nonself = src != dst
        keys = np.unique(src[nonself] * n + dst[nonself])
        source, target = keys // n, keys % n
        relation_summary = {"unique_nonself_directed_pairs": int(len(keys))}
        for direction, owner, neighbor in (("out", source, target),
                                           ("in", target, source)):
            count = np.bincount(owner, minlength=n).astype(np.int32)
            relation_summary[f"accounts_with_{direction}_neighbors"] = int(
                np.count_nonzero(count))
            for profile_index in PROFILE_INDICES:
                feature_name = PROFILE_SCHEMA[profile_index][0]
                sums = np.bincount(owner,
                                   weights=profile[neighbor, profile_index],
                                   minlength=n)
                mean = np.full(n, np.nan, dtype=np.float32)
                present = count > 0
                mean[present] = (sums[present] / count[present]).astype(np.float32)
                values.append(mean)
                supports.append(count)
                names.append(f"published_{relation}_{direction}_neighbor_"
                             f"{feature_name}_mean")
        summary[relation] = relation_summary
    return (np.stack(values, axis=1), np.stack(supports, axis=1),
            names, summary)


def build(data_dir: Path, output: Path) -> dict:
    feature_path = data_dir / "features.pt"
    edge_path = data_dir / "edge_index.pt"
    type_path = data_dir / "edge_type.pt"
    features = torch.load(feature_path, map_location="cpu", weights_only=True).numpy()
    edge = torch.load(edge_path, map_location="cpu", weights_only=True).numpy()
    kind = torch.load(type_path, map_location="cpu", weights_only=True).numpy()
    values, supports, names, summary = extract(features, edge, kind)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=np.arange(len(features)),
                        names=np.asarray(names), values=values, supports=supports)
    manifest = {
        "dataset": "mgtab", "account_count": len(features),
        "feature_names": names, "relation_indices": list(RELATION_INDICES),
        "profile_indices": list(PROFILE_INDICES),
        "profile_paper_url": PAPER_URL, "profile_paper_table": PAPER_TABLE,
        "source_level": "published_profile_tensor_and_typed_edge_tensor",
        "source_operation": "mean of named normalized profile coordinate over unique non-self directed neighbors; missing when no neighbors",
        "support_definition": "number of unique non-self directed neighbors",
        "recommended_support_prior": SUPPORT_PRIOR,
        "label_access": "none", "relation_summary": summary,
        "input_sha256": {path.name: sha256(path)
                         for path in (feature_path, edge_path, type_path)},
        "facts_sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    return manifest
