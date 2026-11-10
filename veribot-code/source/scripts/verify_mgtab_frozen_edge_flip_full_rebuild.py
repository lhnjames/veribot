"""Independently rebuild the complete MGTAB graph for every reported edge flip."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime
from verifiable_bot.mgtab_full_source_audit import recompute
from verifiable_bot.mgtab_source_certificate import RELATIONS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_probability(runtime: MGTABFullAtomicRuntime,
                       fact: np.ndarray, support: np.ndarray) -> tuple[float, float]:
    batch = runtime.reference.transform(fact.reshape(1, -1))
    reliability = batch.reliability.copy()
    reliability[:, 62:86] *= support.reshape(1, -1) / (
        support.reshape(1, -1) + 5.0)
    encoded = np.column_stack((np.log1p(batch.upper) * reliability,
                               np.log1p(batch.lower) * reliability))
    if encoded.shape != (1, 300) or not np.isfinite(encoded).all():
        raise ValueError("independent full rebuild produced invalid model input")
    raw_logit = float(runtime.model.booster_.predict(
        encoded, raw_score=True)[0])
    return float(expit(raw_logit)), raw_logit


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    prior_path = root / "research/certificates/mgtab_frozen_typed_edge_source_response.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    detail_path = root / prior["detail_file"]
    if sha256(detail_path) != prior["detail_sha256"]:
        raise ValueError("source response detail changed")
    candidates = []
    with gzip.open(detail_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item["target_class_flip"] or item["other_endpoint_class_flip"]:
                candidates.append(item)
    if len(candidates) != 18:
        raise ValueError("reported target or collateral flip count changed")
    data = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    source_paths = {name: data / name for name in (
        "features.pt", "edge_index.pt", "edge_type.pt")}
    if any(sha256(path) != prior["published_source_sha256"][name]
           for name, path in source_paths.items()):
        raise ValueError("published graph or profile tensor changed")
    features = torch.load(source_paths["features.pt"], map_location="cpu",
                          weights_only=True).numpy()
    edge = torch.load(source_paths["edge_index.pt"], map_location="cpu",
                      weights_only=True).numpy()
    kinds = torch.load(source_paths["edge_type.pt"], map_location="cpu",
                       weights_only=True).numpy()
    runtimes = {seed: MGTABFullAtomicRuntime(
        root, seed, variant="ab_group_additive") for seed in (0, 1, 2)}
    result_rows = []
    max_probability_difference = 0.0
    for index, record in enumerate(candidates, start=1):
        seed = int(record["seed"])
        source = int(record["source_account"])
        target = int(record["destination_account"])
        row = int(record["source_edge_row"])
        relation = RELATIONS.index(record["relation"])
        if (tuple(edge[:, row]) != (source, target)
                or int(kinds[row]) != relation):
            raise ValueError("reported source row differs from published graph")
        edited_edge = np.delete(edge, row, axis=1)
        edited_kind = np.delete(kinds, row)
        base, structure, neighbor, supports, _ = recompute(
            features, edited_edge, edited_kind)
        graph_facts = np.column_stack((base, structure, neighbor))
        runtime = runtimes[seed]
        original = runtime.facts[:, 20:86]
        unaffected = np.ones(len(features), dtype=bool)
        unaffected[[source, target]] = False
        if (not np.array_equal(graph_facts[unaffected, :42],
                               original[unaffected, :42])
                or not np.allclose(graph_facts[unaffected, 42:],
                                   original[unaffected, 42:],
                                   atol=1e-7, rtol=0, equal_nan=True)
                or not np.array_equal(supports[unaffected],
                                      runtime.supports[unaffected])):
            raise ValueError("unexpected graph fact or support collateral")
        checked = {}
        for name, account in (("target", source), ("other_endpoint", target)):
            before_probability, before_logit = frozen_probability(
                runtime, runtime.facts[account], runtime.supports[account])
            modified = runtime.facts[account].copy()
            modified[20:86] = graph_facts[account]
            after_probability, after_logit = frozen_probability(
                runtime, modified, supports[account])
            prior_before = float(record[f"{name}_before_probability"])
            prior_after = float(record[f"{name}_after_probability"])
            delta = max(abs(before_probability - prior_before),
                        abs(after_probability - prior_after))
            max_probability_difference = max(max_probability_difference, delta)
            if delta > 1e-12:
                raise ValueError(f"full graph prediction differs from local edit: {delta}")
            changed_class = ((before_probability >= runtime.threshold) !=
                             (after_probability >= runtime.threshold))
            if changed_class != record[f"{name}_class_flip"]:
                raise ValueError("source edit class flip differs")
            checked[name] = {
                "account": account,
                "before_probability": before_probability,
                "after_probability": after_probability,
                "before_logit": before_logit,
                "after_logit": after_logit,
                "class_flip": bool(changed_class),
                "minimum_threshold_margin": float(min(
                    abs(before_probability - runtime.threshold),
                    abs(after_probability - runtime.threshold))),
            }
        result_rows.append({
            "seed": seed, "source_edge_row": row,
            "relation": RELATIONS[relation], "source_account": source,
            "destination_account": target,
            "nonendpoint_facts_and_supports_unchanged": True,
            "target": checked["target"],
            "other_endpoint": checked["other_endpoint"],
            "group_mask_pivotal_for_target": record[
                "relation_group_pivotal_under_encoding_mask"],
        })
        print(json.dumps({"checked": index, "seed": seed,
                          "edge_row": row, "target_flip": checked[
                              "target"]["class_flip"]}), flush=True)
    output = {
        "scope": "all_reported_mgtab_typed_edge_source_flip_events",
        "verification_method": "delete_published_edge_row_recompute_all_graph_facts_and_supports_reencode_same_frozen_model",
        "account_labels_read": False,
        "prior_summary_sha256": sha256(prior_path),
        "prior_detail_sha256": sha256(detail_path),
        "source_sha256": prior["published_source_sha256"],
        "verified_source_edits": len(result_rows),
        "verified_target_class_flips": sum(
            r["target"]["class_flip"] for r in result_rows),
        "verified_other_endpoint_class_flips": sum(
            r["other_endpoint"]["class_flip"] for r in result_rows),
        "max_probability_difference_from_local_audit": max_probability_difference,
        "rows": result_rows,
    }
    target_path = root / "research/certificates/mgtab_edge_flip_full_rebuild_verification.json"
    target_path.write_text(json.dumps(output, ensure_ascii=False, indent=2)
                           + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
