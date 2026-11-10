"""Audit actual published typed-edge deletion in a frozen MGTAB detector.

For each held-out account and relation type, select the smallest destination ID
whose directed edge has exactly one published row. Delete that one row, update
both endpoints' complete graph-derived facts, keep the trained reference and
weights fixed, and measure the class response. No account labels are read.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime
from verifiable_bot.mgtab_full_source_audit import recompute
from verifiable_bot.mgtab_neighbor_facts import PROFILE_INDICES
from verifiable_bot.mgtab_source_certificate import RELATIONS


NEIGHBOR_SLOT = {0: 0, 1: 8, 3: 16}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph_index(edge: np.ndarray, kinds: np.ndarray, features: np.ndarray
                ) -> list[dict]:
    n = len(features)
    profile = features[:, PROFILE_INDICES].astype(np.float64)
    output = []
    for relation in range(len(RELATIONS)):
        rows = np.flatnonzero(kinds == relation)
        src, dst = edge[:, rows]
        row_out = np.bincount(src, minlength=n)
        row_in = np.bincount(dst, minlength=n)
        nonself = src != dst
        keys = src[nonself].astype(np.int64) * n + dst[nonself]
        order = np.argsort(keys, kind="stable")
        unique, start, counts = np.unique(
            keys[order], return_index=True, return_counts=True)
        source, target = unique // n, unique % n
        edge_row = rows[nonself][order[start]]
        reverse = np.isin(unique, target * n + source,
                         assume_unique=True)
        unique_out = np.bincount(source, minlength=n)
        unique_in = np.bincount(target, minlength=n)
        reciprocal_out = np.bincount(source[reverse], minlength=n)
        singleton = counts == 1
        owner, first = np.unique(source[singleton], return_index=True)
        destination = target[singleton][first]
        candidate_row = edge_row[singleton][first]
        candidate_reverse = reverse[singleton][first]
        entry = {
            "row_out": row_out, "row_in": row_in,
            "unique_out": unique_out, "unique_in": unique_in,
            "reciprocal_out": reciprocal_out,
            "candidate_owner": owner, "candidate_destination": destination,
            "candidate_row": candidate_row,
            "candidate_reverse": candidate_reverse,
        }
        if relation in NEIGHBOR_SLOT:
            entry["out_sum"] = np.column_stack([
                np.bincount(source, weights=profile[target, j], minlength=n)
                for j in range(4)])
            entry["in_sum"] = np.column_stack([
                np.bincount(target, weights=profile[source, j], minlength=n)
                for j in range(4)])
        output.append(entry)
    return output


def changed_endpoint(fact: np.ndarray, support: np.ndarray,
                     info: dict, relation: int, account: int,
                     other: int, outbound: bool, reverse: bool,
                     profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fact = fact.copy()
    support = support.copy()
    direction = 0 if outbound else 1
    row_count = int(info["row_out" if outbound else "row_in"][account])
    unique_count = int(info[
        "unique_out" if outbound else "unique_in"][account])
    if row_count < 1 or unique_count < 1:
        raise ValueError("selected source edge not present")
    fact[20 + 2 * relation + direction] = np.log1p(
        np.float32(row_count - 1))
    fact[34 + 4 * relation + direction] = np.log1p(
        np.float32(unique_count - 1))
    if reverse:
        reciprocal = int(info["reciprocal_out"][account])
        if reciprocal < 1:
            raise ValueError("reciprocal source index differs")
        fact[34 + 4 * relation + 2] = np.log1p(
            np.float32(reciprocal - 1))
    if relation in NEIGHBOR_SLOT:
        offset = NEIGHBOR_SLOT[relation] + 4 * direction
        if not np.all(support[offset:offset + 4] == unique_count):
            raise ValueError("published neighbor support count differs")
        if unique_count == 1:
            fact[62 + offset:62 + offset + 4] = np.nan
        else:
            sums = info["out_sum" if outbound else "in_sum"][account]
            fact[62 + offset:62 + offset + 4] = np.asarray(
                (sums - profile[other, PROFILE_INDICES]) /
                (unique_count - 1), dtype=np.float32)
        support[offset:offset + 4] = unique_count - 1
    return fact, support


def encode_probability(runtime: MGTABFullAtomicRuntime,
                       facts: np.ndarray, supports: np.ndarray) -> np.ndarray:
    batch = runtime.reference.transform(facts)
    reliability = batch.reliability.copy()
    reliability[:, 62:86] *= supports / (supports + 5.0)
    model_input = np.column_stack((
        np.log1p(batch.upper) * reliability,
        np.log1p(batch.lower) * reliability))
    if model_input.shape != (len(facts), 300) or not np.isfinite(model_input).all():
        raise ValueError("edited source facts do not produce valid model input")
    return runtime.model.predict_proba(model_input)[:, 1]


def check_full_graph_rebuild(runtime: MGTABFullAtomicRuntime,
                             edge: np.ndarray, kinds: np.ndarray,
                             features: np.ndarray, info: list[dict]) -> list[dict]:
    """Compare local endpoint update with an independent whole-graph rebuild."""
    selected = []
    for relation in range(len(RELATIONS)):
        entry = info[relation]
        selected.append((relation, int(entry["candidate_owner"][0]),
                         int(entry["candidate_destination"][0]),
                         int(entry["candidate_row"][0]),
                         bool(entry["candidate_reverse"][0])))
    for relation, entry in enumerate(info):
        reverse = np.flatnonzero(entry["candidate_reverse"])
        if len(reverse):
            j = int(reverse[0])
            selected.append((relation, int(entry["candidate_owner"][j]),
                             int(entry["candidate_destination"][j]),
                             int(entry["candidate_row"][j]), True))
            break
    checks = []
    for relation, source, target, row, reverse in selected:
        if (tuple(edge[:, row]) != (source, target)
                or int(kinds[row]) != relation):
            raise ValueError("selected edge row differs from published tensor")
        edited_edge = np.delete(edge, row, axis=1)
        edited_type = np.delete(kinds, row)
        base, structural, neighbor, supports, _ = recompute(
            features, edited_edge, edited_type)
        expected = np.column_stack((base, structural, neighbor))
        original = runtime.facts[:, 20:86]
        unaffected = np.ones(len(features), dtype=bool)
        unaffected[[source, target]] = False
        if (not np.array_equal(expected[unaffected, :42],
                               original[unaffected, :42])
                or not np.allclose(expected[unaffected, 42:],
                                   original[unaffected, 42:],
                                   atol=1e-7, rtol=0, equal_nan=True)):
            raise ValueError("deleting one edge changed a non-endpoint fact")
        for account, other, outbound in ((source, target, True),
                                         (target, source, False)):
            fact, support = changed_endpoint(
                runtime.facts[account], runtime.supports[account],
                info[relation], relation, account, other,
                outbound, reverse, features)
            if (not np.array_equal(fact[20:62], expected[account, :42])
                    or not np.allclose(fact[62:86], expected[account, 42:],
                                       atol=1e-7, rtol=0, equal_nan=True)
                    or not np.array_equal(support, supports[account])):
                raise ValueError("local source operation differs from whole graph")
        checks.append({"relation": RELATIONS[relation], "edge_row": row,
                       "source": source, "target": target,
                       "reverse_edge_present": reverse,
                       "all_nonendpoint_facts_unchanged": True,
                       "both_endpoint_facts_and_supports_match": True})
    return checks


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    audit_path = root / "research/certificates/mgtab_full_source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    files = {name: data / name for name in (
        "features.pt", "edge_index.pt", "edge_type.pt")}
    if (audit["status"] != "verified_all_published_tensor_derived_fact_cells"
            or any(sha256(files[name]) != audit["source_sha256"][name]
                   for name in files)):
        raise ValueError("published MGTAB source differs from full audit")
    features = torch.load(files["features.pt"], map_location="cpu",
                          weights_only=True).numpy()
    edge = torch.load(files["edge_index.pt"], map_location="cpu",
                      weights_only=True).numpy()
    kinds = torch.load(files["edge_type.pt"], map_location="cpu",
                       weights_only=True).numpy()
    info = graph_index(edge, kinds, features)
    group_summary_path = (root / "research/certificates/"
                          "mgtab_ab_group_decision_certificate.json")
    group_summary = json.loads(group_summary_path.read_text(encoding="utf-8"))
    if len(group_summary["runs"]) != 3:
        raise ValueError("frozen group certificate population differs")
    records = []
    summaries = []
    full_checks = None
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(root, seed, variant="ab_group_additive")
        saved_path = (root / "output/v3_mgtab_ab_group_additive_tree" /
                      f"seed{seed}/test_evidence.npz")
        with np.load(saved_path, allow_pickle=False) as saved:
            test_ids = saved["ids"].astype(int)
            saved_prob = saved["probability"].copy()
            if (len(test_ids) != 1020 or len(np.unique(test_ids)) != len(test_ids)
                    or not np.allclose(saved["raw_values"],
                                       runtime.facts[test_ids],
                                       atol=0, rtol=0, equal_nan=True)):
                raise ValueError("saved test population or raw facts differ")
        group_run = group_summary["runs"][seed]
        group_path = root / group_run["certificate_file"]
        if (group_run["seed"] != seed or
                sha256(group_path) != group_run["certificate_sha256"]):
            raise ValueError("frozen group-mask certificate differs")
        group_by_id = {int(row["account_id"]): row for row in
                       map(json.loads, group_path.read_text(
                           encoding="utf-8").splitlines())}
        if set(group_by_id) != set(test_ids.tolist()):
            raise ValueError("group-mask and source-edit populations differ")
        if full_checks is None:
            full_checks = check_full_graph_rebuild(
                runtime, edge, kinds, features, info)
        candidates = []
        for relation, entry in enumerate(info):
            present = np.isin(entry["candidate_owner"], test_ids)
            for u, v, row, reverse in zip(
                    entry["candidate_owner"][present],
                    entry["candidate_destination"][present],
                    entry["candidate_row"][present],
                    entry["candidate_reverse"][present]):
                candidates.append((relation, int(u), int(v), int(row),
                                   bool(reverse)))
        affected_ids = np.unique([account for _, u, v, _, _ in candidates
                                  for account in (u, v)])
        original_prob = encode_probability(
            runtime, runtime.facts[affected_ids],
            runtime.supports[affected_ids])
        before = dict(zip(affected_ids.tolist(), original_prob.tolist()))
        if max(abs(before[int(u)] - float(p)) for u, p in
               zip(test_ids, saved_prob) if int(u) in before) > 1e-12:
            raise ValueError("frozen model baseline differs from saved test score")
        modified_facts, modified_supports = [], []
        for relation, u, v, _, reverse in candidates:
            for account, other, outbound in ((u, v, True), (v, u, False)):
                fact, support = changed_endpoint(
                    runtime.facts[account], runtime.supports[account],
                    info[relation], relation, account, other,
                    outbound, reverse, features)
                modified_facts.append(fact)
                modified_supports.append(support)
        after = encode_probability(
            runtime, np.asarray(modified_facts),
            np.asarray(modified_supports))
        counters = {relation: {"selected_source_rows": 0,
                               "target_nonzero_score_response": 0,
                               "target_class_flips": 0,
                               "target_flips_with_group_mask_pivotal": 0,
                               "target_flips_without_group_mask_pivotal": 0,
                               "other_endpoint_class_flips": 0}
                    for relation in RELATIONS}
        for j, (relation, u, v, row, reverse) in enumerate(candidates):
            target_before, other_before = before[u], before[v]
            target_after, other_after = map(float, after[2*j:2*j+2])
            target_flip = ((target_before >= runtime.threshold) !=
                           (target_after >= runtime.threshold))
            other_flip = ((other_before >= runtime.threshold) !=
                          (other_after >= runtime.threshold))
            relation_group_pivotal = (
                "published_native_relations" in
                group_by_id[u]["pivotal_source_groups"])
            summary = counters[RELATIONS[relation]]
            summary["selected_source_rows"] += 1
            summary["target_nonzero_score_response"] += int(
                abs(target_after - target_before) > 1e-12)
            summary["target_class_flips"] += int(target_flip)
            summary["target_flips_with_group_mask_pivotal"] += int(
                target_flip and relation_group_pivotal)
            summary["target_flips_without_group_mask_pivotal"] += int(
                target_flip and not relation_group_pivotal)
            summary["other_endpoint_class_flips"] += int(other_flip)
            records.append({
                "seed": seed, "relation": RELATIONS[relation],
                "source_edge_row": row, "source_account": u,
                "destination_account": v, "reverse_edge_present": reverse,
                "target_before_probability": target_before,
                "target_after_probability": target_after,
                "other_endpoint_before_probability": other_before,
                "other_endpoint_after_probability": other_after,
                "threshold": runtime.threshold,
                "relation_group_pivotal_under_encoding_mask": bool(
                    relation_group_pivotal),
                "target_class_flip": bool(target_flip),
                "other_endpoint_class_flip": bool(other_flip),
            })
        summaries.append({"seed": seed, "test_accounts": len(test_ids),
                          "selected_edits": len(candidates),
                          "group_mask_certificate_sha256": sha256(group_path),
                          "group_mask_pivotal_test_accounts": group_run[
                              "pivotal_source_group_counts"][
                                  "published_native_relations"],
                          "by_relation": counters})
        print(json.dumps({"seed": seed, "selected_edits": len(candidates),
                          "target_flips": sum(v["target_class_flips"]
                                              for v in counters.values())}),
              flush=True)
    detail = root / "research/certificates/mgtab_frozen_typed_edge_source_response.jsonl.gz"
    with gzip.open(detail, "wt", encoding="utf-8", newline="\n") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path = root / "research/certificates/mgtab_frozen_typed_edge_source_response.json"
    result = {
        "scope": "frozen_mgtab_ab_group_additive_one_unique_outgoing_edge_per_test_account_per_relation",
        "source_operation": "delete_exactly_one_published_directed_typed_edge_row_and_rebuild_both_endpoint_facts",
        "selection_rule": "smallest destination ID among unique-row nonself outgoing edges for each held-out account and relation",
        "labels_read_for_selection_or_response": False,
        "weights_and_training_reference_held_fixed": True,
        "source_audit_sha256": sha256(audit_path),
        "group_mask_summary_sha256": sha256(group_summary_path),
        "published_source_sha256": {name: sha256(path)
                                    for name, path in files.items()},
        "whole_graph_rebuild_checks": full_checks,
        "detail_file": str(detail.relative_to(root)),
        "detail_sha256": sha256(detail),
        "runs": summaries,
    }
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2)
                            + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
