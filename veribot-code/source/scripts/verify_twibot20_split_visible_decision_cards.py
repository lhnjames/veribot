"""Independently verify split-visible TwiBot-20 dev cards and reject mutations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import ijson
from lightgbm import Booster
import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_compact20_capacity import GROUPS
from screen_twibot20_compact20_ab_group_tree_dev import source_groups
from screen_twibot20_compact20_ab_support_dev import evidence_subset
from train_v3_twibot20 import sha256
from verifiable_bot.twibot20_full_source_audit import direct_facts
from verifiable_bot.twibot20_facts import FACT_NAMES
from verifiable_bot.twibot20_source_certificate import direct_query
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


GRAPH = ("graph_out_degree", "graph_in_degree", "graph_reciprocal_degree")
OFFICIAL_ZIP_SHA256 = "fae4387b5c0f835edba36a12b4367566b3df52b46c76b2274da33b5deea8e216"
EPS = 1e-12


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def equal(actual: float, expected: float, field: str) -> None:
    if not np.isfinite(actual) or abs(float(actual) - float(expected)) > EPS:
        raise ValueError(f"Card differs: {field}")


def independent_source(archive: Path, dev_ids: list[str], dev_positions: list[int],
                       expected_values: np.ndarray, all_names: list[str]) -> tuple[list[dict], dict]:
    """Open only train/dev/support; derive raw rows and incident declarations."""
    dev_set = set(dev_ids)
    rows: list[dict] = []
    declarations: dict[tuple[str, str], list[str]] = defaultdict(list)
    with zipfile.ZipFile(archive) as zipped:
        for part in ("train", "dev", "support"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for ordinal, row in enumerate(ijson.items(stream, "item")):
                    owner = str(row["ID"])
                    if part == "dev":
                        k = len(rows)
                        if (k >= len(dev_ids) or owner != dev_ids[k]
                                or ordinal != dev_positions[k]):
                            raise ValueError(f"Raw dev row order differs: {ordinal}")
                        actual = direct_facts(row)
                        if not np.allclose(actual, expected_values[k, :len(actual)],
                                           rtol=0, atol=EPS, equal_nan=True):
                            raise ValueError(f"Raw direct facts differ: {owner}")
                        rows.append(row)
                    neighbor = row.get("neighbor") or {}
                    for field, reverse in (("following", False), ("follower", True)):
                        for slot, other in enumerate(neighbor.get(field) or []):
                            other = str(other)
                            if other == owner:
                                continue
                            edge = (other, owner) if reverse else (owner, other)
                            if edge[0] in dev_set or edge[1] in dev_set:
                                declarations[edge].append(
                                    f"Twibot-20/{part}.json[{ordinal}].neighbor.{field}[{slot}]")
    if len(rows) != len(dev_ids):
        raise ValueError("Raw dev population differs")
    incident: dict[str, list] = defaultdict(list)
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    for (source, target), locators in declarations.items():
        item = [source, target, sorted(locators)]
        if source in dev_set:
            incident[source].append(item)
            outgoing[source].add(target)
        if target in dev_set:
            incident[target].append(item)
            incoming[target].add(source)
    graph = {}
    for uid in dev_ids:
        entries = sorted(incident[uid])
        graph[uid] = {
            "query": "deduplicated directed following/follower declarations in official train/dev/support",
            "incident_edges": len(entries),
            "source_declarations": sum(len(item[2]) for item in entries),
            "incident_declarations_sha256": digest_json(entries),
            "out_degree": len(outgoing[uid]),
            "in_degree": len(incoming[uid]),
            "reciprocal_degree": len(outgoing[uid] & incoming[uid]),
        }
    return rows, graph


def validate_cards(cards: dict, rows: list[dict], graph: dict, ids: list[str],
                   positions: list[int], values: np.ndarray, all_names: list[str],
                   names: list[str], batch, signs: np.ndarray, model: Booster,
                   expected_model_hash: str, expected_threshold: float) -> dict:
    entries = cards["decision_cards"]
    if len(entries) != len(ids):
        raise ValueError("Card population differs")
    groups = dict(zip(GROUPS, source_groups(names), strict=True))
    n = len(names)
    x = np.column_stack((np.log1p(batch.upper) * batch.reliability,
                         np.log1p(batch.lower) * batch.reliability)) * (signs != 0)
    raw = np.asarray(model.predict(x, raw_score=True))
    zero = np.asarray(model.predict(np.zeros_like(x), raw_score=True))
    group_logits = {}
    for group, cols in groups.items():
        masked = x.copy()
        masked[:, cols] = 0
        group_logits[group] = np.asarray(model.predict(masked, raw_score=True))
    threshold = float(cards["decision_cards"][0]["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Invalid threshold")
    equal(threshold, expected_threshold, "screen_threshold")
    if cards["reference_version_with_support_rule"] != batch.reference_version:
        raise ValueError("Reference version differs")
    labels = []
    for k, card in enumerate(entries):
        uid = ids[k]
        row = rows[k]
        if (card["account_id"] != uid
                or card["official_source_locator"] != f"Twibot-20/dev.json[{positions[k]}]"
                or card["source_record_without_label_sha256"] != digest_json(
                    {key: value for key, value in row.items() if key != "label"})
                or card["model_sha256"] != expected_model_hash
                or card["reference_version"] != batch.reference_version):
            raise ValueError(f"Card identity/source binding differs: {uid}")
        if card["tweet_sample_support_count"] != int(values[k, all_names.index("tweet_sample_count")]):
            raise ValueError(f"Tweet support differs: {uid}")
        terms = card["fact_terms"]
        if len(terms) != n or [term["fact_name"] for term in terms] != names:
            raise ValueError(f"Card fact order differs: {uid}")
        for j, term in enumerate(terms):
            name = names[j]
            fact = values[k, all_names.index(name)]
            expected_query = (graph[uid] if name in GRAPH else
                              direct_query("dev", positions[k], name, row))
            if term["source_query"] != expected_query:
                raise ValueError(f"Source query differs: {uid} {name}")
            if name in GRAPH:
                equal(fact, graph[uid][name.removeprefix("graph_")],
                      f"raw_graph_degree {uid} {name}")
            if (term["missing"] != bool(np.isnan(fact))
                    or (term["source_value"] is None) != bool(np.isnan(fact))):
                raise ValueError(f"Source availability differs: {uid} {name}")
            if np.isfinite(fact):
                equal(term["source_value"], fact, f"source_value {uid} {name}")
            for key, expected in (("upper_tail_score", batch.upper[k, j]),
                                  ("lower_tail_score", batch.lower[k, j]),
                                  ("reliability", batch.reliability[k, j]),
                                  ("upper_model_input", x[k, j]),
                                  ("lower_model_input", x[k, j + n])):
                equal(term[key], expected, f"{key} {uid} {name}")
            if (term["upper_monotone_direction"] != int(signs[j])
                    or term["lower_monotone_direction"] != int(signs[j + n])):
                raise ValueError(f"A-split direction differs: {uid} {name}")
        equal(card["threshold"], threshold, f"threshold {uid}")
        equal(card["threshold_logit"], logit(threshold), f"threshold_logit {uid}")
        equal(card["all_zero_input_logit"], zero[k], f"zero_logit {uid}")
        equal(card["baseline_margin"], zero[k] - logit(threshold), f"baseline_margin {uid}")
        equal(card["full_logit"], raw[k], f"full_logit {uid}")
        equal(card["full_probability"], expit(raw[k]), f"probability {uid}")
        equal(card["full_margin"], raw[k] - logit(threshold), f"full_margin {uid}")
        if (set(card["group_logit_effects"]) != set(groups)
                or set(card["group_zeroed_logits"]) != set(groups)):
            raise ValueError(f"Group fields differ: {uid}")
        for group in groups:
            equal(card["group_zeroed_logits"][group], group_logits[group][k],
                  f"group_zeroed_logits {uid} {group}")
            equal(card["group_logit_effects"][group], raw[k] - group_logits[group][k],
                  f"group_logit_effects {uid} {group}")
        prediction = int(expit(raw[k]) >= threshold)
        if card["prediction"] != prediction:
            raise ValueError(f"Prediction differs: {uid}")
        # Labels are used only to verify the saved evaluation fields, never to form evidence.
        if (card["evaluation_label"] != int(row["label"])
                or card["prediction_correct"] != (prediction == card["evaluation_label"])):
            raise ValueError(f"Evaluation fields differ: {uid}")
        labels.append(card["evaluation_label"])
    if (cards["dev_n"] != len(ids)
            or cards["correct_cards"] != sum(c["prediction_correct"] for c in entries)
            or cards["wrong_cards"] != sum(not c["prediction_correct"] for c in entries)):
        raise ValueError("Card aggregate differs")
    return {"verified_dev_cards": len(entries), "verified_fact_terms": len(entries) * n,
            "verified_group_responses": len(entries) * len(groups),
            "correct_cards": cards["correct_cards"], "wrong_cards": cards["wrong_cards"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--raw-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    replay = json.loads(args.raw_replay.read_text(encoding="utf-8"))
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    if (sha256(args.zip) != OFFICIAL_ZIP_SHA256
            or cards["source_archive_sha256"] != OFFICIAL_ZIP_SHA256
            or replay["source_archive_sha256"] != OFFICIAL_ZIP_SHA256
            or replay["facts_sha256"] != sha256(args.facts)
            or replay["status"] != "exact_split_visible_raw_source_replay"
            or cards["fact_cache_sha256"] != sha256(args.facts)
            or cards["reference_artifact_sha256"] != sha256(args.reference)
            or cards["frozen_model_sha256"] != sha256(args.model)
            or cards["group_screen_sha256"] != sha256(args.screen)
            or screen["split_visible_facts_sha256"] != sha256(args.facts)
            or screen["reference_sha256"] != sha256(args.reference)
            or screen["seed0_model_sha256"] != sha256(args.model)):
        raise ValueError("Bound source/model/reference artifacts differ")
    with np.load(args.facts, allow_pickle=False) as saved:
        split = saved["split"].astype(str)
        dev = np.flatnonzero(split == "dev")
        ids = saved["ids"][dev].astype(str).tolist()
        positions = saved["positions"][dev].astype(int).tolist()
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        all_names = saved["names"].astype(str).tolist()
    if all_names != list(FACT_NAMES):
        raise ValueError("Fact vocabulary differs")
    names = compact["retained_facts"]
    chosen = next(row for row in screen["rows"] if row["ab_seed"] == cards["a_seed"])
    signs_by_name = chosen["tail_directions_by_fact"]
    signs = np.r_[[signs_by_name[name]["upper"] for name in names],
                  [signs_by_name[name]["lower"] for name in names]]
    reference = LabelBlindReference.load(args.reference)
    full = reference.transform(values, contexts=contexts)
    batch_all = evidence_subset(full, [all_names.index(name) for name in names], names,
                                values[:, all_names.index("tweet_sample_count")], True)
    batch = type(batch_all)(batch_all.upper[dev], batch_all.lower[dev],
                            batch_all.reliability[dev], batch_all.available[dev],
                            batch_all.reference_version, batch_all.names)
    raw_rows, graph = independent_source(args.zip, ids, positions, values[dev], all_names)
    model = Booster(model_file=str(args.model))
    summary = validate_cards(cards, raw_rows, graph, ids, positions, values[dev], all_names,
                             names, batch, signs, model, sha256(args.model),
                             float(chosen["threshold"]))
    mutations = (
        ("source_record_hash", lambda c: c.update(source_record_without_label_sha256="0" * 64)),
        ("source_locator", lambda c: c.update(official_source_locator="Twibot-20/dev.json[1]")),
        ("source_value", lambda c: c["fact_terms"][0].update(source_value=999999.0)),
        ("graph_declaration_hash", lambda c: c["fact_terms"][-1]["source_query"].update(incident_declarations_sha256="0" * 64)),
        ("tail_score", lambda c: c["fact_terms"][0].update(upper_tail_score=1e6)),
        ("reliability", lambda c: c["fact_terms"][0].update(reliability=0.0)),
        ("model_input", lambda c: c["fact_terms"][0].update(upper_model_input=1e6)),
        ("group_effect", lambda c: c["group_logit_effects"].update(profile_counts=100.0)),
        ("full_logit", lambda c: c.update(full_logit=100.0)),
        ("threshold", lambda c: c.update(threshold=0.99)),
        ("prediction", lambda c: c.update(prediction=1-c["prediction"])),
        ("evaluation_label", lambda c: c.update(evaluation_label=1-c["evaluation_label"])),
    )
    rejected = {}
    for label, mutation in mutations:
        altered = copy.deepcopy(cards)
        mutation(altered["decision_cards"][0])
        try:
            validate_cards(altered, raw_rows, graph, ids, positions, values[dev], all_names,
                           names, batch, signs, model, sha256(args.model),
                           float(chosen["threshold"]))
        except ValueError as error:
            rejected[label] = str(error)
        else:
            raise AssertionError(f"Tampered card accepted: {label}")
    report = {"scope": "all split-visible dev cards: raw record and graph declaration binding, reference input, frozen group effects and class",
              "source_archive_sha256": sha256(args.zip), "facts_sha256": sha256(args.facts),
              "reference_sha256": sha256(args.reference), "model_sha256": sha256(args.model),
              "cards_sha256": sha256(args.cards), "test_member_opened": False,
              "labels_used_to_form_evidence": False,
              "official_labels_used_only_for_evaluation_field_verification": True, **summary,
              "tamper_cases_rejected": rejected, "status": "all_cards_verified_all_mutations_rejected"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
