"""Predict TwiBot-20 accounts from official ZIP records, not supplied facts.

The frozen runtime verifies the archive, source audits, reference and model.
This entry point scans official train/dev/test records without opening account
labels, recomputes the requested 22 facts including native follow degrees,
and passes source-located tweets into the evidence-only classifier.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import ijson
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.twibot20_facts import FACT_NAMES, account_facts
from verifiable_bot.decision_dependency import additive_dependency
from verifiable_bot.twibot20_sparse_runtime import TwiBot20SparseRuntime, sha256


def project_paths(root: Path, archive: Path | None = None,
                  base_facts: Path | None = None) -> dict[str, Path]:
    model = root / "output/v3_twibot20_sparse_text_evidence"
    return {
        "archive": archive or Path("C:/Users/Lenovo/Downloads/Twibot-20.zip"),
        "base_facts": base_facts or root / "data/processed/twibot20_v3/facts.npz",
        "base_source_audit": root / "research/certificates/twibot20_full_source_audit.json",
        "active_source_audit": root / "research/certificates/twibot20_sparse_active_unigram_full_source_audit.json",
        "active_cache": root / "data/processed/twibot20_v3/sparse_active_unigram_audit.npz",
        "screen": model / "validation_only.json",
        "selection_lock": model / "selection_lock.json",
        "artifact": model / "sparse_l1_c1_model.npz",
        "reference": model / "reference.json.gz",
        "selected_report": model / "selected/report.json",
    }


def source_rows(archive: Path, cache: dict, requested: list[str]) -> tuple[dict, set]:
    requested_set = set(requested)
    found = {}
    incident: set[tuple[str, str]] = set()
    expected_counts = {"train": 8278, "dev": 2365, "test": 1183}
    ids = cache["ids"]
    parts = cache["split"]
    positions = cache["positions"]
    by_id = {uid: i for i, uid in enumerate(ids)}
    if len(by_id) != len(ids) or not requested_set <= set(by_id):
        raise ValueError("requested account absent from official labeled population")
    with zipfile.ZipFile(archive) as zipped:
        offset = 0
        for part, count in expected_counts.items():
            member = f"Twibot-20/{part}.json"
            with zipped.open(member) as stream:
                seen = 0
                for position, row in enumerate(ijson.items(stream, "item")):
                    i = offset + position
                    uid = str(row["ID"])
                    if (i >= len(ids) or ids[i] != uid or parts[i] != part
                            or positions[i] != position):
                        raise ValueError("official ZIP and audited account order differ")
                    neighbor = row.get("neighbor")
                    if neighbor is not None:
                        for target in neighbor.get("following") or []:
                            target = str(target)
                            if uid != target and (uid in requested_set or target in requested_set):
                                incident.add((uid, target))
                        for source in neighbor.get("follower") or []:
                            source = str(source)
                            if uid != source and (uid in requested_set or source in requested_set):
                                incident.add((source, uid))
                    if uid in requested_set:
                        found[uid] = {"row": row, "member": member,
                                      "position": position, "cache_index": i}
                    seen += 1
            if seen != count:
                raise ValueError("official ZIP split size differs")
            offset += seen
    if offset != len(ids) or set(found) != requested_set:
        raise ValueError("official ZIP omitted a requested account")
    return found, incident


def structured_queries(member: str, position: int, uid: str) -> list[dict]:
    account = f"{member}[{position}]"
    queries = []
    for name in FACT_NAMES:
        if name in FACT_NAMES[:9]:
            ids = [f"{account}.profile"]
        elif name in FACT_NAMES[9:16]:
            ids = [f"{account}.tweet[*]"]
        elif name == "description_length":
            ids = [f"{account}.profile.description"]
        elif name.startswith("sampled_"):
            ids = [f"{account}.neighbor"]
        else:
            ids = [f"Twibot-20/train.json,dev.json,test.json#incident_follow_declarations={uid}"]
        queries.append({"fact_name": name, "source_ids": ids})
    return queries


def predict(root: Path, account_ids: list[str], *, archive: Path | None = None,
            base_facts: Path | None = None) -> dict:
    if not account_ids or len(set(account_ids)) != len(account_ids):
        raise ValueError("request at least one unique account ID")
    paths = project_paths(root, archive, base_facts)
    runtime = TwiBot20SparseRuntime(**paths)
    with np.load(paths["base_facts"], allow_pickle=False) as saved:
        cache = {name: saved[name].astype(str if name in {"ids", "split", "contexts", "names"}
                                               else int if name == "positions" else float)
                 for name in ("ids", "split", "positions", "contexts", "names", "values")}
    if cache["names"].tolist() != list(FACT_NAMES):
        raise ValueError("official source fact vocabulary differs")
    found, incident = source_rows(paths["archive"], cache, account_ids)
    facts, contexts, tweets, locators = [], [], [], []
    max_difference = 0.0
    for uid in account_ids:
        item = found[uid]
        row = item["row"]
        base, _, context = account_facts(row)
        outgoing = {target for source, target in incident if source == uid}
        incoming = {source for source, target in incident if target == uid}
        generated = np.r_[base, [len(outgoing), len(incoming),
                                  len(outgoing & incoming)]]
        expected = cache["values"][item["cache_index"]]
        if generated.shape != (22,) or not np.allclose(
                generated, expected, rtol=0, atol=1e-10, equal_nan=True):
            raise ValueError("official raw profile, tweet or follow facts differ")
        finite = np.isfinite(generated) & np.isfinite(expected)
        if finite.any():
            max_difference = max(max_difference, float(np.max(
                np.abs(generated[finite] - expected[finite]))))
        if context != cache["contexts"][item["cache_index"]]:
            raise ValueError("official context differs from audited source")
        facts.append(generated)
        contexts.append(context)
        tweets.append([str(text or "") for text in row.get("tweet") or []])
        locators.append((item["member"], item["position"]))
    decisions = runtime.predict(np.asarray(facts), contexts, tweets,
                                account_ids, source_rows=locators)
    max_unigram_difference = 0.0
    with np.load(paths["active_cache"], allow_pickle=False) as checked:
        if not np.array_equal(checked["ids"].astype(str), cache["ids"]):
            raise ValueError("independent unigram audit account order differs")
        for uid, decision in zip(account_ids, decisions, strict=True):
            audited = checked["values"][found[uid]["cache_index"]].astype(float)
            actual = np.asarray(decision["active_unigram_values"], dtype=float)
            if audited.shape != actual.shape:
                raise ValueError("independent unigram audit width differs")
            difference = float(np.max(np.abs(audited - actual)))
            if difference > 1e-10:
                raise ValueError("official tweet unigram values differ from independent audit")
            max_unigram_difference = max(max_unigram_difference, difference)
    for decision, (member, position), uid, raw_facts in zip(
            decisions, locators, account_ids, facts, strict=True):
        decision["structured_recompute_queries"] = structured_queries(
            member, position, uid)
        decision["structured_raw_facts"] = {
            name: (float(value) if np.isfinite(value) else None)
            for name, value in zip(FACT_NAMES, raw_facts, strict=True)
        }
        decision["decision_dependency"] = additive_dependency(
            decision["structured_names"] + decision["active_unigram_terms"],
            decision["structured_contributions"] +
            decision["active_unigram_contributions"],
            decision["intercept"], decision["threshold"],
            decision["logit"], decision["prediction"])
    return {
        "dataset": "twibot20", "source_level": "official_profile_tweets_and_sampled_follow_declarations",
        "source_binding": {name: sha256(paths[name]) for name in
                           ("archive", "base_facts", "base_source_audit",
                            "active_source_audit", "active_cache")},
        "model_binding": {name: sha256(paths[name]) for name in
                          ("selection_lock", "artifact", "reference", "selected_report")},
        "max_raw_to_audited_fact_difference": max_difference,
        "max_independent_active_unigram_value_difference": max_unigram_difference,
        "classification_label_field_used": False,
        "caller_supplied_fact_matrix_accepted": False,
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--base-facts", type=Path)
    parser.add_argument("--account-id", action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = predict(args.project_root.resolve(), args.account_id,
                     archive=args.zip, base_facts=args.base_facts)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
