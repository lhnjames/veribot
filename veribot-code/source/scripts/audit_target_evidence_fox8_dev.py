"""Audit source sufficiency and removal response on the dev split only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import median
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "source/scripts"),
                str(ROOT / "source/hgt_import/code")]

from hgt_model import VerifiableHGT
from relnet_reimplementation import GatedRelNet
from train_aaai_veribot_fox8 import load_inputs, sha256_file
from train_two_stage_fox8_evidence import metrics
from verifiable_bot.fox8_atomic_sources import load_account_atoms, load_relation_atoms
from verifiable_bot.aaai_fox8_raw_features import RAW_PROFILE_FIELDS
from verifiable_bot.fox8_hgt_context_closure import (
    hgt_context_closure, verify_hgt_context_locality,
)
from verifiable_bot.fox8_target_evidence_intervention import (
    FixedBotEvidenceIntervention, TargetEvidenceIntervention,
)
from verifiable_bot.fox8_source_grouping import fine_grained_target_groups
from verifiable_bot.fox8_target_evidence_model import (
    HGTTargetFactSelector, HGTTargetRelationSelector, TargetEvidenceFox8,
)
from verifiable_bot.fox8_two_stage_inputs import ALL_FACTS, Fox8TwoStageInputs
from verifiable_bot.source_coalitions import (
    greedy_bot_certificate, search_source_coalitions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sample-per-class", type=int, default=6)
    parser.add_argument("--sample-bot-only", action="store_true")
    parser.add_argument("--sampling-seed", type=int, default=20261110)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--group-ablation", action="store_true")
    parser.add_argument("--coalition-audit", action="store_true")
    parser.add_argument("--coalition-query-budget", type=int, default=40)
    parser.add_argument("--coalition-beam-width", type=int, default=3)
    parser.add_argument("--fine-grained-groups", action="store_true")
    parser.add_argument("--class-specific-ablation", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--emit-source-ids", action="store_true")
    parser.add_argument("--skip-interventions", action="store_true")
    parser.add_argument("--score-only", action="store_true",
                        help="Audit centered class support for every dev account")
    parser.add_argument("--certificate-only", action="store_true",
                        help="Audit the fixed-selection evidence-only Bot score")
    parser.add_argument("--certificate-greedy", action="store_true")
    parser.add_argument("--certificate-query-budget", type=int, default=60)
    args = parser.parse_args()
    if args.score_only and (args.split != "dev" or args.coalition_audit or
                            args.group_ablation or args.class_specific_ablation or
                            args.skip_interventions or args.sample_bot_only or
                            args.certificate_only):
        parser.error("score-only supports the dev split without source searches")
    if args.certificate_only and (args.split != "dev" or not args.sample_bot_only or
                                  args.group_ablation or args.coalition_audit or
                                  args.class_specific_ablation or
                                  args.skip_interventions):
        parser.error("certificate audit requires Bot-only dev sampling")
    if args.certificate_greedy and not args.certificate_only:
        parser.error("certificate greedy search needs certificate-only mode")
    if args.skip_interventions and (args.group_ablation or args.coalition_audit or
                                    args.class_specific_ablation):
        parser.error("source search and ablations require interventions")
    if args.coalition_audit and args.split != "dev":
        parser.error("coalition search requires the dev split")
    if args.fine_grained_groups and not args.coalition_audit:
        parser.error("fine-grained groups require coalition search")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    raw = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    events = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
    base_path = ROOT / "output/aaai_veribot_fox8/seed0/model.pt"
    checkpoint_path = args.run / "model.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if args.certificate_only and not saved.get("selected_neighbor_sources", False):
        parser.error("certificate audit requires selected neighbor sources")
    if (saved["source_sha256"] != sha256_file(raw) or
            saved["event_index_sha256"] != sha256_file(events) or
            saved["hgt_context_checkpoint_sha256"] != sha256_file(base_path)):
        raise ValueError("source contract differs from frozen model")
    threshold = float(saved["dev_logit_threshold"])
    x, y, split, ids, graph, pack = load_inputs()
    sampler = random.Random(args.sampling_seed)
    sampled_classes = (1,) if args.sample_bot_only else (0, 1)
    selected_ids = tuple(ids[int(i)] for cls in sampled_classes
                         for i in sampler.sample(
                             [int(j) for j in split[args.split] if int(y[j]) == cls],
                             args.sample_per_class))
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3).to(device).eval()
    hgt.load_state_dict(base["model_state"]["hgt"])
    bank = Fox8TwoStageInputs.build(
        load_account_atoms(raw, ids), load_relation_atoms(events, raw, ids), ids)
    relnet = GatedRelNet(len(ALL_FACTS), hidden_channels=128,
                         relation_names=tuple(bank.full_graph),
                         num_layers=1, dropout=0.3, out_channels=2)
    selector_type = (HGTTargetRelationSelector if saved.get("relation_gates", False)
                     else HGTTargetFactSelector)
    model = TargetEvidenceFox8(
        selector_type(context_dim=128), relnet,
        torch.tensor(saved["scaler_mean"]),
        torch.tensor(saved["scaler_scale"]),
        centered_evidence=saved.get("centered_evidence", False),
        sample_training_gates=saved.get("sample_training_gates", True),
        bot_only_support=saved.get("bot_only_support", False),
        selected_neighbor_sources=saved.get(
            "selected_neighbor_sources", False)).to(device).eval()
    model.load_state_dict(saved["model_state"])
    oracle = TargetEvidenceIntervention(
        hgt=hgt, model=model, batch=bank,
        base_mean=np.asarray(base["input_contract"]["scaler_mean"]),
        base_scale=np.asarray(base["input_contract"]["scaler_scale"]))
    with torch.inference_mode():
        context = hgt({"user": x.to(device)}, {
            key: edges.to(device) for key, edges in graph.items()
        })["user_emb"]
        emitted = model(context, bank)
    if args.score_only:
        if not saved.get("centered_evidence", False):
            raise ValueError("score-only audit requires centered evidence scores")
        logits = emitted["logits"].detach().cpu().numpy()
        hard = emitted["hard_selection"]
        dev_rows = []
        for index in split["dev"]:
            index = int(index)
            label = int(y[index])
            pred = int(logits[index, 1] - logits[index, 0] >= threshold)
            p0, p1 = bank.profile_offsets[index:index + 2]
            t0, t1 = bank.tweet_offsets[index:index + 2]
            forward = (bank.relation_target == index).to(
                hard["relation_forward"].device)
            reverse = (bank.relation_source == index).to(
                hard["relation_reverse"].device)
            counts = {}
            for cls in (0, 1):
                if cls == 0 and saved.get("bot_only_support", False):
                    counts["human"] = {
                        "target_profile": 0, "target_tweets": 0,
                        "incident_relation_messages": 0,
                    }
                    continue
                counts["human" if cls == 0 else "bot"] = {
                    "target_profile": int(hard["profile"][p0:p1, cls].sum()),
                    "target_tweets": int(hard["tweet"][t0:t1, cls].sum()),
                    "incident_relation_messages": int(
                        hard["relation_forward"][forward, cls].sum() +
                        hard["relation_reverse"][reverse, cls].sum()),
                }
            dev_rows.append({
                "account_id": ids[index], "label": label,
                "predicted_bot": bool(pred),
                "correct": pred == label,
                "human_support": float(logits[index, 0]),
                "bot_support": float(logits[index, 1]),
                "true_class_support": float(logits[index, label]),
                "rival_class_support": float(logits[index, 1 - label]),
                "selected_source_counts": counts,
            })

        def aggregate(rows: list[dict]) -> dict:
            return {
                "n": len(rows), "correct": sum(row["correct"] for row in rows),
                "true_support_positive": sum(
                    row["true_class_support"] > 0 for row in rows),
                "true_support_at_least_0_5": sum(
                    row["true_class_support"] >= 0.5 for row in rows),
                "median_true_class_support": median(
                    row["true_class_support"] for row in rows),
                "median_rival_class_support": median(
                    row["rival_class_support"] for row in rows),
                "median_true_class_selected_profile": median(
                    row["selected_source_counts"][
                        "human" if row["label"] == 0 else "bot"]["target_profile"]
                    for row in rows),
                "median_true_class_selected_tweets": median(
                    row["selected_source_counts"][
                        "human" if row["label"] == 0 else "bot"]["target_tweets"]
                    for row in rows),
                "median_true_class_selected_relation_messages": median(
                    row["selected_source_counts"][
                        "human" if row["label"] == 0 else "bot"][
                            "incident_relation_messages"] for row in rows),
            }

        summary = {
            "split": "dev", "checkpoint_sha256": sha256_file(checkpoint_path),
            "source_sha256": sha256_file(raw),
            "bot_only_support": saved.get("bot_only_support", False),
            "threshold_selected_on_dev": threshold,
            "gate_probabilities": {
                channel: {
                    name: {
                        "count": int(values.shape[0]),
                        "quantiles_0_1_5_9_10": np.quantile(
                            values[:, cls], [0, 0.1, 0.5, 0.9, 1]).tolist(),
                        "fraction_between_0_45_and_0_55": float(np.mean(
                            (values[:, cls] >= 0.45) &
                            (values[:, cls] <= 0.55))),
                        "hard_selected_fraction": float(np.mean(
                            hard[channel][:, cls].detach().cpu().numpy())),
                    }
                    for cls, name in (((1, "bot"),) if saved.get(
                        "bot_only_support", False) else
                        ((0, "human"), (1, "bot")))
                }
                for channel, probability in emitted[
                    "gate_probabilities"].items()
                for values in [probability.detach().cpu().numpy()]
            },
            "by_true_class": {
                "human": aggregate([row for row in dev_rows
                                    if row["label"] == 0]),
                "bot": aggregate([row for row in dev_rows
                                  if row["label"] == 1]),
            },
            "accounts": dev_rows,
        }
        output = args.run / "dev_centered_support_audit.json"
        output.write_text(json.dumps(summary, indent=2) + "\n",
                          encoding="utf-8")
        print(json.dumps(summary["by_true_class"]), flush=True)
        return
    if args.split == "test":
        logits = emitted["logits"].detach().cpu().numpy()
        labels = y.numpy()
        report = {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "source_sha256": sha256_file(raw),
            "event_index_sha256": sha256_file(events),
            "threshold_from_dev": threshold,
            "uncalibrated_test": metrics(logits, labels, split["test"]),
            "calibrated_test": metrics(logits, labels, split["test"], threshold),
            "test_predictions": [{
                "account_id": ids[int(i)], "label": int(labels[i]),
                "human_score": float(logits[i, 0]),
                "bot_score": float(logits[i, 1]),
            } for i in split["test"]],
        }
        (args.run / "frozen_test_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items()
                          if key != "test_predictions"}), flush=True)
    source_universe = frozenset(oracle.lookup)
    if args.certificate_only:
        with torch.inference_mode():
            fixed_certificate = model.verify_bot_evidence(
                bank, emitted["hard_selection"]).detach().cpu().numpy()
        fixed_oracle = FixedBotEvidenceIntervention(
            oracle, emitted["hard_selection"])
    result_path = args.run / f"{args.split}_source_interventions{args.tag}.json"
    header = {
        "split": args.split, "sampling_seed": args.sampling_seed,
        "sample_per_class": args.sample_per_class,
        "sampled_classes": sampled_classes,
        "selected_account_ids": selected_ids,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_sha256": sha256_file(raw),
        "threshold_selected_on_dev": threshold,
    }
    cases = []
    for uid in selected_ids:
        account = bank.accounts[bank.ids.index(uid)]
        p0, p1 = bank.profile_offsets[bank.ids.index(uid):bank.ids.index(uid) + 2]
        t0, t1 = bank.tweet_offsets[bank.ids.index(uid):bank.ids.index(uid) + 2]
        selected = set()
        if saved.get("bot_only_support", False):
            profile_keep = emitted["hard_selection"]["profile"][p0:p1, 1].bool()
            tweet_keep = emitted["hard_selection"]["tweet"][t0:t1, 1].bool()
        else:
            profile_keep = emitted["hard_selection"]["profile"][p0:p1].any(dim=1)
            tweet_keep = emitted["hard_selection"]["tweet"][t0:t1].any(dim=1)
        target_profile = {account.source_id("profile", field)
                          for field, keep in zip(account.profile_field_ids,
                                                 profile_keep.tolist()) if keep}
        target_tweets = {account.source_id("tweet", ordinal)
                         for ordinal, keep in zip(account.tweet_ordinals,
                                                  tweet_keep.tolist()) if keep}
        selected.update(target_profile)
        selected.update(target_tweets)
        neighbors = {uid}
        native_relations = set()
        incident = []
        for j, relation in enumerate(bank.relations):
            if relation.source_account_id == uid:
                other = relation.target_account_id
                channel = "relation_reverse"
            elif relation.target_account_id == uid:
                other = relation.source_account_id
                channel = "relation_forward"
            else:
                continue
            incident.append((j, relation, channel))
            if channel not in emitted["hard_selection"]:
                chosen = True
            elif saved.get("bot_only_support", False):
                chosen = bool(emitted["hard_selection"][channel][j, 1])
            else:
                chosen = bool(emitted["hard_selection"][channel][j].any())
            if chosen:
                neighbors.add(other)
                native_relations.add(relation.source_id)
        selected.update(native_relations)
        neighbor_profile, neighbor_tweets = set(), set()
        for neighbor_id in neighbors - {uid}:
            neighbor_index = bank.ids.index(neighbor_id)
            neighbor = bank.accounts[neighbor_index]
            np0, np1 = bank.profile_offsets[neighbor_index:neighbor_index + 2]
            nt0, nt1 = bank.tweet_offsets[neighbor_index:neighbor_index + 2]
            profile_mask = (emitted["hard_selection"]["profile"][np0:np1, 1]
                            if saved.get("selected_neighbor_sources", False)
                            else [True] * len(neighbor.profile_field_ids))
            tweet_mask = (emitted["hard_selection"]["tweet"][nt0:nt1, 1]
                          if saved.get("selected_neighbor_sources", False)
                          else [True] * len(neighbor.tweet_ordinals))
            neighbor_profile.update(
                neighbor.source_id("profile", field)
                for field, keep in zip(neighbor.profile_field_ids, profile_mask)
                if keep)
            neighbor_tweets.update(
                neighbor.source_id("tweet", ordinal)
                for ordinal, keep in zip(neighbor.tweet_ordinals, tweet_mask)
                if keep)
        selected.update(neighbor_profile)
        selected.update(neighbor_tweets)
        stage2 = frozenset(selected)
        if args.certificate_only:
            def fixed_score(removed: frozenset[str]) -> float:
                return fixed_oracle.predict(uid, removed)["margin"]

            original_score = float(fixed_certificate[bank.ids.index(uid)])
            emitted_score = float(emitted["logits"][bank.ids.index(uid), 1])
            keep_score = fixed_score(source_universe - stage2)
            remove_score = fixed_score(stage2)
            case = {
                "account_id": uid, "label": int(y[bank.ids.index(uid)]),
                "source_count": len(stage2),
                "emitted_bot_score": emitted_score,
                "fixed_full_bot_score": original_score,
                "fixed_keep_bot_score": keep_score,
                "fixed_remove_bot_score": remove_score,
                "full_vs_emitted_absolute_error": abs(original_score - emitted_score),
                "keep_vs_full_absolute_error": abs(keep_score - original_score),
                "bot_when_kept": keep_score >= threshold,
                "bot_when_removed": remove_score >= threshold,
            }
            if args.certificate_greedy:
                groups = fine_grained_target_groups(
                    account=account, target_profile=target_profile,
                    target_tweets=target_tweets,
                    incident_relations=(relation for _, relation, _ in incident),
                    selected_relations=native_relations,
                    neighbor_profile=neighbor_profile,
                    neighbor_tweets=neighbor_tweets,
                    hgt_remainder=frozenset())
                case["greedy_certificate"] = greedy_bot_certificate(
                    account_id=uid, groups=groups,
                    candidate_sources=stage2, threshold=threshold,
                    margin_buffer=0.05, oracle=fixed_oracle,
                    query_budget=args.certificate_query_budget)
            if args.emit_source_ids:
                case["source_ids"] = sorted(stage2)
            cases.append(case)
            result_path.write_text(json.dumps({**header,
                                               "audit_kind": "fixed_selection_certificate",
                                               "completed_cases": len(cases),
                                               "cases": cases}, indent=2) + "\n",
                                   encoding="utf-8")
            print(json.dumps({key: value for key, value in case.items()
                              if key != "source_ids"}), flush=True)
            continue
        hgt_report, restricted_graph, nodes = hgt_context_closure(
            bank, graph, [uid])
        hgt_context_max_difference = verify_hgt_context_locality(
            hgt, x.to(device), graph, restricted_graph, nodes, [uid],
            bank.ids)
        full = set(stage2)
        for i in nodes:
            source_account = bank.accounts[i]
            full.update(source_account.source_id("profile", field)
                        for field in source_account.profile_field_ids)
            full.update(source_account.source_id("tweet", ordinal)
                        for ordinal in source_account.tweet_ordinals)
        full.update(hgt_report["native_relation_source_ids"])
        full = frozenset(full)
        if full - source_universe:
            raise ValueError("dependency closure contains an unknown source")
        reference = emitted["logits"][bank.ids.index(uid)].detach().cpu().numpy()
        if args.skip_interventions:
            case = {
                "account_id": uid, "label": int(y[bank.ids.index(uid)]),
                "emitted_human_score": float(reference[0]),
                "emitted_bot_score": float(reference[1]),
                "stage2_source_count": len(stage2),
                "full_dependency_source_count": len(full),
                "hgt_context_account_count": hgt_report["context_account_count"],
                "hgt_context_max_difference": hgt_context_max_difference,
            }
            if args.emit_source_ids:
                case["source_ids_stage2"] = sorted(stage2)
                case["source_ids_hgt_context_remainder"] = sorted(full - stage2)
                locators = []
                for source_id in sorted(full):
                    kind, position = oracle.lookup[source_id]
                    if kind == "profile":
                        a = bank.accounts[int(bank.profile_account[position])]
                        field = RAW_PROFILE_FIELDS[int(bank.profile_field[position])]
                        value = json.dumps(a.profile[field], ensure_ascii=False,
                                           sort_keys=True, default=str)
                        locator = {"source_id": source_id, "kind": kind,
                                   "raw_line": a.line_number, "field": field,
                                   "value_sha256": hashlib.sha256(
                                       value.encode("utf-8")).hexdigest()}
                    elif kind == "tweet":
                        i = int(bank.tweet_account[position])
                        a = bank.accounts[i]
                        tweet = a.tweets[position - int(bank.tweet_offsets[i])]
                        locator = {"source_id": source_id, "kind": kind,
                                   "raw_line": tweet.line_number,
                                   "tweet_ordinal": tweet.ordinal,
                                   "tweet_id": tweet.tweet_id,
                                   "record_sha256": tweet.record_sha256}
                    else:
                        relation = bank.relations[position]
                        locator = {"source_id": source_id, "kind": kind,
                                   "relation": relation.relation,
                                   "raw_locators": relation.raw_locators}
                    locators.append(locator)
                case["raw_source_locators"] = locators
            cases.append(case)
            result_path.write_text(json.dumps({**header,
                                               "completed_cases": len(cases),
                                               "cases": cases}, indent=2) + "\n",
                                   encoding="utf-8")
            print(json.dumps({key: value for key, value in case.items()
                              if not key.startswith("source_ids") and
                              key != "raw_source_locators"}), flush=True)
            continue
        baseline = oracle.predict(uid)
        if abs(baseline["margin"] - float(reference[1] - reference[0])) > 1e-4:
            raise ValueError("full-path intervention baseline differs from frozen model")
        original = baseline["margin"] >= threshold
        scores = {}
        for name, sources in (("stage2", stage2), ("stage2_plus_hgt", full)):
            keep = oracle.predict(uid, source_universe - sources)
            remove = oracle.predict(uid, sources)
            scores[name] = {
                "source_count": len(sources),
                "keep_margin": keep["margin"],
                "remove_margin": remove["margin"],
                "sufficient_for_original_class":
                    (keep["margin"] >= threshold) == original,
                "collectively_necessary_for_original_class":
                    (remove["margin"] >= threshold) != original,
            }
        case = {"account_id": uid, "label": int(y[bank.ids.index(uid)]),
                "predicted_bot": original, "baseline_margin": baseline["margin"],
                "hgt_context_account_count": hgt_report["context_account_count"],
                "hgt_context_max_difference": hgt_context_max_difference,
                "tests": scores}
        if args.class_specific_ablation:
            class_sources = []
            for cls in (0, 1):
                keep_profile = emitted["hard_selection"]["profile"][p0:p1, cls]
                keep_tweet = emitted["hard_selection"]["tweet"][t0:t1, cls]
                current = {account.source_id("profile", field)
                           for field, keep in zip(account.profile_field_ids,
                                                  keep_profile.tolist()) if keep}
                current.update(account.source_id("tweet", ordinal)
                               for ordinal, keep in zip(account.tweet_ordinals,
                                                        keep_tweet.tolist()) if keep)
                for j, relation, channel in incident:
                    if (channel in emitted["hard_selection"] and
                            bool(emitted["hard_selection"][channel][j, cls])):
                        current.add(relation.source_id)
                class_sources.append(frozenset(current))
            own = class_sources[int(original)]
            rival = class_sources[1 - int(original)]
            shared = own & rival
            partitions = {
                "predicted_class_selected": own,
                "rival_class_selected": rival,
                "predicted_class_unique": own - rival,
                "rival_class_unique": rival - own,
                "shared_selected": shared,
            }
            direction = 1 if original else -1
            class_effects = {}
            for name, sources in partitions.items():
                if sources:
                    after = oracle.predict(uid, sources)["margin"]
                else:
                    after = baseline["margin"]
                class_effects[name] = {
                    "source_count": len(sources),
                    "margin_after_removal": after,
                    "signed_support_loss": direction * (baseline["margin"] - after),
                    "removal_flips_original_class":
                        (after >= threshold) != original,
                }
            case["class_specific_target_ablation"] = class_effects
        if args.group_ablation or args.coalition_audit:
            groups = (fine_grained_target_groups(
                account=account, target_profile=target_profile,
                target_tweets=target_tweets,
                incident_relations=(relation for _, relation, _ in incident),
                selected_relations=native_relations,
                neighbor_profile=neighbor_profile,
                neighbor_tweets=neighbor_tweets,
                hgt_remainder=full - stage2)
                if args.fine_grained_groups else {
                    "target_profile": frozenset(target_profile),
                    "target_tweets": frozenset(target_tweets),
                    "incident_native_relations": frozenset(native_relations),
                    "direct_neighbor_profile": frozenset(neighbor_profile),
                    "direct_neighbor_tweets": frozenset(neighbor_tweets),
                    "hgt_context_remainder": full - stage2,
                })
            if any(groups[a] & groups[b] for a in groups for b in groups
                   if a < b) or frozenset().union(*groups.values()) != full:
                raise ValueError("source groups must partition the complete dependency")
            if args.group_ablation:
                direction = 1 if original else -1
                group_results = {}
                for name, sources in groups.items():
                    if not sources:
                        continue
                    removed = oracle.predict(uid, sources)["margin"]
                    kept = oracle.predict(uid, source_universe - sources)["margin"]
                    group_results[name] = {
                        "source_count": len(sources),
                        "remove_margin": removed,
                        "keep_only_margin": kept,
                        "signed_support_loss":
                            direction * (baseline["margin"] - removed),
                        "individual_removal_flips_class":
                            (removed >= threshold) != original,
                        "group_alone_preserves_class":
                            (kept >= threshold) == original,
                    }
                interactions = []
                names = list(group_results)
                for i, first in enumerate(names):
                    for second in names[i + 1:]:
                        removed = oracle.predict(
                            uid, groups[first] | groups[second])["margin"]
                        joint_loss = direction * (baseline["margin"] - removed)
                        interactions.append({
                            "first": first, "second": second,
                            "joint_remove_margin": removed,
                            "joint_removal_flips_class":
                                (removed >= threshold) != original,
                            "joint_signed_support_loss": joint_loss,
                            "nonadditive_support_loss": joint_loss -
                                group_results[first]["signed_support_loss"] -
                                group_results[second]["signed_support_loss"],
                        })
                case["group_ablation"] = {
                    "groups": group_results,
                    "pairwise_removal": interactions,
                    "interpretation": "signed loss is positive when removing sources harms the original predicted class; pairwise nonadditivity is an intervention interaction, not a causal identification claim",
                }
            if args.coalition_audit:
                active_groups = {name: sources for name, sources in groups.items()
                                 if sources}
                coalition = search_source_coalitions(
                    account_id=uid, groups=active_groups,
                    full_dependency=full, threshold=threshold, oracle=oracle,
                    query_budget=args.coalition_query_budget,
                    beam_width=args.coalition_beam_width)
                source_group = {source_id: name for name, sources in
                                active_groups.items() for source_id in sources}
                source_nodes = []
                account_nodes = {uid}
                for source_id in sorted(full):
                    kind, position = oracle.lookup[source_id]
                    node = {"source_id": source_id, "kind": kind,
                            "group": source_group[source_id]}
                    if kind == "profile":
                        owner_index = int(bank.profile_account[position])
                        owner = bank.ids[owner_index]
                        field = RAW_PROFILE_FIELDS[int(bank.profile_field[position])]
                        value = json.dumps(bank.accounts[owner_index].profile[field],
                                           ensure_ascii=False, sort_keys=True,
                                           default=str)
                        node.update({
                            "account_id": owner,
                            "raw_line": bank.accounts[owner_index].line_number,
                            "field": field,
                            "value_sha256": hashlib.sha256(
                                value.encode("utf-8")).hexdigest(),
                        })
                        account_nodes.add(owner)
                    elif kind == "tweet":
                        owner_index = int(bank.tweet_account[position])
                        owner = bank.ids[owner_index]
                        tweet = bank.accounts[owner_index].tweets[
                            position - int(bank.tweet_offsets[owner_index])]
                        node.update({
                            "account_id": owner,
                            "raw_line": tweet.line_number,
                            "tweet_ordinal": tweet.ordinal,
                            "tweet_id": tweet.tweet_id,
                            "record_sha256": tweet.record_sha256,
                        })
                        account_nodes.add(owner)
                    else:
                        relation = bank.relations[position]
                        node.update({
                            "source_account_id": relation.source_account_id,
                            "target_account_id": relation.target_account_id,
                            "relation": relation.relation,
                            "raw_locators": relation.raw_locators,
                        })
                        account_nodes.update((relation.source_account_id,
                                              relation.target_account_id))
                    source_nodes.append(node)
                case["source_graph"] = {
                    "root_account_id": uid,
                    "account_nodes": sorted(account_nodes),
                    "source_nodes": source_nodes,
                    "source_node_count": len(source_nodes),
                    "typed_attachment_edges": sum(
                        2 if node["kind"] == "relation" else 1
                        for node in source_nodes),
                }
                case["coalition_audit"] = coalition
        cases.append(case)
        result_path.write_text(json.dumps({**header, "completed_cases": len(cases),
                                           "cases": cases}, indent=2) + "\n",
                               encoding="utf-8")
        logged = {key: value for key, value in case.items()
                  if key not in ("source_graph", "coalition_audit")}
        if args.coalition_audit:
            logged["coalition_count"] = len(case["coalition_audit"]["coalitions"])
            logged["searched_bidirectional_group_sets"] = case[
                "coalition_audit"]["searched_bidirectional_group_sets"]
        print(json.dumps(logged), flush=True)


if __name__ == "__main__":
    main()
