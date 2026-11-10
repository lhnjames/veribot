"""Dev-only screen of source-addressable reply-partner facts without two score cues."""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json

import numpy as np

from gate0_shallow_tree import load, sha256_file
from screen_botsim_observed_means_dev import (
    ACTIVITY, DATA, FOLDER as CURRENT, NATIVE, ROOT, USERS, corrected_facts)
from screen_botsim_simulator_shortcuts_dev import candidate


TRACE = ROOT / "data/processed/botsim_v3/reply_sources.jsonl.gz"
MANIFEST = ROOT / "data/processed/botsim_v3/manifest.json"
OUT = ROOT / "output/v3_botsim_reply_partner_evidence/dev_only.json"
EXCLUDED = ("avg_upvote_ratio", "avg_comment_score")
GROUPS = {
    "partner_repetition": (
        "native_reply_out_repeat_fraction",
        "native_reply_in_repeat_fraction",
        "native_reply_out_top_partner_fraction",
        "native_reply_in_top_partner_fraction"),
    "partner_reciprocity": (
        "native_reply_out_mutual_partner_fraction",
        "native_reply_in_mutual_partner_fraction"),
}
CONFIGS = {
    "no_new_partner_facts": (),
    "repetition_only": ("partner_repetition",),
    "reciprocity_only": ("partner_reciprocity",),
    "repetition_and_reciprocity": tuple(GROUPS),
}


def partner_facts(ids: list[str]) -> np.ndarray:
    """Count released cross-account comments by source and parent author."""
    id_set = set(ids)
    if len(id_set) != len(ids):
        raise ValueError("Duplicate BotSim IDs")
    outgoing: dict[str, Counter[str]] = defaultdict(Counter)
    incoming: dict[str, Counter[str]] = defaultdict(Counter)
    with gzip.open(TRACE, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            source = str(row["source_account_id"])
            target = str(row["target_account_id"])
            if source not in id_set or target not in id_set or source == target:
                raise ValueError("Reply trace account population differs")
            outgoing[source][target] += 1
            incoming[target][source] += 1
    values = np.full((len(ids), 6), np.nan, dtype=np.float64)
    for i, uid in enumerate(ids):
        out = outgoing[uid]
        inc = incoming[uid]
        mutual = set(out).intersection(inc)
        for offset, partners in ((0, out), (1, inc)):
            total = sum(partners.values())
            if total:
                values[i, offset] = 1.0 - len(partners) / total
                values[i, offset + 2] = max(partners.values()) / total
                values[i, offset + 4] = len(mutual) / len(partners)
    return values


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (sha256_file(TRACE) != manifest["reply_sources_sha256"] or
            sha256_file(USERS) != manifest["input_sha256"][str(USERS.resolve())] or
            sha256_file(ACTIVITY) != manifest["input_sha256"][str(ACTIVITY.resolve())]):
        raise ValueError("BotSim native reply source changed")
    original = json.loads((CURRENT / "dev_only.json").read_text(encoding="utf-8"))
    previous = json.loads((ROOT / "output/v3_botsim_engagement_shortcuts/dev_only.json")
                          .read_text(encoding="utf-8"))
    if previous["fixed_test_predictions_or_metrics_accessed"] is not False:
        raise ValueError("Prior screen selection accessed fixed test")
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    all_runs = []
    for seed in (0, 1, 2):
        raw, labels, ids, names, split, source = load(
            "botsim", DATA, USERS, seed, NATIVE)
        if (source["split_sha256"] != original["runs"][seed]["split_sha256"] or
                source["split_sha256"] != previous["runs"][seed]["split_sha256"]):
            raise ValueError("Official BotSim split differs")
        facts, _ = corrected_facts(raw, names, ids, activity)
        new = partner_facts(ids)
        if np.any((new[np.isfinite(new)] < 0) | (new[np.isfinite(new)] > 1)):
            raise ValueError("Reply-partner ratio outside unit interval")
        new_names = [name for members in GROUPS.values() for name in members]
        if len(new_names) != new.shape[1] or len(set(new_names)) != len(new_names):
            raise ValueError("Partner fact schema differs")
        extended = np.column_stack((facts, new))
        extended_names = list(names) + new_names
        configs = {}
        for config, additions in CONFIGS.items():
            active_groups = dict(original["atomic_groups"])
            active_groups.update({group: list(GROUPS[group]) for group in additions})
            absent_names = set(new_names) - {
                name for group in additions for name in GROUPS[group]}
            excluded = EXCLUDED + tuple(absent_names)
            configs[config] = candidate(
                extended, labels, extended_names,
                split["train"], split["val"], seed, active_groups, excluded)
        prior = previous["runs"][seed]["configs"]["drop_upvote_and_comment_score"]
        current = configs["no_new_partner_facts"]
        if (current["validation"] != prior["validation"] or
                current["threshold"] != prior["threshold"] or
                current["reference_version"] != prior["reference_version"]):
            raise ValueError("Two-score-cue-free dev control failed to replay")
        all_runs.append({"seed": seed,
                         "split_sha256": source["split_sha256"],
                         "partner_fact_defined_counts": {
                             name: int(np.isfinite(new[:, j]).sum())
                             for j, name in enumerate(new_names)},
                         "configs": configs})
        print(seed, {key: val["validation"]["errors"]
                     for key, val in configs.items()}, flush=True)
    baseline = [run["configs"]["no_new_partner_facts"] for run in all_runs]
    eligible = []
    for name in CONFIGS:
        if name == "no_new_partner_facts":
            continue
        rows = [run["configs"][name] for run in all_runs]
        if (all(row["validation"]["errors"] <= base["validation"]["errors"]
                for row, base in zip(rows, baseline)) and
                np.mean([row["validation"]["auroc"] for row in rows]) >
                np.mean([base["validation"]["auroc"] for base in baseline])):
            eligible.append(name)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "BotSim_native_reply_partner_facts_two_score_cues_removed_dev_only",
        "fixed_test_predictions_or_metrics_accessed": False,
        "source_reply_trace_sha256": sha256_file(TRACE),
        "source_users_sha256": sha256_file(USERS),
        "source_activity_sha256": sha256_file(ACTIVITY),
        "excluded_score_facts": list(EXCLUDED),
        "new_fact_groups": {key: list(value) for key, value in GROUPS.items()},
        "dev_entry_gate": "each_seed_errors_no_worse_than_prior_two_cue_removed_and_mean_AUROC_strictly_higher",
        "eligible_for_further_evaluation": eligible,
        "runs": all_runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
