"""Separate the common sample-count shock from per-tweet deletion responses."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from lightgbm import Booster
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_compact20_capacity import GROUPS
from screen_twibot20_compact20_ab_group_tree_dev import design, source_groups
from screen_twibot20_compact20_ab_support_dev import evidence_subset
from train_v3_twibot20 import sha256
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
FACTS = ROOT / "data/processed/twibot20_v3/facts_split_visible_train_dev.npz"
REFERENCE = ROOT / "output/twibot20_compact20_ab_group_tree_dev/reference_split_visible_train_dev.json.gz"
COMPACT = ROOT / "research/certificates/twibot20_compact20_capacity.json"
SCREEN = ROOT / "research/certificates/twibot20_split_visible_graph_ab_dev.json"
MODEL = ROOT / "output/twibot20_compact20_ab_group_tree_dev/seed0_split_visible_train_dev_model.txt"
CARDS = ROOT / "research/certificates/twibot20_split_visible_graph_ab_dev_decisions.json"
ALL_EDITS = ROOT / "research/certificates/twibot20_split_visible_all_tweet_source_responses_dev.json"
DELTAS = ROOT / "research/certificates/twibot20_split_visible_all_tweet_source_response_deltas_dev.npz"
OUT = ROOT / "research/certificates/twibot20_split_visible_tweet_count_response_component.json"
EPS = 1e-12


def main() -> None:
    cards = json.loads(CARDS.read_text(encoding="utf-8"))
    all_edits = json.loads(ALL_EDITS.read_text(encoding="utf-8"))
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    names = json.loads(COMPACT.read_text(encoding="utf-8"))["retained_facts"]
    if (sha256(FACTS) != cards["fact_cache_sha256"]
            or sha256(REFERENCE) != cards["reference_artifact_sha256"]
            or sha256(MODEL) != cards["frozen_model_sha256"]
            or sha256(CARDS) != all_edits["cards_sha256"]
            or sha256(DELTAS) != all_edits["full_deltas_npz_sha256"]):
        raise ValueError("Frozen fact, model, card or source edit version differs")
    with np.load(FACTS, allow_pickle=False) as saved:
        split = saved["split"].astype(str)
        ids = saved["ids"].astype(str)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        all_names = saved["names"].astype(str).tolist()
    dev = np.flatnonzero(split == "dev")
    with np.load(DELTAS, allow_pickle=False) as saved:
        offsets = saved["offsets"].astype(int)
        deltas = saved["deletion_logit_delta"].astype(float)
        if saved["dev_ids"].astype(str).tolist() != ids[dev].tolist():
            raise ValueError("Dev response order differs")
    chosen = next(row for row in screen["rows"] if row["ab_seed"] == cards["a_seed"])
    signs = np.r_[[chosen["tail_directions_by_fact"][name]["upper"] for name in names],
                  [chosen["tail_directions_by_fact"][name]["lower"] for name in names]]
    reference = LabelBlindReference.load(REFERENCE)
    keep = [all_names.index(name) for name in names]
    count_col = all_names.index("tweet_sample_count")
    def encode(source: np.ndarray) -> np.ndarray:
        return design(evidence_subset(
            reference.transform(source, contexts=contexts[dev]), keep, names,
            source[:, count_col], True), signs)
    base = values[dev].copy()
    count_only = base.copy()
    nonempty = count_only[:, count_col] > 0
    count_only[nonempty, count_col] -= 1
    x0 = encode(base)
    xc = encode(count_only)
    group_cols = source_groups(names)[list(GROUPS).index("sampled_tweet_stats")]
    other_cols = [j for j in range(x0.shape[1]) if j not in group_cols]
    unchanged_error = float(np.max(np.abs(xc[:, other_cols] - x0[:, other_cols])))
    if unchanged_error > EPS:
        raise ValueError("Count-only analytic input changed another source group")
    model = Booster(model_file=str(MODEL))
    y0 = np.asarray(model.predict(x0, raw_score=True))
    yc = np.asarray(model.predict(xc, raw_score=True))
    card_rows = cards["decision_cards"]
    original_error = float(np.max(np.abs(y0 - [c["full_logit"] for c in card_rows])))
    if original_error > EPS:
        raise ValueError("Original frozen logits differ from cards")
    threshold = float(card_rows[0]["threshold"])
    rows = []
    for k, card in enumerate(card_rows):
        d = deltas[offsets[k]:offsets[k+1]]
        if len(d) != int(base[k, count_col]):
            raise ValueError("Source edit count differs from model input")
        common = float(yc[k] - y0[k])
        equal_count = int(np.count_nonzero(np.abs(d - common) <= EPS))
        original_class = bool(expit(y0[k]) >= threshold)
        count_flip = bool((expit(yc[k]) >= threshold) != original_class)
        full_flip = (expit(y0[k] + d) >= threshold) != original_class
        rows.append({
            "account_id": card["account_id"],
            "sampled_tweets": len(d),
            "count_only_analytic_logit_delta": common,
            "count_only_analytic_class_flip": count_flip,
            "full_deletions_equal_to_count_only_delta": equal_count,
            "nonzero_full_deletions_equal_to_count_only_delta": int(np.count_nonzero(
                (np.abs(d) > EPS) & (np.abs(d - common) <= EPS))),
            "nonzero_full_deletions_with_additional_statistic_response": int(
                np.count_nonzero((np.abs(d) > EPS) & (np.abs(d - common) > EPS))),
            "full_deletions_with_additional_statistic_response": len(d) - equal_count,
            "full_deletion_class_flips": int(np.count_nonzero(full_flip)),
            "full_deletion_class_flips_equal_to_count_only_delta": int(np.count_nonzero(
                full_flip & (np.abs(d - common) <= EPS))),
        })
    report = {
        "scope": "analytic count-only intermediate for every frozen sampled-tweet deletion",
        "facts_sha256": sha256(FACTS),
        "reference_sha256": sha256(REFERENCE),
        "model_sha256": sha256(MODEL),
        "cards_sha256": sha256(CARDS),
        "all_source_edits_sha256": sha256(ALL_EDITS),
        "all_source_deltas_sha256": sha256(DELTAS),
        "accounts_with_tweets": int(np.count_nonzero(nonempty)),
        "source_deletions": len(deltas),
        "accounts_count_only_analytic_class_flip": sum(
            row["count_only_analytic_class_flip"] for row in rows),
        "accounts_with_nonzero_count_only_logit_response": sum(
            abs(row["count_only_analytic_logit_delta"]) > EPS for row in rows),
        "source_deletions_equal_to_count_only_delta": sum(
            row["full_deletions_equal_to_count_only_delta"] for row in rows),
        "source_deletions_with_additional_statistic_response": sum(
            row["full_deletions_with_additional_statistic_response"] for row in rows),
        "nonzero_full_deletions_equal_to_count_only_delta": sum(
            row["nonzero_full_deletions_equal_to_count_only_delta"] for row in rows),
        "nonzero_full_deletions_with_additional_statistic_response": sum(
            row["nonzero_full_deletions_with_additional_statistic_response"]
            for row in rows),
        "class_changing_full_deletions_equal_to_count_only_delta": sum(
            row["full_deletion_class_flips_equal_to_count_only_delta"] for row in rows),
        "max_non_tweet_group_model_input_change": unchanged_error,
        "max_original_card_logit_error": original_error,
        "rows": rows,
        "interpretation": "Only tweet_sample_count is reduced in the analytic intermediate; tweet fractions and lengths remain at their original values. This is not a valid raw-record edit. Full deletion delta equals count-only delta plus a conditional additional-statistics response; the decomposition is order-dependent in a nonlinear model. Equality does not imply the text is semantically irrelevant in all contexts.",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
