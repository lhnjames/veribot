"""Predict BotSim account IDs with source-defined averages only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.botsim_literal_zero_runtime import BotSimObservedMeanRuntime


def predict(root: Path, dataset: str, seed: int, account_ids: list[str]) -> dict:
    if dataset != "botsim" or seed not in (0, 1, 2):
        raise ValueError("observed-mean runtime requires BotSim and a frozen seed")
    root = Path(root).resolve()
    certificate_path = root / "research/certificates/botsim_observed_means_source_to_decision.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if (certificate["status"] !=
            "botsim_observed_means_full_raw_source_and_decisions_verified" or
            len(certificate["runs"]) != 3 or
            certificate["runs"][seed]["seed"] != seed):
        raise ValueError("independent observed-mean source audit is absent")
    runtime = BotSimObservedMeanRuntime(root, seed)
    if (certificate["raw_users_sha256"] != runtime.source_binding["raw_users_sha256"] or
            certificate["raw_activity_sha256"] !=
            runtime.source_binding["raw_activity_sha256"] or
            certificate["native_graph_facts_sha256"] !=
            runtime.source_binding["native_graph_facts_sha256"]):
        raise ValueError("raw source differs from independent observed-mean audit")
    return {
        "dataset": "botsim", "seed": seed,
        "source_level": "raw_user_post_comment_and_native_reply_graph",
        "source_binding": {
            **runtime.source_binding,
            "independent_decision_audit_sha256": sha256_file(certificate_path),
        },
        "classification_labels_accessed": False,
        "caller_supplied_fact_matrix_accepted": False,
        "decisions": runtime.predict_many(account_ids),
    }
